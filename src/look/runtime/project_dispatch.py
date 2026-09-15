"""Automatic real work dispatch for selected native replications and LOOK cases.

Shares the account submission lock and native run claims. Never cancels another
allocation, never steals a live claim, and never submits when no runnable work
exists. Each granted single-GPU allocation immediately executes useful work.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from look.runtime.state import atomic_write_json,file_sha256,stable_hash


def read(path,default=None):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)


def source_binding(spec,config):
    framework = config['framework_pythonpaths'][spec['framework']['commit']]
    if any(not (Path(framework)/'mhd_framework'/name).is_file() or
           file_sha256(Path(framework)/'mhd_framework'/name)!=digest
           for name,digest in spec['framework']['source_sha256'].items()):
        raise ValueError('Original native framework snapshot changed')
    for source in config['native_sources']:
        if all((Path(source)/name).is_file() and file_sha256(Path(source)/name)==digest
               for name,digest in spec['trainer_source_sha256'].items()):
            return dict(source=source,pythonpath=source+':'+framework+':'+config['dependency_pythonpath'])
    raise ValueError('No immutable source matches the selected native specification')


def work(config):
    """Fresh finite feeds; a model is never selected here from its performance."""
    result=[];feed=read(config['project_feed'])
    if feed:
        if feed.get('schema')!='look_project_work_feed_v1' or feed.get('test_access') is not False:raise ValueError('Unsealed project feed')
        result.extend(dict(t,execution='look') for t in feed['tasks'])
    for path in config.get('supplement_feeds',[]):
        supplement=read(path)
        if not supplement:continue
        if supplement.get('schema')!='look_mechanism_work_feed_v1' or supplement.get('test_access') is not False:
            raise ValueError('Unsealed supplement feed')
        result.extend(dict(t,execution='look_mechanism') for t in supplement['tasks'])
    for path in config.get('spatial_feeds',[]):
        spatial=read(path)
        if not spatial:continue
        if spatial.get('schema')!='look_spatial_work_feed_v1' or spatial.get('test_access') is not False:
            raise ValueError('Unsealed spatial feed')
        result.extend(dict(t,execution='look_spatial') for t in spatial['tasks'])
    for path in config.get('linear_feeds',[]):
        linear=read(path)
        if not linear:continue
        if linear.get('schema')!='look_linear_work_feed_v1' or linear.get('test_access') is not False:
            raise ValueError('Unsealed linear feed')
        result.extend(dict(t,execution='look_linear') for t in linear['tasks'])
    for path in config.get('terminal_feeds',[]):
        stage=read(path)
        if not stage:continue
        if stage.get('schema')!='look_terminal_work_feed_v1' or stage.get('test_access') is not False:
            raise ValueError('Unsealed terminal stage feed')
        result.extend(dict(t,execution='look_terminal') for t in stage['tasks'])
    native_paths=[config['native_feed']]+config.get('additional_native_feeds',[])
    for native_path in native_paths:
        native=read(native_path)
        if not native:continue
        if native.get('schema')!='look_native_work_feed_v1' or native.get('test_access') is not False:raise ValueError('Unsealed replication feed')
        for item in native['queues']:
            if file_sha256(Path(item['queue']))!=item['queue_sha256']:raise ValueError('Replication queue changed')
            for task in read(item['queue'])['tasks']:
                result.append(dict(task,execution='native'))
    unique={}
    for task in result:
        if file_sha256(Path(task['spec']))!=task['spec_sha256']:raise ValueError('Work specification changed')
        run=str(Path(task['run_dir']).resolve())
        if run in unique:
            previous=unique[run]
            if any(previous[k]!=task[k] for k in ('spec_sha256','execution')):
                raise ValueError('Conflicting duplicate work')
            continue  # Same immutable run may be referenced by several studies.
        unique[run]=task
    return list(unique.values())


def eligible(task,claims):
    state=read(claims.path(task['run_dir']));status=read(Path(task['run_dir'])/'status.json')
    if state.get('state') in ('claimed','running','failed','completed','liveness_needs_review'):return False
    if status.get('state') in ('needs_review','needs_review_epoch_cap','failed'):return False
    if (Path(task['run_dir'])/'accepted.json').exists():
        spec=read(task['spec'])
        if task['execution']=='native':
            from runtime.training_state import verify_completion
            verify_completion(task['run_dir'],spec)
        else:
            if task['execution']=='look_terminal':
                from look.studies.terminal_case import verify_case
            elif task['execution']=='look_linear':
                from look.studies.linear_case import verify_case
            elif task['execution']=='look_spatial':
                from look.studies.spatial_case import verify_case
            elif task['execution']=='look_mechanism':
                from look.studies.mechanism_case import verify_case
            else:
                from look.studies.project_case import verify_case
            verify_case(task['run_dir'],spec)
        return False
    return True


def native_api_preflight(task, config):
    """No torch/data/GPU allocation: reject incompatible probes before salloc."""
    spec = read(task['spec']); binding = source_binding(spec, config)
    script = Path(config['native_profile_source'])/'scheduling/profile_native.py'
    helper = script.with_name('native_profile_api.py')
    key = (task['spec_sha256'], binding['source'], file_sha256(script), file_sha256(helper))
    cache = config.setdefault('_native_api_receipts', {})
    if key not in cache:
        env = os.environ.copy(); env['PYTHONPATH'] = binding['pythonpath']
        result = subprocess.run([config['python'], str(script), '--source', binding['source'],
            '--spec', task['spec'], '--check-api'], env=env, text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise ValueError('Native API preflight rejected: '+result.stderr[-2000:])
        value = json.loads(result.stdout)
        if value.get('test_access') is not False or not value.get('adapter'):
            raise ValueError('Malformed API preflight receipt')
        cache[key] = value
    return cache[key]


def admissible_work(config, claims):
    result = []; rejected = []
    for task in work(config):
        if not eligible(task, claims): continue
        if task.get('execution') == 'native':
            try: native_api_preflight(task, config)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                rejected.append(dict(run=task['run_dir'], error=str(exc))); continue
        result.append(task)
    if config.get('output'):
        atomic_write_json(dict(rejected=rejected, ready=len(result), updated_at=time.time(),
                               test_access=False), Path(config['output'])/'api_admission.json')
    return result


def worker_environment(environ):
    env = dict(environ)
    # The lightweight owner uses one CPU; the child step explicitly requests14.
    for key in ('SLURM_CPUS_PER_TASK', 'SLURM_TRES_PER_TASK', 'SLURM_MEM_PER_CPU', 'SLURM_MEM_PER_NODE'):
        env.pop(key, None)
    return env


def exited_step(job, step, presence, attempts=30, sleep=time.sleep):
    if step is None: return False
    for index in range(attempts):
        try: observed = presence(job, step)
        except (OSError, subprocess.SubprocessError): observed = None
        if observed is False: return True
        if index + 1 < attempts: sleep(2)
    return False


def allocation_command(config_path,name,python,memory_gib=128):
    if memory_gib not in (128,512):raise ValueError("Unregistered allocation memory class")
    return ['salloc','--account=pi-mengy','--nodes=1','--ntasks=1','--cpus-per-task=16',
        '--mem='+str(memory_gib)+'G','--gres=gpu:a100:1','--constraint=gpu_a100','--time=48:00:00',
        '--job-name='+name,python,'-m','look.runtime.project_dispatch','--config',str(config_path),'--allocation-owner']



def failed_before_submission(record):
    return (record.get('state')=='owner_exited' and not record.get('job_id') and
        Path(record.get('log','/nonexistent')).is_file() and
        Path(record['log']).read_text().strip()=='salloc: error: no controlling terminal: please set --no-shell')


def reconcile_expired(config):
    from scheduling.policy import Claims
    from scheduling.quota_guard import snapshot
    claims=Claims(config['claims']);snap=snapshot();events=[]
    for task in work(config):
        run=Path(task['run_dir']);record=read(claims.path(run))
        if not str(record.get('owner','')).startswith('look-workflow-'):continue
        if record.get('state') not in ('running','claimed','liveness_needs_review'):continue
        job=str(record['job_id'])
        if job in snap['jobs']:continue
        # Accounting and the control plane must agree; age alone grants nothing.
        text=subprocess.check_output(['sacct','-X','-nP','-j',job,'--format=JobIDRaw,State'],text=True,timeout=20)
        rows=[r.split('|') for r in text.splitlines() if r.split('|')[0]==job]
        if len(rows)!=1:continue
        state=rows[0][1].split()[0]
        if state not in ('TIMEOUT','PREEMPTED','NODE_FAIL'):continue
        checkpoint=run/('last.pt' if task['execution']=='native' else 'training/last.pt' if task['execution']=='look_mechanism' else 'host/last.pt')
        if not checkpoint.is_file():
            if task['execution'] not in ('look_mechanism','look_spatial','look_linear','look_terminal'):continue
            # Fitting can restart at its last accepted site; prior process death
            # has been established above, never inferred from heartbeat age.
            checkpoint=Path(task['spec'])
        claims.release(run,record['owner'],'paused',step_dead=True)
        event=dict(run=str(run),allocation=job,terminal_state=state,checkpoint_sha256=file_sha256(checkpoint),
                   next_step='full_resource_resume_preflight_before_formal_continuation')
        events.append(event)
    if events:atomic_write_json(dict(events=events,time=time.time()),Path(config['output'])/('recovery_'+str(time.time_ns())+'.json'))
    return events

def submit_one(config,path,journal):
    from scheduling.quota_guard import snapshot
    from scheduling.renewal import job_from_log
    from scheduling.policy import Claims
    claims=Claims(config['claims'])
    if config.get('output') and (Path(config['output'])/'admission_hold.json').exists():return 'waiting_incident_repair'
    candidates=admissible_work(config,claims)
    if not candidates:return 'waiting_dependencies'
    lock=Path(config['account_submission_lock'])
    with lock.open('a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return 'waiting_submission_lock'
        snap=snapshot()
        # Include every existing allocation and durable unconfirmed request.
        for entry in journal['requests']:
            if not entry.get('job_id'):return 'submission_identity_needs_review'
        for pattern in config['submission_journal_globs']:
            import glob
            for prior in glob.glob(pattern,recursive=True):
                value=read(prior)
                if not isinstance(value.get('requests'),list):raise ValueError('Unknown prior journal')
                if any(not str(r.get('job_id','')).isdigit() and not failed_before_submission(r) for r in value['requests']):
                    return 'prior_submission_identity_needs_review'
        active=sum(str(e['job_id']) in snap['jobs'] for e in journal['requests'])
        from scheduling.project_priority import project_limit
        maximum=project_limit(config,config['maximum_workflow_allocations'],'LOOK')
        if active>=maximum:return 'workflow_allocations_active'
        if snap['total_gpus']>=snap['limit']:return 'waiting_account_capacity'
        name='look_auto_'+str(time.time_ns());log=Path(config['output'])/(name+'.log')
        entry=dict(name=name,state='intent',log=str(log),time=time.time())
        journal['requests'].append(entry);atomic_write_json(journal,Path(config['output'])/'requests.json')
        memory_gib=512 if any(t['execution'] in ('look_spatial','look_linear') for t in candidates) else 128
        command=allocation_command(path,name,config['python'],memory_gib);entry['command']=command
        with log.open('x') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
        entry.update(pid=child.pid,host=socket.gethostname(),state='submitted_waiting_identity')
        atomic_write_json(journal,Path(config['output'])/'requests.json')
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            job=job_from_log(log.read_text())
            if job:
                entry.update(job_id=job,state='submitted');break
            if child.poll() is not None:
                entry.update(state='identity_needs_review',returncode=child.returncode);break
            time.sleep(.5)
        atomic_write_json(journal,Path(config['output'])/'requests.json')
    return entry['state']


def daemon(config_path):
    config=read(config_path);out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    if not 1<=config['maximum_workflow_allocations']<=2:raise ValueError('Workflow allocation limit')
    with open('/dev/tty','rb'):pass
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    with (out/'dispatcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=read(out/'requests.json',dict(schema='look_workflow_requests_v1',requests=[]))
        while not (out/'stop.json').exists():
            try:
                from look.runtime.mechanism_handover import retire_parent
                retire_parent(out)
                if config.get('session_guard_receipts'):
                    from scheduling.project_priority import verify_session_guards
                    verify_session_guards(config)
                reconcile_expired(config)
                state=submit_one(config,config_path,journal);error=None
            except Exception as exc:state='needs_review';error=repr(exc)
            atomic_write_json(dict(state=state,error=error,updated_at=time.time(),pid=os.getpid(),test_access=False),out/'status.json')
            time.sleep(60)


def allocation_owner(config_path):
    config=read(config_path);job=os.environ['SLURM_JOB_ID']
    text=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,timeout=20)
    fields=dict(x.split('=',1) for x in text.split() if '=' in x)
    end=datetime.fromisoformat(fields['EndTime']).timestamp()
    environment=os.environ.copy();environment['LOOK_ALLOCATION_END']=str(end)
    tres=dict(x.split('=',1) for x in fields.get('AllocTRES',fields.get('ReqTRES','')).split(',') if '=' in x)
    memory=tres.get('mem','0M')
    unit=memory[-1];amount=float(memory[:-1])
    environment['LOOK_ALLOCATION_MEMORY_GIB']=str(amount*({'M':1/1024,'G':1,'T':1024}[unit]))
    subprocess.run(['srun','--jobid='+job,'--overlap','--exact','--nodes=1','--ntasks=1','--gpus=1',
        '--cpus-per-task=1','--mem=2G','--unbuffered',config['python'],'-m','look.runtime.project_dispatch',
        '--config',str(config_path),'--gpu-owner'],env=environment,check=True)


def gpu_owner(config_path):
    import torch
    from scheduling.policy import Claims
    from scheduling.slurm_liveness import step_presence
    config=read(config_path);job=os.environ['SLURM_JOB_ID'];end=float(os.environ['LOOK_ALLOCATION_END'])
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_properties(0).total_memory<78*1024**3:raise ValueError('A10080 single-device binding required')
    root=Path(config['output'])/job;root.mkdir(exist_ok=True);claims=Claims(config['claims']);owner='look-workflow-'+job
    while time.time()<end-1800:
        candidates=admissible_work(config,claims)
        candidates=[t for t in candidates if t['execution'] not in ('look_spatial','look_linear') or float(os.environ.get('LOOK_ALLOCATION_MEMORY_GIB','0'))>=512]
        if not candidates:break
        prior=read(root/'last_execution.json').get('kind')
        primary=[t for t in candidates if t['execution']=='look']
        supplement=[t for t in candidates if t['execution'] in ('look_mechanism','look_spatial','look_linear','look_terminal')]
        if supplement and (not primary or prior=='look' or any(t['execution']=='look_terminal' for t in supplement)):
            # Neural continuations first, then controls, then sample curves.
            priority={'host_training':0,'student_training':1,'correction':2,'sample_curve':3}
            supplement.sort(key=lambda t: -2 if t['execution']=='look_terminal' else -1 if t['execution'] in ('look_spatial','look_linear') else priority[read(t['spec'])['task']['kind']])
            task=supplement[0]
        else:task=primary[0] if primary else candidates[0]
        atomic_write_json(dict(kind=task['execution']),root/'last_execution.json')
        run=Path(task['run_dir']);spec=read(task['spec'])
        try:token=claims.acquire(run,task['spec_sha256'],owner,job)
        except RuntimeError:continue
        attempt=root/(run.name+'_'+str(token['generation']));attempt.mkdir()
        record=attempt/'step.json';environment=worker_environment(os.environ)
        command=['srun','--jobid='+job,'--overlap','--exact','--nodes=1','--ntasks=1','--gpus=1',
            '--cpus-per-task=14','--mem='+('384G' if task['execution'] in ('look_spatial','look_linear') else '100G'),'--unbuffered',config['python'],'-m','look.runtime.project_dispatch',
            '--config',str(config_path),'--execute',task['spec'],'--run',str(run),
            '--kind',task['execution'],'--record',str(record)]
        (run/'pause.json').unlink(missing_ok=True)
        with (attempt/'worker.log').open('x') as log:
            child=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
        step=None
        priority=None
        if config.get('project_priority_enabled'):
            from scheduling.project_priority import PriorityProbe
            priority=PriorityProbe(config_path,config,job,attempt,task,'look')
        while child.poll() is None:
            r=read(record)
            if r.get('step') and step is None:
                step=r['step'];claims.update(run,owner,state='running',step=step)
            if priority is not None:
                try:priority.poll([t for t in work(config) if t['execution'] not in ('native','look_spatial','look_linear') and eligible(t,claims)])
                except Exception as exc:
                    atomic_write_json(dict(state='priority_deferred',error=repr(exc)),attempt/'priority_error.json')
            if time.time()>end-900:
                atomic_write_json(dict(reason='allocation_expiry_checkpoint'),run/'pause.json')
            atomic_write_json(dict(state='running',task=task['id'],run=str(run),step=step,updated_at=time.time()),root/'status.json')
            time.sleep(10)
        if priority is not None:priority.finish()
        # A terminated client is not proof that its remote step is dead.
        step = step or read(record).get('step')  # Worker may exit before the first poll.
        if not exited_step(job,step,step_presence):
            claims.update(run,owner,state='liveness_needs_review');raise RuntimeError('Cannot prove worker step exit')
        if child.returncode not in (0, 75):
            claims.release(run,owner,'failed',step_dead=True)
            atomic_write_json(dict(reason='worker_failed', run=str(run), job=job, step=step,
                returncode=child.returncode, log=str(attempt/'worker.log'), time=time.time()),
                Path(config['output'])/'admission_hold.json')
            break  # No retry of a deterministic probe failure or fresh allocation loop.
        state=read(run/'status.json').get('state')
        if state=='completed' or (state=='infeasible' and task['execution']=='look_mechanism'):
            if task['execution']=='native':
                from runtime.training_state import verify_completion
                verify_completion(run,spec)
            else:
                if task['execution']=='look_terminal':
                    from look.studies.terminal_case import verify_case
                elif task['execution']=='look_linear':
                    from look.studies.linear_case import verify_case
                elif task['execution']=='look_spatial':
                    from look.studies.spatial_case import verify_case
                elif task['execution']=='look_mechanism':
                    from look.studies.mechanism_case import verify_case
                else:
                    from look.studies.project_case import verify_case
                verify_case(run,spec)
            claims.release(run,owner,'completed',step_dead=True)
        elif state=='paused':claims.release(run,owner,'paused',step_dead=True)
        else:claims.release(run,owner,'failed',step_dead=True)
    atomic_write_json(dict(state='owner_finished',updated_at=time.time()),root/'status.json')


def execute_work(config_path,spec_path,run,kind,record):
    import torch
    config=read(config_path);spec=read(spec_path);run=Path(run)
    if kind in ('look_mechanism','look_spatial','look_linear','look_terminal'):
        # Each scientific task executes its own immutable pin, including when a
        # newer management overlay dispatches an older accepted study.
        import look
        roots={str(Path(row['path'].split('/src/look/')[0])/'src')
               for row in spec.get('source_pins',[]) if '/src/look/' in row['path']}
        if len(roots)!=1:raise ValueError('Ambiguous pinned scientific package')
        source=next(iter(roots))
        if Path(look.__file__).resolve().parent!=Path(source)/'look':
            env=os.environ.copy();env['PYTHONPATH']=source+':'+env.get('PYTHONPATH','')
            os.execvpe(config['python'],[config['python'],'-m','look.runtime.project_dispatch',
                '--config',str(config_path),'--execute',str(spec_path),'--run',str(run),
                '--kind',kind,'--record',str(record)],env)
    step=dict(job_id=os.environ['SLURM_JOB_ID'],step=os.environ['SLURM_STEP_ID'],pid=os.getpid(),run=str(run),time=time.time())
    atomic_write_json(step,Path(record))
    profile_root=Path(record).parent/'profile'
    environment=os.environ.copy()
    environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if kind=='look' and config.get('legacy_project_pythonpath'):
        environment['PYTHONPATH']=config['legacy_project_pythonpath']
        os.execvpe(config['python'],[config['python'],'-m','look.runtime.project_dispatch','--config',str(config_path),
            '--execute',str(spec_path),'--run',str(run),'--kind','look','--record',str(record)],environment)
    if kind=='native':
        binding=source_binding(spec,config)
        environment['PYTHONPATH']=binding['pythonpath']
        command=[config['python'],str(Path(config['native_profile_source'])/'scheduling/profile_native.py'),
            '--source',binding['source'],'--spec',str(spec_path),'--output',str(profile_root),
            '--full-development','--full-train-read','--fresh-train-batches']
        if (run/'last.pt').exists(): command.extend(['--checkpoint',str(run/'last.pt')])
        subprocess.run(command,env=environment,check=True)
        from scheduling.prepared_owner import complete_profile
        receipt=read(profile_root/'accepted.json')
        if not complete_profile(receipt):raise ValueError('Native full resource profile rejected')
        # A single exclusive workflow worker; the remaining allocation RAM is reserved.
        if receipt.get('peak_gpu_gib',float('inf'))*1.2+2>70:raise ValueError('Native GPU reserve failed')
        command=[config['python'],'-m','expanded.native','--spec',str(spec_path),'--output',str(run),'--mode','train']
        os.execvpe(config['python'],command,environment)
    elif kind=='look_terminal':
        environment['LOOK_WORKER_MEMORY_BYTES']=str(100*1024**3)
        subprocess.run([config['python'],'-m','look.studies.terminal_case','--spec',str(spec_path),
            '--output',str(profile_root),'--profile'],env=environment,check=True)
        receipt_path=profile_root/'profile_accepted.json';receipt=read(receipt_path)
        if receipt.get('state')!='accepted' or receipt.get('identity')!=stable_hash(spec):
            raise ValueError('Terminal resource preflight failed')
        environment['LOOK_TERMINAL_PROFILE_RECEIPT']=str(receipt_path)
        os.execvpe(config['python'],[config['python'],'-m','look.studies.terminal_case',
            '--spec',str(spec_path),'--output',str(run)],environment)
    elif kind=='look_linear':
        environment['LOOK_WORKER_MEMORY_BYTES']=str(384*1024**3)
        subprocess.run([config['python'],'-m','look.studies.linear_case','--spec',str(spec_path),
            '--output',str(profile_root),'--profile'],env=environment,check=True)
        receipt_path=profile_root/'profile_accepted.json';receipt=read(receipt_path)
        if receipt.get('state')!='accepted' or receipt.get('identity')!=stable_hash(spec):
            raise ValueError('Linear full-path resource receipt mismatch')
        environment['LOOK_LINEAR_PROFILE_RECEIPT']=str(receipt_path)
        os.execvpe(config['python'],[config['python'],'-m','look.studies.linear_case',
            '--spec',str(spec_path),'--output',str(run)],environment)
    elif kind=='look_spatial':
        environment['LOOK_WORKER_MEMORY_BYTES']=str(384*1024**3)
        subprocess.run([config['python'],'-m','look.studies.spatial_case','--spec',str(spec_path),
            '--output',str(profile_root),'--profile'],env=environment,check=True)
        receipt=read(profile_root/'profile_accepted.json')
        if receipt.get('state')!='accepted' or receipt.get('identity')!=stable_hash(spec):
            raise ValueError('Spatial full-path resource receipt mismatch')
        os.execvpe(config['python'],[config['python'],'-m','look.studies.spatial_case',
            '--spec',str(spec_path),'--output',str(run)],environment)
    elif kind=='look_mechanism':
        environment['LOOK_WORKER_MEMORY_BYTES']=str(100*1024**3)
        subprocess.run([config['python'],'-m','look.runtime.project_dispatch','--config',str(config_path),
            '--profile-mechanism',str(spec_path),'--run',str(profile_root)],env=environment,check=True)
        receipt=read(profile_root/'accepted.json')
        if receipt.get('status')!='accepted' or receipt.get('case_identity')!=stable_hash(spec):
            raise ValueError('Supplement resource receipt mismatch')
        # Exec releases every profile allocation before formal loading.
        os.execvpe(config['python'],[config['python'],'-m','look.studies.mechanism_case',
            '--spec',str(spec_path),'--output',str(run)],environment)
    else:
        # Profiling runs in a subprocess so its optimizer, CUDA cache and limits
        # cannot leak into the formal model's RNG or memory accounting.
        subprocess.run([config['python'],'-m','look.runtime.project_dispatch','--config',str(config_path),
            '--profile-project',str(spec_path),'--run',str(profile_root)],env=environment,check=True)
        receipt=read(profile_root/'accepted.json')
        if receipt.get('status')!='accepted' or receipt.get('case_identity')!=stable_hash(spec):raise ValueError('Project resource receipt mismatch')
        from look.studies.project_case import execute
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction((min(.875*total,total-10*1024**3)-2*1024**3)/total)
        execute(spec,run)


def standby(config_path):
    config=read(config_path);out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    # Verify imports, finite feeds and source pins before the old CPU dispatcher
    # is retired. No allocation or claim is touched in this phase.
    from scheduling.policy import Claims
    from look.studies.mechanism_case import verify_dependencies
    work(config)
    for row in config['source_pins']:
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('Dispatcher snapshot changed')
    atomic_write_json(dict(state='ready',pid=os.getpid(),host=socket.gethostname(),config=str(config_path),time=time.time()),out/'handover_ready.json')
    while not (out/'handover_armed.json').exists():time.sleep(2)
    daemon(config_path)


def configure_control_imports(config):
    # Add only new management modules; keep expanded.native and scientific data
    # adapters bound to the existing immutable dependency source.
    if config.get('control_source'):
        import scheduling
        control=str(Path(config['control_source'])/'scheduling')
        for row in config.get('control_source_pins',[]):
            if file_sha256(Path(row['path']))!=row['sha256']:raise ValueError('Management snapshot changed')
        if control not in scheduling.__path__:scheduling.__path__.append(control)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--standby',action='store_true');p.add_argument('--allocation-owner',action='store_true');p.add_argument('--gpu-owner',action='store_true')
    p.add_argument('--execute');p.add_argument('--kind',choices=['native','look','look_mechanism','look_spatial','look_linear','look_terminal']);p.add_argument('--record');p.add_argument('--run');p.add_argument('--profile-project');p.add_argument('--profile-mechanism')
    a=p.parse_args()
    configure_control_imports(read(a.config))
    if a.standby:standby(a.config)
    elif a.allocation_owner:allocation_owner(a.config)
    elif a.gpu_owner:gpu_owner(a.config)
    elif a.execute:execute_work(a.config,a.execute,a.run,a.kind,a.record)
    elif a.profile_mechanism:
        import torch
        from look.studies.mechanism_profile import profile
        profile(read(a.profile_mechanism),a.run,torch.device('cuda:0'))
    elif a.profile_project:
        import torch
        from look.studies.project_profile import profile
        profile(read(a.profile_project),a.run,torch.device('cuda:0'))
    else:daemon(a.config)

if __name__=='__main__':main()
