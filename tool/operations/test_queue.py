"""Detached CPU dependency waiter and resource-bounded test scheduler."""
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from look_core.dual_queue import read, record, verify, atomic, digest, check_source, stop_process
from look_core.test_evaluation import dependency_status, freeze


def execution_module():
    path=Path(__file__).with_name('flexible_queue.py')
    spec=importlib.util.spec_from_file_location('test_execution',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def supervise(root, request_path, output, entrypoint):
    root,request_path,output=map(Path,(root,request_path,output))
    output.mkdir(parents=True,exist_ok=True)
    req=read(request_path);request_record=record(request_path);source_hash=check_source(root)
    execution=execution_module();active={};handles=[]
    identity=dict(request=request_record,source_manifest_sha256=source_hash)
    state=dict(status='waiting_validation',identity=identity,test_access=False,active={},completed={},peaks_mib={},
               control=req['control'],validation_queue=req['validation_queue'],look_budget_mib=14336,updated_at_unix=time.time())
    status_path=output/'queue_status.json'
    if status_path.exists():
        old=read(status_path)
        if old['identity']!=identity:raise ValueError('Test queue identity changed')
        state=old
    def save(status):
        if state.get('status')!=status:print(f'{time.time():.0f} test_queue_status={status}',flush=True)
        state.update(status=status,updated_at_unix=time.time(),active={str(g):dict(job_id=v['job']['id'],pid=v['process'].pid,microbatch=v['micro'],phase=v['phase']) for g,v in active.items()})
        atomic(state,status_path)
    def event(kind,**kw):
        with (output/'events.jsonl').open('a') as f:
            import json
            f.write(json.dumps(dict(at=time.time(),event=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())
    def interrupt(signum,frame):raise InterruptedError(f'Signal {signum}')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupt)
    with (output/'controller.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            event('queue_started',request=request_record,test_access=state['test_access'])
            print('Test follow-on queued; waiting for validation dependencies.',flush=True)
            while True:
                verify(request_record)
                predecessor=read(req['validation_queue'])
                state['validation_completed_jobs']=len(predecessor.get('completed',{}))
                state['validation_expected_jobs']=len(read(req['validation_plan']['path'])['jobs'])
                dep=dependency_status(req);save(dep)
                if dep=='ready':break
                # A failed predecessor stays blocked; this waiter never restarts it.
                time.sleep(30)
            # This is the same exclusive project execution lock held by validation.
            # Acquiring it is a prerequisite to replay or test dispatch.
            shared=Path(req['lock_path']).open('a');handles.append(shared)
            while True:
                try:fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:save('waiting_validation_lock');time.sleep(10)
            if dependency_status(req)!='ready':raise ValueError('Validation dependency regressed')
            save('freezing_configuration');manifest=freeze(request_path,output/'frozen_manifest.json',source_hash)
            frozen=record(output/'frozen_manifest.json');jobs=manifest['jobs'];event('configuration_frozen',manifest=frozen,jobs=len(jobs))
            for phase in ('validation_replay','test'):
                completed={};attempts={j['id']:0 for j in jobs}
                state.update(phase=phase,completed=completed,expected_jobs=len(jobs))
                if phase=='test':
                    from look_core.test_runtime import validate_gate
                    validate_gate(output/'frozen_manifest.json',output/'replay_gate.json')
                    # Declared before first loader construction, including partial/failed test attempts.
                    state['test_access']=True;save('running_test');event('test_opened',cohort='test',participants=290)
                for job in jobs:
                    p=output/phase/job['id']/'complete.json'
                    if p.exists():
                        r=read(p)
                        if r.get('identity')!=dict(frozen=frozen,job_sha256=digest(job),phase=phase) or r.get('status')!='complete':
                            raise ValueError('Cross-job/phase receipt')
                        for a in r['predictions'].values():verify(a)
                        completed[job['id']]=record(p)
                while len(completed)<len(jobs):
                    stats=execution.gpu_snapshot();state['device_memory']=stats
                    try:
                        policy=read(req['control']);devices=execution.validate_policy(policy,stats)
                        state.pop('control_error',None);state['policy']=policy
                    except (ValueError,KeyError,OSError) as e:
                        devices=[];policy=None;state['control_error']=repr(e)
                    for g,v in list(active.items()):
                        s=stats[g]
                        if s['uuid']!=v['uuid']:raise ValueError('Active GPU identity changed')
                        state['peaks_mib'][str(g)]=max(state['peaks_mib'].get(str(g),0),s['look_mib'])
                        if s['look_mib']>14336:stop_process(v['process']);v['over_budget']=True
                        elif policy and g not in devices and policy['mode']=='interrupt':
                            stop_process(v['process']);v['policy_interrupted']=True
                        code=v['process'].poll()
                        if code is None:continue
                        stop_process(v['process']);v['handle'].close();del active[g]
                        rp=output/phase/v['job']['id']/'complete.json'
                        if rp.exists() and not v.get('over_budget'):
                            r=read(rp)
                            if r['identity']!=dict(frozen=frozen,job_sha256=digest(v['job']),phase=phase) or r['status']!='complete':
                                raise ValueError('Invalid worker receipt')
                            for a in r['predictions'].values():verify(a)
                            completed[v['job']['id']]=record(rp);event('completed',phase=phase,job=v['job']['id']);continue
                        if v.get('policy_interrupted'):continue
                        tail=v['log'].read_text(errors='replace')[-20000:]
                        oom=v.get('over_budget') or 'OutOfMemoryError' in tail or 'CUDA out of memory' in tail
                        if oom and attempts[v['job']['id']]<3:
                            # Inference is deterministic and has no optimizer/RNG progress;
                            # replay the complete case at smaller batch, never skip samples.
                            if rp.exists():raise ValueError('A completed result exceeded the execution budget')
                            attempts[v['job']['id']]+=1;event('oom_retry',phase=phase,job=v['job']['id']);continue
                        raise RuntimeError(f'{phase} {v["job"]["id"]} failed, exit={code}; microbatch={v["micro"]}')
                    running={v['job']['id'] for v in active.values()}
                    pending=[j for j in jobs if j['id'] not in completed and j['id'] not in running]
                    for g in devices:
                        if g in active or not pending:continue
                        s=stats[g]
                        if s['look_mib'] or s['unknown_mib'] or s['free_mib']<14336:continue
                        verify(request_record);verify(frozen);check_source(root)
                        import shutil
                        if shutil.disk_usage(output).free<10*1024**3:raise RuntimeError('Insufficient output disk space')
                        job=pending.pop(0);micro=(8,4,2,1)[attempts[job['id']]]
                        log=output/'logs'/f'{phase}_{job["id"]}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
                        handle=log.open('w')
                        argv=[sys.executable,str(entrypoint),'--request',str(request_path),'--output',str(output),
                              '--worker',job['id'],'--phase',phase,'--microbatch',str(micro)]
                        proc=subprocess.Popen(argv,env=execution.worker_environment(g,root,micro,s['uuid']),stdout=handle,stderr=subprocess.STDOUT,
                            start_new_session=True,pass_fds=(own.fileno(),shared.fileno()))
                        active[g]=dict(process=proc,handle=handle,job=job,phase=phase,micro=micro,uuid=s['uuid'],log=log)
                        event('launched',phase=phase,job=job['id'],gpu=g,microbatch=micro)
                    state['completed']=completed;state['attempts']=attempts
                    save('running_'+phase if active else 'paused' if not devices else 'waiting_resources');time.sleep(.5)
                if phase=='validation_replay':
                    atomic(dict(status='complete',test_access=False,frozen_manifest=frozen,replays=completed),output/'replay_gate.json')
                    event('validation_replay_passed',jobs=len(completed))
            save('reporting')
            from look_core.test_report import report
            report(output)
            save('complete');event('complete',report=str(output/'report'/'REPORT.md'))
        except BaseException as e:
            for v in active.values():stop_process(v['process']);v['handle'].close()
            active.clear();state['error']=repr(e);save('blocked');event('blocked',error=repr(e));raise
        finally:
            for h in handles:h.close()
