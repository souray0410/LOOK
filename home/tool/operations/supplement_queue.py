"""CPU diagnostics immediately; bounded GPU replay after the existing test queue."""
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from look_core.dual_queue import read, record, verify, atomic, check_source, stop_process
from look_core.supplement_diagnostics import freeze, cpu_analysis, report


def predecessor_status(req):
    q=read(req['test_queue'])
    if q['identity']!=req['test_queue_identity']:raise ValueError('Test predecessor identity changed')
    if q['status']=='complete':
        if q.get('active') or len(q.get('completed',{}))!=87:raise ValueError('Incomplete predecessor receipt')
        return 'ready'
    return 'blocked_dependency' if q['status']=='blocked' else 'waiting_existing_queue'


def execution_module():
    path=Path(__file__).with_name('flexible_queue.py')
    spec=importlib.util.spec_from_file_location('supplement_execution',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def supervise(root, request_path, output, entrypoint):
    root,request_path,out=map(Path,(root,request_path,output));out.mkdir(parents=True,exist_ok=True)
    source=check_source(root);req=read(request_path);execution=execution_module();active={}
    state=dict(identity=dict(request=record(request_path),source_manifest_sha256=source),
               status='preparing_cpu',test_access=False,completed={},active={},attempts={},peaks_mib={})
    status=out/'queue_status.json'
    if status.exists():
        old=read(status)
        if old['identity']!=state['identity']:raise ValueError('Diagnostic queue identity changed')
        state=old
    def save(value):
        state.update(status=value,updated_at_unix=time.time(),active={str(g):dict(seed=v['job']['seed'],pid=v['process'].pid,microbatch=v['micro']) for g,v in active.items()})
        atomic(state,status)
    def interrupt(sig,frame):raise InterruptedError(f'Signal {sig}')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupt)
    with (out/'controller.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            plan=freeze(request_path,out,source);cpu_analysis(out);state['cpu']=record(out/'cpu_complete.json')
            print('CPU diagnostics complete; GPU replay waits for existing validation and test.',flush=True)
            while predecessor_status(req)!='ready':
                save(predecessor_status(req));time.sleep(30)
            with Path(req['lock_path']).open('a') as shared:
                while True:
                    try:fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:save('waiting_project_lock');time.sleep(10)
                if predecessor_status(req)!='ready':raise ValueError('Predecessor regressed')
                frozen=record(out/'diagnostic_manifest.json');jobs=plan['gpu_jobs']
                # Reconstruct durable completion; a stale status is not a receipt.
                state['completed']={}
                for job in jobs:
                    path=out/'gpu'/job['id']/'complete.json'
                    if path.exists():
                        value=read(path)
                        if value['identity']['manifest']!=frozen or value['identity']['seed']!=job['seed'] or value['acceptance_only']:
                            raise ValueError('Cross-seed or acceptance-only receipt')
                        for rec in value['outputs']:verify(rec)
                        state['completed'][job['id']]=record(path)
                while len(state['completed'])<3:
                    stats=execution.gpu_snapshot();state['device_memory']=stats
                    try:
                        policy=read(req['control']);devices=execution.validate_policy(policy,stats);state['policy']=policy
                        state.pop('control_error',None)
                    except (ValueError,KeyError,OSError) as e:
                        devices=[];policy=None;state['control_error']=repr(e)
                    for g,v in list(active.items()):
                        s=stats[g]
                        if s['uuid']!=v['uuid']:raise ValueError('Active device changed')
                        state['peaks_mib'][str(g)]=max(state['peaks_mib'].get(str(g),0),s['look_mib'])
                        if s['look_mib']>14336:stop_process(v['process']);v['over_budget']=True
                        elif policy and g not in devices and policy['mode']=='interrupt':
                            stop_process(v['process']);v['interrupted']=True
                        code=v['process'].poll()
                        if code is None:continue
                        stop_process(v['process']);v['handle'].close();del active[g]
                        path=out/'gpu'/v['job']['id']/'complete.json'
                        if path.exists() and not v.get('over_budget'):
                            value=read(path)
                            if value['identity']!=dict(manifest=frozen,seed=v['job']['seed'],microbatch=v['micro'],limit=None) or value['acceptance_only']:
                                raise ValueError('Invalid GPU completion')
                            for rec in value['outputs']:verify(rec)
                            state['completed'][v['job']['id']]=record(path);report(out);continue
                        if v.get('interrupted'):continue
                        tail=v['log'].read_text(errors='replace')[-16000:]
                        oom=v.get('over_budget') or 'OutOfMemoryError' in tail or 'CUDA out of memory' in tail
                        attempt=state['attempts'].get(v['job']['id'],0)
                        if oom and attempt<3 and not path.exists():
                            state['attempts'][v['job']['id']]=attempt+1;continue
                        raise RuntimeError(f'Diagnostic seed {v["job"]["seed"]} failed at batch {v["micro"]}; exit {code}')
                    running={v['job']['id'] for v in active.values()}
                    pending=[j for j in jobs if j['id'] not in state['completed'] and j['id'] not in running]
                    for g in devices:
                        if g in active or not pending:continue
                        s=stats[g]
                        if s['look_mib'] or s['unknown_mib'] or s['free_mib']<14336:continue
                        verify(state['identity']['request']);verify(frozen);check_source(root)
                        job=pending.pop(0);micro=(8,4,2,1)[state['attempts'].get(job['id'],0)]
                        log=out/'logs'/f'{job["id"]}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True);handle=log.open('w')
                        argv=[sys.executable,str(entrypoint),'--request',str(request_path),'--output',str(out),'--worker',str(job['seed']),'--microbatch',str(micro)]
                        proc=subprocess.Popen(argv,env=execution.worker_environment(g,root,micro,s['uuid']),stdout=handle,stderr=subprocess.STDOUT,
                            start_new_session=True,pass_fds=(own.fileno(),shared.fileno()))
                        active[g]=dict(job=job,process=proc,handle=handle,micro=micro,uuid=s['uuid'],log=log)
                    save('running_diagnostics' if active else 'waiting_resources');time.sleep(.5)
                report(out);save('complete')
        except BaseException as e:
            for v in active.values():stop_process(v['process']);v['handle'].close()
            active.clear();state['error']=repr(e);save('blocked');raise
