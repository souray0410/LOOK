"""Live device policy, independent of the frozen case plan and worker source."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from look_core.dual_queue import (atomic, read, record, verify, digest, check_source,
                                 validate_plan, validate_receipt, ready_jobs, stop_process)
BUDGET_MIB = 14 * 1024


def load_module(path):
    spec = importlib.util.spec_from_file_location('look_execution_' + Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_policy(policy, stats):
    if policy.get('schema') != 'look_devices_v1' or policy.get('mode') not in ('drain', 'interrupt'):
        raise ValueError('Expected look_devices_v1 and drain or interrupt')
    devices = policy['devices']
    if not isinstance(devices, list) or any(type(g) is not int or g < 0 for g in devices):
        raise ValueError('Device indices must be distinct nonnegative integers')
    if len(set(devices)) != len(devices) or any(g not in stats for g in devices):
        raise ValueError('Duplicate or unavailable GPU index')
    if policy.get('uuids') != {str(g):stats[g]['uuid'] for g in devices}:
        raise ValueError('Device UUID changed; explicitly select devices again')
    if policy.get('look_budget_mib') != BUDGET_MIB:
        raise ValueError('LOOK per-card budget must remain 14 GiB')
    return devices


def classify_process(pid, proc_root=Path('/proc')):
    """A shared Python executable is NOT evidence of LOOK ownership."""
    p = proc_root / str(pid)
    try:
        args = (p/'cmdline').read_bytes().split(b'\0')[1:]
    except (FileNotFoundError, PermissionError):
        return 'unknown'
    if any(b'/LOOK/' in a and b'/tool/environment/' not in a for a in args):
        return 'look'
    try:
        env = (p/'environ').read_bytes().split(b'\0')
        if b'LOOK_BOUNDED_WORKER=1' in env: return 'look'
        return 'other' if any(args) else 'unknown'
    except (FileNotFoundError, PermissionError):
        # Do not add unknown/external allocations to LOOK's project budget.
        return 'unknown'


def gpu_snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    stats = {}
    for line in raw.splitlines():
        index, uuid, free = map(str.strip, line.split(','))
        stats[int(index)] = dict(uuid=uuid, free_mib=int(free), look_mib=0,
                                 other_mib=0, unknown_mib=0, processes=[])
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    for line in raw.splitlines():
        uuid, pid, used = map(str.strip, line.split(','))
        owner = classify_process(int(pid))
        for value in stats.values():
            if value['uuid'] != uuid: continue
            if not used.isdecimal(): raise RuntimeError('NVML process memory unavailable')
            value[owner+'_mib'] += int(used)
            value['processes'].append(dict(pid=int(pid), owner=owner, mib=int(used)))
    return stats


def set_devices(path, devices, mode='drain', snapshot=gpu_snapshot):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stats = snapshot()
        policy = dict(schema='look_devices_v1', devices=devices, mode=mode,
                      uuids={str(g):stats[g]['uuid'] for g in devices if g in stats},
                      look_budget_mib=BUDGET_MIB, updated_at_unix=time.time())
        validate_policy(policy, stats)
        policy['revision'] = read(path)['revision'] + 1 if path.exists() else 1
        atomic(policy, path)
        return policy


def worker_environment(gpu, root, micro, uuid):
    if type(gpu) is not int or gpu < 0 or not uuid.startswith('GPU-'):
        raise ValueError('An explicit physical GPU and UUID are required')
    env = dict(os.environ)
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID', LOOK_ASSIGNED_GPU_UUID=uuid,
               LOOK_PHYSICAL_GPU=str(gpu), LOOK_BOUNDED_WORKER='1', LOOK_ALLOCATOR_GIB='11',
               LOOK_EXECUTION_MICROBATCH=str(micro), LOOK_PROJECT_ROOT=str(root),
               PYTHONPATH=str(root/'tool'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return env


def supervise(root, plan_path, output, lock_path, control, entrypoint, *, snapshot=gpu_snapshot, poll=.5, execution_acceptance=None):
    root, output, entrypoint = Path(root).resolve(), Path(output).resolve(), Path(entrypoint).resolve()
    plan = read(plan_path); jobs = validate_plan(plan)
    source_hash = check_source(root)
    identity = dict(plan_sha256=digest(plan), source_manifest_sha256=source_hash)
    output.mkdir(parents=True, exist_ok=True)
    active = {}; completed = {}; attempts = {j['id']:0 for j in jobs}
    state = dict(identity=identity, status='starting', test_access=False, completed={}, active={},
                 peaks_mib={}, execution='flexible_independent_cases_v1', gpu_budget_gib=14,
                 controller=record(__file__), adapter=record(Path(__file__).with_name('flexible_worker.py')),
                 entrypoint=record(entrypoint),
                 worker_root=str(root), control_path=str(Path(control).resolve()))
    def event(kind, **fields):
        with (output/'execution_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(at=time.time(),event=kind,**fields))+'\n')
            stream.flush(); os.fsync(stream.fileno())
    def save(status):
        state.update(status=status,completed=completed,attempts=attempts,updated_at_unix=time.time())
        state['active']={str(g):dict(case_id=r['job']['id'],pid=r['process'].pid,
                                   seed=r['job']['seed'],microbatch=r['micro'],uuid=r['uuid']) for g,r in active.items()}
        atomic(state,output/'queue_status.json')
    lock_path = Path(lock_path); lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        ip=output/'queue_identity.json'
        if ip.exists() and read(ip)!=identity: raise ValueError('Frozen queue identity mismatch')
        if not ip.exists(): atomic(identity,ip)
        if (output/'queue_status.json').exists():
            old = read(output/'queue_status.json')
            if old.get('identity') != identity: raise ValueError('Status identity mismatch')
            attempts.update(old.get('attempts',{}))
            atomic(old,output/'execution_history'/f'status_before_{time.time_ns()}.json')
        if not plan.get('acceptance_only',False):
            accepted=read(verify(plan['acceptance_record']))
            if (accepted.get('status')!='complete' or accepted.get('test_access') is not False
                    or accepted.get('source_manifest_sha256')!=source_hash
                    or accepted.get('prediction_equivalence') is not True
                    or accepted.get('single_visible_device_per_worker') is not True
                    or not accepted.get('peaks_mib')
                    or any(v>BUDGET_MIB for v in accepted['peaks_mib'].values())):
                raise ValueError('Matching scientific worker acceptance required')
            if execution_acceptance is None: raise ValueError('Execution acceptance required')
            verified=read(execution_acceptance)
            if (verified.get('status')!='complete' or verified.get('test_access') is not False
                    or verified.get('worker_manifest_sha256')!=source_hash
                    or verified.get('controller')!=state['controller']
                    or verified.get('adapter')!=state['adapter']
                    or verified.get('entrypoint')!=state['entrypoint']
                    or verified.get('prediction_equivalence') is not True):
                raise ValueError('Execution acceptance identity mismatch')
            state['execution_acceptance']=record(execution_acceptance)
        for job in jobs:
            rp=output/'cases'/job['id']/'queue_complete.json'
            if rp.exists():
                validate_receipt(rp,identity,job); completed[job['id']]=record(rp)
        event('controller_started',reused=list(completed),identity=identity,controller=state['controller'])
        def interrupted(signum, frame): raise InterruptedError(f'Signal {signum}')
        previous={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
        revision=None
        try:
            while len(completed)<len(jobs):
                stats=snapshot(); state['device_memory']=stats
                # Invalid changes block new dispatch, but allow owned cases to finish.
                policy_ok=True
                try:
                    policy=read(control); devices=validate_policy(policy,stats)
                    if revision != policy['revision']:
                        event('device_policy',policy=policy); revision=policy['revision']
                    state.pop('control_error',None)
                    state.update(gpu_devices=devices,max_concurrent_cases=len(devices),policy=policy)
                except (OSError,ValueError,KeyError,TypeError) as error:
                    policy_ok=False;devices=[];state['control_error']=repr(error)
                for gpu,run in list(active.items()):
                    value=stats.get(gpu)
                    if value is None or value['uuid'] != run['uuid']:
                        raise RuntimeError('An active GPU disappeared or changed identity')
                    state['peaks_mib'][str(gpu)]=max(state['peaks_mib'].get(str(gpu),0),value['look_mib'])
                    if value['look_mib']>BUDGET_MIB:
                        stop_process(run['process']);run['over_budget']=True
                    elif policy_ok and gpu not in devices and policy['mode']=='interrupt':
                        stop_process(run['process']);run['policy_interrupted']=True
                    code=run['process'].poll()
                    if code is None: continue
                    stop_process(run['process']);run['handle'].close();del active[gpu]
                    receipt=output/'cases'/run['job']['id']/'queue_complete.json'; job=run['job']
                    # Atomic valid completion wins a simultaneous device-removal request.
                    if receipt.exists():
                        validate_receipt(receipt,identity,job);completed[job['id']]=record(receipt)
                        event('completed',job=job['id'],gpu=gpu);continue
                    if run.get('policy_interrupted'):
                        event('interrupted_for_device_change',job=job['id'],gpu=gpu);continue
                    tail=run['log'].read_text(errors='replace')[-12000:]
                    oom=run.get('over_budget') or 'OutOfMemoryError' in tail or 'CUDA out of memory' in tail
                    if oom and job['kind']=='method' and job['case']['method']=='ssf' and attempts[job['id']]<3:
                        attempts[job['id']]+=1;event('microbatch_retry',job=job['id'],attempt=attempts[job['id']])
                    else:
                        raise RuntimeError(f"Case {job['id']} stopped: {'LOOK memory budget/OOM' if oom else 'worker failure'} (exit={code})")
                pending=ready_jobs(jobs,completed,{r['job']['id'] for r in active.values()})
                disk_ready=shutil.disk_usage(output).free>=plan.get('min_free_disk_gib',0)*1024**3
                if pending and not active and not disk_ready: raise RuntimeError('Insufficient disk headroom')
                for gpu in devices if policy_ok else []:
                    if gpu in active or not pending or not disk_ready:continue
                    value=stats[gpu]
                    # Other projects consume free memory, never LOOK's 14 GiB allowance.
                    # Unknown ownership prevents NEW dispatch only; it is not a kill criterion.
                    if value['look_mib'] or value.get('unknown_mib',0) or value['free_mib']<BUDGET_MIB:continue
                    for name in ('controller','adapter','entrypoint'):verify(state[name])
                    job=pending.pop(0);micro=(8,4,2,1)[attempts[job['id']]]
                    log=output/'logs'/f"{job['id']}_flex_{time.time_ns()}.log"
                    log.parent.mkdir(parents=True,exist_ok=True);handle=log.open('w')
                    argv=[sys.executable,str(entrypoint),'--worker-root',str(root),'--plan',str(Path(plan_path).resolve()),
                          '--output',str(output),'--worker',job['id']]
                    process=subprocess.Popen(argv,env=worker_environment(gpu,root,micro,value['uuid']),
                        stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
                    active[gpu]=dict(process=process,job=job,micro=micro,uuid=value['uuid'],log=log,handle=handle)
                    event('launched',job=job['id'],gpu=gpu,uuid=value['uuid'],pid=process.pid,microbatch=micro)
                save('running' if active else ('invalid_control' if not policy_ok else 'paused' if not devices else 'waiting_resources'))
                time.sleep(poll)
            save('complete');event('queue_complete')
        except BaseException as error:
            state['error']=repr(error)
            for run in active.values():stop_process(run['process']);run['handle'].close()
            active.clear();save('blocked');event('blocked',error=repr(error));raise
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
    return state
