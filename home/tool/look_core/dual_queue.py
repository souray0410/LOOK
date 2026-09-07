"""Two independent, single-visible-GPU cases; one authoritative queue writer."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

BUDGET_MIB = 14 * 1024


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def record(path):
    p = Path(path).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def verify(rec):
    if record(rec['path']) != rec:
        raise ValueError(f"Artifact changed: {rec['path']}")
    return Path(rec['path'])


def atomic(value, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.partial')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)


def validate_plan(plan):
    if plan.get('protocol') != 'independent_single_gpu_cases_v1' or plan.get('test_access') is not False:
        raise ValueError('A sealed explicit case plan is required')
    jobs = plan['jobs']; seen = set(); scientific = set()
    for job in jobs:
        name = job['id']
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', name) or name in seen:
            raise ValueError('Duplicate or unsafe case id')
        if job['kind'] not in ('method', 'suffix', 'acceptance'):
            raise ValueError('Unsupported case kind')
        if job['kind'] == 'acceptance' and not plan.get('acceptance_only', False):
            raise ValueError('Acceptance jobs cannot be scientific results')
        if plan.get('acceptance_only', False) and job['kind'] != 'acceptance':
            raise ValueError('Acceptance cannot launch fitting jobs')
        if not set(job.get('after', [])).issubset(seen):
            raise ValueError('Dependencies must reference earlier explicit jobs')
        if not isinstance(job['seed'], int):
            raise ValueError('Explicit integer seed required')
        if job['kind'] != 'acceptance':
            fingerprint = digest({k: v for k, v in job.items() if k not in ('id', 'after')})
            if fingerprint in scientific: raise ValueError('Duplicate scientific case')
            scientific.add(fingerprint)
        seen.add(name)
    return jobs


def ready_jobs(jobs, completed, active):
    return [j for j in jobs if j['id'] not in completed and j['id'] not in active
            and set(j.get('after', [])).issubset(completed)]


def worker_environment(gpu, project_root, micro, gpu_uuid=None):
    if gpu not in (0, 1):
        raise ValueError('Only physical GPU 0 and 1 are permitted')
    env = dict(os.environ)
    # Remove inherited distributed launch settings: each child is an independent case.
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=gpu_uuid or str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               LOOK_ASSIGNED_GPU_UUID=gpu_uuid or str(gpu),
               LOOK_PHYSICAL_GPU=str(gpu), LOOK_BOUNDED_WORKER='1',
               LOOK_EXECUTION_MICROBATCH=str(micro), LOOK_PROJECT_ROOT=str(project_root),
               PYTHONPATH=str(Path(project_root)/'tool'), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return env


def gpu_snapshot():
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    stats = {}
    for line in devices.splitlines():
        index, uuid, free = map(str.strip, line.split(','))
        stats[int(index)] = dict(uuid=uuid, free_mib=int(free), look_mib=0)
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    for line in rows.splitlines():
        uuid, pid, used = map(str.strip, line.split(','))
        p = Path('/proc')/pid
        try:
            args = (p/'cmdline').read_bytes().split(b'\0')[1:]
        except FileNotFoundError:
            continue  # A short-lived worker exited between NVML and /proc.
        is_look = any(b'/LOOK/' in a and b'/tool/environment/' not in a for a in args)
        if not is_look:
            try:
                is_look = b'LOOK_BOUNDED_WORKER=1' in (p/'environ').read_bytes().split(b'\0')
            except FileNotFoundError:
                continue
            except PermissionError:
                # Dying tasks can lose /proc/environ readability before NVML drops them.
                # Count unknown ownership conservatively; never stop another project.
                is_look = True
        if is_look:
            for value in stats.values():
                if value['uuid'] == uuid:
                    value['look_mib'] += int(used)
    return stats


def check_source(project_root):
    root = Path(project_root)
    manifest = read(root/'file_manifest.json')
    for row in manifest['files']:
        actual = record(root/row['path'])
        if actual['sha256'] != row['sha256'] or actual['bytes'] != row['bytes']:
            raise ValueError(f"Source manifest mismatch: {row['path']}")
    return record(root/'file_manifest.json')['sha256']


def validate_receipt(path, identity, job):
    rec = read(path)
    if (rec['identity'] != identity or rec['job_sha256'] != digest(job)
            or rec['status'] != 'complete' or rec['test_access'] is not False):
        raise ValueError('Cross-case or cross-source completion receipt')
    for artifact in rec['artifacts']:
        verify(artifact)
    return rec


def stop_process(process):
    # A process group includes data-loader children; never signal unrelated projects.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)


def supervise(project_root, plan_path, output, lock_path, *, snapshot=gpu_snapshot, poll=.25):
    root, output = Path(project_root).resolve(), Path(output).resolve()
    plan = read(plan_path); jobs = validate_plan(plan)
    source_hash = check_source(root)
    identity = dict(plan_sha256=digest(plan), source_manifest_sha256=source_hash)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = Path(lock_path); lock_path.parent.mkdir(parents=True, exist_ok=True)
    active = {}; completed = {}; attempts = {j['id']: 0 for j in jobs}
    state = dict(identity=identity, status='planned', test_access=False, gpu_devices=[0, 1],
                 max_concurrent_cases=2, gpu_budget_gib=14, completed={}, active={}, peaks_mib={},
                 acceptance_only=plan.get('acceptance_only', False))
    def save(status=None):
        if status: state['status'] = status
        state['completed'] = completed
        state['active'] = {str(g): dict(case_id=x['job']['id'], seed=x['job']['seed'],
                                      pid=x['process'].pid, microbatch=x['micro']) for g, x in active.items()}
        state['updated_at_unix'] = time.time()
        atomic(state, output/'queue_status.json')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ip = output/'queue_identity.json'
        if ip.exists() and read(ip) != identity:
            raise ValueError('Queue identity changed; use a new output directory')
        atomic(identity, ip)
        if jobs and not plan.get('acceptance_only', False):
            accepted = read(verify(plan['acceptance_record']))
            if (accepted.get('status') != 'complete' or accepted.get('test_access') is not False
                    or accepted.get('source_manifest_sha256') != source_hash
                    or accepted.get('devices') != [0, 1]
                    or accepted.get('parallel_cases_overlap') is not True
                    or accepted.get('prediction_equivalence') is not True
                    or accepted.get('single_visible_device_per_worker') is not True
                    or set(accepted.get('peaks_mib', {})) != {'0', '1'}
                    or any(v > BUDGET_MIB for v in accepted.get('peaks_mib', {}).values())):
                raise ValueError('Matching dual-GPU acceptance required before new fitting')
        for job in jobs:
            rp = output/'cases'/job['id']/'queue_complete.json'
            if rp.exists():
                validate_receipt(rp, identity, job); completed[job['id']] = record(rp)
        def interrupted(signum, frame):
            raise InterruptedError(f'Signal {signum}')
        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            while len(completed) < len(jobs):
                stats = snapshot()
                if not all(g in stats for g in (0, 1)):
                    raise RuntimeError('Both configured GPUs must exist')
                for gpu, value in stats.items():
                    state['peaks_mib'][str(gpu)] = max(state['peaks_mib'].get(str(gpu), 0), value['look_mib'])
                    if value['look_mib'] > BUDGET_MIB and gpu in active:
                        stop_process(active[gpu]['process']); active[gpu]['over_budget'] = True
                for gpu, run in list(active.items()):
                    code = run['process'].poll()
                    if code is None: continue
                    # Clean up a child that survived the main process, before releasing its slot.
                    stop_process(run['process']); run['log_handle'].close()
                    job = run['job']; del active[gpu]
                    receipt = output/'cases'/job['id']/'queue_complete.json'
                    if code == 0 and not run.get('over_budget'):
                        validate_receipt(receipt, identity, job); completed[job['id']] = record(receipt)
                        continue
                    tail = run['log'].read_text(errors='replace')[-12000:]
                    oom = run.get('over_budget') or 'OutOfMemoryError' in tail or 'CUDA out of memory' in tail
                    # Only SSF has an execution microbatch fallback; source inference batches are frozen.
                    retry = (oom and job['kind'] == 'method' and job['case']['method'] == 'ssf'
                             and attempts[job['id']] < 3)
                    if retry:
                        attempts[job['id']] += 1
                    else:
                        raise RuntimeError(f"Case {job['id']} stopped: {'memory budget/OOM' if oom else 'worker failure'} (exit={code})")
                pending = ready_jobs(jobs, completed, {r['job']['id'] for r in active.values()})
                for gpu in (0, 1):
                    if gpu in active or not pending: continue
                    value = stats[gpu]
                    if value['look_mib'] or value['free_mib'] < BUDGET_MIB: continue
                    job = pending.pop(0); micro = (8, 4, 2, 1)[attempts[job['id']]]
                    log = output/'logs'/f"{job['id']}_try{attempts[job['id']]}_{time.time_ns()}.log"
                    log.parent.mkdir(parents=True, exist_ok=True); handle = log.open('w')
                    argv = [sys.executable, str(root/'pipeline/46_run_dual_gpu_queue.py'),
                            '--project-root', str(root), '--plan', str(Path(plan_path).resolve()),
                            '--output', str(output), '--worker', job['id']]
                    process = subprocess.Popen(argv, env=worker_environment(gpu, root, micro, value.get('uuid')),
                                               stdout=handle, stderr=subprocess.STDOUT,
                                               start_new_session=True, pass_fds=(lock.fileno(),))
                    active[gpu] = dict(process=process, job=job, micro=micro, log=log, log_handle=handle)
                save('running' if active else 'waiting_resources')
                time.sleep(poll)
            save('complete')
        except BaseException as error:
            state['error'] = repr(error)
            for run in active.values():
                stop_process(run['process']); run['log_handle'].close()
            active.clear(); save('blocked')
            raise
        finally:
            for sig, handler in previous.items(): signal.signal(sig, handler)
    return state


def worker(project_root, plan_path, output, job_id):
    root, output = Path(project_root).resolve(), Path(output).resolve()
    plan = read(plan_path); jobs = validate_plan(plan)
    job = next(j for j in jobs if j['id'] == job_id)
    identity = dict(plan_sha256=digest(plan), source_manifest_sha256=check_source(root))
    if read(output/'queue_identity.json') != identity:
        raise ValueError('Parent and worker identities differ')
    out = output/'cases'/job_id; out.mkdir(parents=True, exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rp = out/'queue_complete.json'
        if rp.exists(): return validate_receipt(rp, identity, job)
        physical = os.environ['LOOK_PHYSICAL_GPU']
        if physical not in ('0', '1') or os.environ.get('CUDA_VISIBLE_DEVICES') != os.environ.get('LOOK_ASSIGNED_GPU_UUID'):
            raise ValueError('Worker must expose exactly its assigned physical GPU')
        import torch
        from .bounded_runtime import install_runtime, BudgetRunner
        if torch.cuda.device_count() != 1:
            raise ValueError('An independent case must see exactly one CUDA device')
        install_runtime()
        from .start_study import verify_source
        source = read(verify(job['source']))
        verify_source(source)
        if source['selection']['seed'] != job['seed']:
            raise ValueError('Source seed differs from explicit case')
        begun = time.time()
        if job['kind'] == 'acceptance':
            from .dual_queue_acceptance import probe
            result_path = probe(source, out, job['seed'])
        elif job['kind'] == 'method':
            from .method_study import make_method_cases, run_method_case
            from .self_input_study import make_cases
            spec = job['spec']; case = job['case']
            legal = make_cases(spec) if spec['protocol'].startswith('self_input_') else make_method_cases(spec)
            if case not in legal or case['seed'] != job['seed']:
                raise ValueError('Method case is outside frozen specification')
            if (source['selection']['fusion_position'] != case['fusion_position']
                    or source['selection']['filling_strategy'] != case['filling']):
                raise ValueError('Method source fusion/filling mismatch')
            run_method_case(source, case, spec, out/'result', torch.device('cuda:0'), (0,))
            result_path = out/'result'/'validation_result.json'
        else:
            from .start_study import sites_for_fusion
            case = job['case']; sites = sites_for_fusion(source['selection']['fusion_position'])
            start = case['start_ordinal']
            if (not 1 <= start <= len(sites) or case['candidate_sites'] != sites
                    or case['eligible_sites'] != sites[start-1:] or case['allowed_start'] != sites[start-1]):
                raise ValueError('Illegal correction start')
            runner = BudgetRunner(source, case, out/'result', out/'cache', torch.device('cuda:0'), (0,))
            runner.run(); result_path = runner.experiment_dir/'validation_result.json'
        result = read(result_path)
        if result.get('status') != 'complete' or result.get('test_access') is not False:
            # Historical LOOK result records encode the sealed boundary as phase/test.
            if not (job['kind'] == 'suffix' and result.get('status') == 'complete'
                    and result.get('phase') == 'validation' and not result.get('test')):
                raise ValueError('Incomplete or unsealed result')
        # Pin all produced model/config/prediction artifacts, including nested suffix banks.
        artifacts = [record(p) for p in sorted((out/'result').rglob('*'))
                     if p.is_file() and p.suffix in ('.json', '.npz', '.pt')]
        artifacts.extend([record(result_path), job['source']])
        for key in ('result', 'manifest', 'checkpoint', 'pca_manifest', 'labels', 'natural_labels'):
            rec = source[key]; artifacts.append({k: rec[k] for k in ('path', 'bytes', 'sha256')})
        for rec in [*source['pcas'], *source['generators'].values()]:
            artifacts.append({k: rec[k] for k in ('path', 'bytes', 'sha256')})
        for rec in result.get('provenance', []):
            # Normalize records to the queue's exact path/size/hash schema.
            verify({k: rec[k] for k in ('path', 'bytes', 'sha256')})
            artifacts.append({k: rec[k] for k in ('path', 'bytes', 'sha256')})
        value = dict(identity=identity, job_sha256=digest(job), status='complete', test_access=False,
                     physical_gpu=int(physical), physical_uuid=os.environ['LOOK_ASSIGNED_GPU_UUID'],
                     visible_devices=1, seed=job['seed'],
                     started_at_unix=begun, ended_at_unix=time.time(), artifacts=artifacts,
                     execution_microbatch=int(os.environ['LOOK_EXECUTION_MICROBATCH']))
        atomic(value, rp)
        return value
