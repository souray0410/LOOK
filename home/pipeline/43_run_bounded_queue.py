#!/usr/bin/env python3
"""Serial supervised LOOK continuation; train/validation only, 14 GiB/device."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic(value, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def gpu_snapshot():
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    result={}
    for line in rows.splitlines():
        uuid,pid,used=map(str.strip,line.split(','));pid=int(pid)
        path=Path('/proc')/str(pid)/'cmdline'
        args=path.read_bytes().split(b'\0')[1:] if path.exists() else []
        is_look=any(b'/LOOK/' in a and b'/tool/environment/' not in a for a in args)
        # Dataloader children inherit the supervisor's environment marker.
        try: marked=b'LOOK_BOUNDED_WORKER=1' in (Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
        except (OSError,PermissionError):marked=False
        if is_look or marked:result[uuid]=result.get(uuid,0)+int(used)
    return result


def child(args):
    from look_core.bounded_runtime import install_runtime
    install_runtime()
    from look_core.paths import ProjectPaths
    from look_core.state import atomic_write_json
    from look_core.start_study import read_json, file_record, audit_case, verify_source
    paths=ProjectPaths.load(project_root=args.project_root)
    cfg=read_json(args.config);out=Path(cfg['output']);status={}
    if args.task=='acceptance':
        from look_core.bounded_acceptance import run
        run(cfg,out/'real_data')
    elif args.task=='drain':
        from look_core.bounded_runtime import BudgetRunner
        import torch
        cutoff=read_json(Path(cfg['cutover_audit']))
        before=read_json(Path(cfg['cutover_audit']).with_name('start_before.json'))
        name=cutoff['active_case'];done=out/'drain_complete.json'
        if done.exists():return
        if name in before['completed_cases']:
            record=before['completed_cases'][name]
            audit_case(record,Path(record['result']['path']))
        else:
            case=next(c for c in before['identity']['cases'] if c['case_id']==name)
            inventory=read_json(Path(cfg['historical_suffix']).with_name('source_inventory.json'))
            source=inventory['sources'][case['context']];verify_source(source)
            runner=BudgetRunner(source,case,out/'drain_runtime',out/'drain_cache',torch.device('cuda:0'),(0,1))
            runner.run()
            record=audit_case(case,runner.experiment_dir/'validation_result.json')
        atomic_write_json(dict(status='complete',test_access=False,case_id=name,record=record),done)
    elif args.task=='methods':
        from look_core.method_study import run_method_study
        status=run_method_study(paths,Path(cfg['parent_summary']),Path(cfg['parent_source']),None,(0,1),
            execute=True,render=False,start_evidence_manifest=Path(cfg['start_evidence_manifest']))
        atomic_write_json(dict(summary=str(Path(status['output'])/'summary.json')),out/'method_pointer.json')
    else:
        from look_core.self_input_study import run_self_input_study
        pointer=read_json(out/'method_pointer.json')
        status=run_self_input_study(paths,Path(pointer['summary']),(0,1),execute=True)
        atomic_write_json(dict(summary=str(Path(status['output'])/'summary.json')),out/'self_pointer.json')


def supervise(args):
    cfg=json.loads(args.config.read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    if cfg.get('test_access') is not False:raise ValueError('Sealed validation queue required')
    if not args.acceptance:
        rec=cfg['acceptance_record'];path=Path(rec['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=rec['sha256']:raise ValueError('Acceptance record changed')
        accepted=json.loads(path.read_text())
        if accepted.get('status')!='complete' or accepted.get('test_access') is not False:raise ValueError('Acceptance incomplete')
        manifest=json.loads((args.project_root/'file_manifest.json').read_text())
        for row in manifest['files']:
            path=args.project_root/row['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError(f'Deployed source changed: {path}')
    lock=Path('/data/mengh/LOOK/maintenance/bounded_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',LOOK_BOUNDED_WORKER='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='0',NCCL_CUMEM_ENABLE='0',NCCL_CUMEM_HOST_ENABLE='0')
    state=dict(status='preflight',gpu_devices=[0,1],execution_devices=[0],gpu_budget_gib=14,completed_tasks=[],test_access=False,peaks_mib={})
    state_path=out/'queue_status.json';process=None
    def interrupt(signum,frame):
        if process and process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        state.update(status='interrupted',signal=signum);atomic(state,state_path)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    for task in (['acceptance'] if args.acceptance else ['drain','methods','self']):
        for micro in [8,4,2,1]:
            while True:
                free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                if not gpu_snapshot() and all(int(v.strip())>=14*1024 for v in free.splitlines()[:2]):break
                state.update(status='waiting_resources',task=task);atomic(state,state_path);time.sleep(30)
            environment['LOOK_EXECUTION_MICROBATCH']=str(micro)
            log=out/f'{task}_micro{micro}_{int(time.time())}.log'
            state.update(status='running',task=task,micro_batch_size=micro,log=str(log));atomic(state,state_path)
            with log.open('w') as handle:
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--project-root',str(args.project_root),
                    '--config',str(args.config),'--task',task],env=environment,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                violation=False
                try:
                    while process.poll() is None:
                        snapshot=gpu_snapshot()
                        for uuid,used in snapshot.items():state['peaks_mib'][uuid]=max(state['peaks_mib'].get(uuid,0),used)
                        if any(used>14*1024 for used in snapshot.values()):
                            violation=True;os.killpg(process.pid,signal.SIGTERM);break
                        atomic(state,state_path);time.sleep(.5)
                except BaseException:
                    if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
                    raise
                try:code=process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    state.update(status='blocked',reason='worker did not exit');atomic(state,state_path);raise SystemExit(1)
            if code==0:
                state['completed_tasks'].append(task);break
            tail=log.read_text(errors='replace')[-12000:]
            oom=violation or 'OutOfMemoryError' in tail or 'CUDA out of memory' in tail
            if not oom or micro==1:
                state.update(status='blocked',reason='memory_budget' if oom else 'worker_failure',exit_code=code)
                atomic(state,state_path);raise SystemExit(1)
            state.update(status='retry_smaller_microbatch',reason='memory_budget');atomic(state,state_path)
        else:raise RuntimeError('Unreachable retry state')
    state.update(status='complete',task=None);atomic(state,state_path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--task',choices=['acceptance','drain','methods','self'])
    p.add_argument('--acceptance',action='store_true')
    args=p.parse_args()
    if args.task:child(args)
    else:supervise(args)


if __name__=='__main__':main()
