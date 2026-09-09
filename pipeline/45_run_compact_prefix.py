#!/usr/bin/env python3
"""Audit nine existing starts, optionally replay one frozen validation-only path."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--start-evidence-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lock-path',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.worker:
        from look_core.prefix_diagnostic import run_gpu
        run_gpu(a.start_evidence_manifest,a.output);return
    # Verify immutable deployment before either CPU provenance or GPU replay.
    manifest=json.loads((a.project_root/'file_manifest.json').read_text())
    for rec in manifest['files']:
        if hashlib.sha256((a.project_root/rec['path']).read_bytes()).hexdigest()!=rec['sha256']:
            raise ValueError('Source manifest mismatch: '+rec['path'])
    from look_core.prefix_diagnostic import cpu_evidence
    cpu_evidence(a.start_evidence_manifest,a.output)
    if not a.execute:return
    if not a.lock_path:raise ValueError('--lock-path required for GPU replay')
    # Same global LOOK lock and NVML process-accounting implementation as queue 43.
    helpers=runpy.run_path(str(a.project_root/'pipeline/43_run_bounded_queue.py'))
    atomic=helpers['atomic']
    def snapshot():
        # Another project's short-lived GPU process may exit between /proc reads.
        for attempt in range(2):
            try:return helpers['gpu_snapshot']()
            except FileNotFoundError:
                if attempt:raise
    a.lock_path.parent.mkdir(parents=True,exist_ok=True)
    lock=a.lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='preflight',test_access=False,gpu_devices=[0,1],gpu_budget_gib=14,peaks_mib={})
    sp=a.output/'queue_status.json';proc=None
    def interrupt(signum,frame):
        if proc and proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
        state.update(status='interrupted');atomic(state,sp);raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    # Inference batch must not be reduced: preserve source cuDNN numerical behavior.
    # A failure stops this diagnostic; it never launches training or another case.
    while True:
        free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(free)>=2 and not snapshot() and all(int(v)>=14*1024 for v in free[:2]):break
        state.update(status='waiting_resources');atomic(state,sp);time.sleep(30)
    log=a.output/'replay.log'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',LOOK_BOUNDED_WORKER='1',LOOK_EXECUTION_MICROBATCH='8',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='0',NCCL_CUMEM_ENABLE='0',NCCL_CUMEM_HOST_ENABLE='0')
    cmd=[sys.executable,str(Path(__file__).resolve()),'--project-root',str(a.project_root),
         '--start-evidence-manifest',str(a.start_evidence_manifest),'--output',str(a.output),'--worker']
    state.update(status='running',log=str(log));atomic(state,sp)
    with log.open('a') as f:
        proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                used=snapshot()
                for gpu,mib in used.items():state['peaks_mib'][gpu]=max(state['peaks_mib'].get(gpu,0),mib)
                if any(mib>14*1024 for mib in used.values()):
                    os.killpg(proc.pid,signal.SIGTERM);state.update(reason='memory_budget');break
                atomic(state,sp);time.sleep(.5)
        except BaseException:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            raise
        try:code=proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            state.update(status='blocked',reason='worker_did_not_exit');atomic(state,sp);raise SystemExit(1)
    state.update(status='complete' if code==0 else 'blocked',exit_code=code);atomic(state,sp)
    if code:raise SystemExit(code)


if __name__=='__main__':main()
