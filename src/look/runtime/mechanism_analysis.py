"""One CPU-only final statistical job, independent of GPU allocation ownership."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from look.runtime.state import atomic_write_json,file_sha256


def read(p):return json.loads(Path(p).read_text())


def request(config):
    root=Path(config['output']);status=root/'analysis_request.json'
    if status.exists():return read(status)
    # Called under registry.lock, after every task has a verified terminal receipt.
    script=root/'analyze.sh'
    script.write_text('#!/bin/bash\nset -euo pipefail\nexport OMP_NUM_THREADS=8\nexport MKL_NUM_THREADS=8\n'+
        shlex.join([sys.executable,'-m','look.runtime.mechanism_analysis','--config',config['config_path'],'--execute'])+'\n')
    atomic_write_json(dict(state='submission_intent',script_sha256=file_sha256(script),time=time.time()),status)
    command=['sbatch','--parsable','--account=pi-mengy','--nodes=1','--ntasks=1','--cpus-per-task=8',
             '--mem=128G','--time=48:00:00','--job-name=look_mechanism_statistics',
             '--output='+str(root/'analysis_%j.log'),str(script)]
    try:
        result=subprocess.run(command,text=True,capture_output=True,timeout=60,check=True)
        job=result.stdout.strip().split(';')[0]
        if not job.isdigit():raise ValueError('Ambiguous CPU analysis submission')
        receipt=dict(state='submitted',job_id=job,command=command,time=time.time())
    except Exception as exc:receipt=dict(state='needs_review',reason=repr(exc),command=command,time=time.time())
    atomic_write_json(receipt,status);return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    c=read(a.config);root=Path(c['output'])
    if not a.execute:raise ValueError('Submission must be guarded by the registry lock')
    from look.analysis.mechanism_report import progress,aggregate
    feed=read(root/'queue.json');base=read(c['base_feed'])
    if not progress(feed,root/'report')['complete']:raise ValueError('Unresolved study cannot be analyzed')
    result=aggregate(feed,base,root/'report'/'statistics')
    atomic_write_json(dict(state='completed',result=result,time=time.time()),root/'analysis_status.json')

if __name__=='__main__':main()
