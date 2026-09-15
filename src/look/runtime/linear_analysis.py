"""Finite CPU Slurm analysis, invoked under the linear registry lock."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from look.runtime.state import atomic_write_json,file_sha256


def request(config,runs,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);status=out/'request.json'
    if status.exists():return json.loads(status.read_text())
    manifest=out/'analysis.json'
    atomic_write_json(dict(runs=[str(p) for p in runs],output=str(out),test_access=False),manifest)
    script=out/'analyze.sh'
    script.write_text('#!/bin/bash\nset -euo pipefail\nmodule load python/3.11\nexport OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8\n'+
        'export PYTHONPATH='+shlex.quote(config['pythonpath'])+'\n'+
        shlex.join([sys.executable,'-m','look.runtime.linear_analysis','--manifest',str(manifest)])+'\n')
    atomic_write_json(dict(state='submission_intent',script_sha256=file_sha256(script),time=time.time()),status)
    command=['sbatch','--parsable','--account=pi-mengy','--nodes=1','--ntasks=1','--cpus-per-task=8',
        '--mem=64G','--time=48:00:00','--job-name=look_linear_statistics','--output='+str(out/'analysis_%j.log'),str(script)]
    try:
        result=subprocess.run(command,check=True,capture_output=True,text=True,timeout=60)
        job=result.stdout.strip().split(';')[0]
        if not job.isdigit():raise ValueError('Ambiguous analysis submission')
        r=dict(state='submitted',job_id=job,command=command,time=time.time())
    except Exception as exc:r=dict(state='needs_review',error=repr(exc),command=command,time=time.time())
    atomic_write_json(r,status);return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args();m=json.loads(Path(a.manifest).read_text())
    if m.get('test_access') is not False:raise ValueError('Test remains sealed')
    from look.analysis.linear_report import group_report
    group_report(m['runs'],m['output'])
    atomic_write_json(dict(state='completed',time=time.time()),Path(m['output'])/'status.json')

if __name__=='__main__':main()
