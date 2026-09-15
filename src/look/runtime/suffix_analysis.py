"""Bounded CPU analysis, with durable submission identity and no GPU requests."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from look.runtime.state import atomic_write_json, file_sha256


def request(config,manifest):
    out=Path(manifest['output']);out.mkdir(parents=True,exist_ok=True);record=out/'request.json'
    if record.exists():return json.loads(record.read_text())
    path=out/'analysis.json';atomic_write_json(manifest,path)
    script=out/'analysis.sh'
    script.write_text('#!/bin/bash\nset -euo pipefail\nexport OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\n'+
        'export LD_LIBRARY_PATH='+shlex.quote(config['ld_library_path'])+'\nexport PYTHONPATH='+shlex.quote(config['pythonpath'])+'\n'+
        shlex.join([config['python'],'-m','look.runtime.suffix_analysis','--manifest',str(path)])+'\n')
    atomic_write_json(dict(state='submission_intent',script_sha256=file_sha256(script),time=time.time()),record)
    command=['sbatch','--parsable','--account=pi-mengy','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=32G',
        '--time=48:00:00','--job-name=look_suffix_statistics','--output='+str(out/'analysis_%j.log'),str(script)]
    try:
        r=subprocess.run(command,text=True,capture_output=True,check=True,timeout=60);job=r.stdout.strip().split(';')[0]
        if not job.isdigit():raise ValueError('Uncertain submission identity')
        result=dict(state='submitted',job_id=job,time=time.time())
    except Exception as e:result=dict(state='needs_review',error=repr(e),time=time.time())
    atomic_write_json(result,record);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args();m=json.loads(Path(a.manifest).read_text())
    from look.analysis.suffix_report import report
    try:report(m)
    except Exception as e:
        atomic_write_json(dict(state='needs_review',error=repr(e),time=time.time()),Path(m['output'])/'status.json');raise
    atomic_write_json(dict(state='completed',time=time.time()),Path(m['output'])/'status.json')

if __name__=='__main__':main()
