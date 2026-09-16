"""One read-only CPU analysis owner; never claims or interrupts GPU training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.studies.gate_manifest import PROTOCOL


def read(p): return json.loads(Path(p).read_text())


def tick(config):
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'registry.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=dict(updated_at=time.time(),accepted=0,waiting_source=0,errors=[],submitted=[],test_access=False)
        tasks=[]
        for feed in config['feeds']:
            data=read(feed)
            if data.get('test_access') is not False:raise ValueError('Unsealed feed')
            for row in data['tasks']:
                root=Path(row['run_dir'])
                if not (root/'accepted.json').exists():status['waiting_source']+=1;continue
                key=stable_hash(dict(root=str(root),receipt=file_sha256(root/'accepted.json'),protocol=PROTOCOL))
                target=out/'runs'/key
                if (target/'accepted.json').exists():
                    r=read(target/'accepted.json')
                    if r.get('state')!='accepted' or r.get('test_access') is not False:
                        status['errors'].append(dict(run=str(root),reason='Invalid review receipt'));continue
                    status['accepted']+=1;continue
                if (target/'status.json').exists() and read(target/'status.json').get('state')=='needs_review':
                    status['errors'].append(dict(run=str(root),reason=read(target/'status.json')));continue
                tasks.append(dict(run=str(root),output=str(target)))
        # One finite CPU batch at a time. Ambiguous submission remains blocking.
        for request in sorted((out/'requests').glob('*/request.json')) if (out/'requests').exists() else []:
            r=read(request)
            if (request.parent/'completed.json').exists():continue
            if r['state']!='submitted':
                status['errors'].append(dict(request=str(request),reason='Ambiguous submission; inspect before retry'))
                atomic_write_json(status,out/'status.json');return status
            result=subprocess.run(['squeue','--noheader','--jobs='+str(r['job_id']),'--format=%i'],text=True,capture_output=True,timeout=30)
            if result.returncode or result.stdout.strip():
                status['submitted'].append(r);atomic_write_json(status,out/'status.json');return status
            result=subprocess.run(['sacct','-n','-X','-j',str(r['job_id']),'--format=State','--parsable2'],text=True,capture_output=True,timeout=30)
            states=[line.strip().split('|')[0] for line in result.stdout.splitlines() if line.strip()]
            if result.returncode or not states or any(s not in ('COMPLETED','TIMEOUT','PREEMPTED','NODE_FAIL','CANCELLED') for s in states):
                status['errors'].append(dict(request=str(request),reason='Failed or unknown CPU job',states=states))
                atomic_write_json(status,out/'status.json');return status
            atomic_write_json(dict(state='ended',states=states,time=time.time()),request.parent/'completed.json')
        if tasks:
            batch=out/'requests'/str(time.time_ns());batch.mkdir(parents=True)
            manifest=dict(tasks=tasks,protocol=PROTOCOL,test_access=False)
            atomic_write_json(manifest,batch/'batch.json')
            script=batch/'run.sh'
            script.write_text('#!/bin/bash\nset -euo pipefail\n'+
                'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n'+
                'export PYTHONPATH='+shlex.quote(config['pythonpath'])+'\n'+
                'export LD_LIBRARY_PATH='+shlex.quote(config['ld_library_path'])+'\n'+
                shlex.join([config['python'],'-m','look.runtime.gate_review','--batch',str(batch/'batch.json')])+'\n')
            atomic_write_json(dict(state='submission_intent',script_sha256=file_sha256(script)),batch/'request.json')
            cmd=['sbatch','--parsable','--account=pi-mengy','--nodes=1','--ntasks=1','--cpus-per-task=4',
                '--mem=16G','--time=48:00:00','--job-name=look_gate_review',
                '--output='+str(batch/'slurm_%j.log'),str(script)]
            try:
                r=subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=60)
                job=r.stdout.strip().split(';')[0]
                if not job.isdigit():raise ValueError('Ambiguous job ID')
                receipt=dict(state='submitted',job_id=job,command=cmd,time=time.time())
            except Exception as e:receipt=dict(state='needs_review',error=repr(e),time=time.time())
            atomic_write_json(receipt,batch/'request.json');status['submitted'].append(receipt)
        atomic_write_json(status,out/'status.json');return status


def batch_run(path):
    from look.studies.gate_manifest import import_run
    from look.analysis.gate_report import report
    path=Path(path);m=read(path)
    if m.get('protocol')!=PROTOCOL or m.get('test_access') is not False:raise ValueError('Unregistered batch')
    failures=[]
    for row in m['tasks']:
        out=Path(row['output']);out.mkdir(parents=True,exist_ok=True)
        with (out/'run.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            atomic_write_json(dict(state='running',pid=os.getpid(),time=time.time()),out/'status.json')
            try:
                manifest=import_run(row['run']);report(manifest,out)
                atomic_write_json(dict(state='completed',time=time.time(),test_access=False),out/'status.json')
            except Exception as e:
                import traceback
                error=dict(state='needs_review',error=repr(e),traceback=traceback.format_exc(),time=time.time())
                atomic_write_json(error,out/'status.json');failures.append(error)
    atomic_write_json(dict(state='completed_with_errors' if failures else 'completed',errors=failures,time=time.time()),path.parent/'completed.json')
    if failures:raise RuntimeError('Some gate reviews need attention')


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--config');g.add_argument('--batch');p.add_argument('--once',action='store_true');a=p.parse_args()
    if a.batch:batch_run(a.batch);return
    config=read(a.config);out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (out/'stop.json').exists():
            try:tick(config)
            except Exception as e:atomic_write_json(dict(state='needs_review',error=repr(e),time=time.time()),out/'status.json')
            if a.once:break
            time.sleep(300)

if __name__=='__main__':main()
