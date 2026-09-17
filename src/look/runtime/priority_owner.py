"""One pinned LOOK continuation before the unchanged allocation owner."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def verify(config):
    for path,digest in config['pins'].items():
        if sha(path)!=digest:raise ValueError('Owner pin changed: '+path)
    if sha(config['continuation_binding'])!=config['continuation_binding_sha256']:
        raise ValueError('Continuation binding changed')


def priority(config,job,run=subprocess.run):
    """Return 75 to retire, otherwise hand over. Errors are quarantined once."""
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    key=config['continuation_binding_sha256'];incident=out/('quarantine_'+key+'.json')
    with (out/'priority.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(state='owned_elsewhere',returncode=0)
        if incident.exists():return dict(state='quarantined',incident=str(incident),returncode=0)
        event=dict(job=str(job),binding_sha256=key,time=time.time())
        log=out/'attempts'/(str(job)+'_'+str(time.time_ns())+'.log');log.parent.mkdir(exist_ok=True)
        try:
            verify(config)
            binding=read(config['continuation_binding'])
            command=[config['python'],config['continuation_program'],'--binding',config['continuation_binding'],
                     '--output',binding['output'],'--job',str(job)]
            env=dict(os.environ,**config['continuation_environment'])
            with log.open('x') as stream:
                admission=run([config['python'],config['admission_program'],'--binding',config['continuation_binding'],
                               '--job',str(job)],env=env,stdout=stream,stderr=subprocess.STDOUT)
                if admission.returncode==76:
                    event.update(state='capacity_skipped',returncode=0,log=str(log))
                    write(event,out/'attempts'/(str(job)+'_'+str(time.time_ns())+'.json'))
                    return event
                if admission.returncode!=0:raise RuntimeError('Admission verification exit '+str(admission.returncode))
                result=run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            rc=result.returncode
            if rc not in (0,75):raise RuntimeError('Continuation exit '+str(rc))
            event.update(state='retire_lease' if rc==75 else 'handover',returncode=rc,log=str(log))
        except Exception as error:
            event.update(state='quarantined',returncode=0,error=repr(error),log=str(log),
                         scope='priority_search_only; fallback_preserved',recovery='explicit_review_and_new_binding_required')
            write(event,incident)
        write(event,out/'attempts'/(str(job)+'_'+str(time.time_ns())+'.json'))
        return event


def gpu_owner(config):
    job=os.environ['SLURM_JOB_ID']
    result=priority(config,job);print(json.dumps(result),flush=True)
    if result['returncode']==75:return 75
    # The fallback has its own immutable scientific bindings and original env.
    # Verify its pins even when priority is quarantined; do not hide corruption.
    for path,digest in config['fallback_pins'].items():
        if sha(path)!=digest:raise ValueError('Fallback pin changed: '+path)
    env=dict(os.environ,**config['fallback_environment'])
    os.execvpe(config['fallback_command'][0],config['fallback_command'],env)


def allocation_owner(config,config_path):
    for path,digest in config['fallback_pins'].items():
        if sha(path)!=digest:raise ValueError('Fallback pin changed: '+path)
    job=os.environ['SLURM_JOB_ID']
    fields=dict(x.split('=',1) for x in subprocess.check_output(
        ['scontrol','show','job',job,'-o'],text=True,timeout=20).split() if '=' in x)
    end=datetime.fromisoformat(fields['EndTime']).timestamp()
    tres=dict(x.split('=',1) for x in fields.get('AllocTRES',fields.get('ReqTRES','')).split(',') if '=' in x)
    memory=tres.get('mem','0M');unit=memory[-1];amount=float(memory[:-1])
    environment=dict(os.environ,LOOK_ALLOCATION_END=str(end),
        LOOK_ALLOCATION_MEMORY_GIB=str(amount*{'M':1/1024,'G':1,'T':1024}[unit]))
    result=subprocess.run(['srun','--jobid='+job,'--overlap','--exact','--nodes=1','--ntasks=1','--gpus=1',
        '--cpus-per-task=1','--mem=2G','--unbuffered',config['python'],config['wrapper_program'],
        '--config',str(config_path),'--gpu-owner'],env=environment)
    return result.returncode


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--allocation-owner',action='store_true');mode.add_argument('--gpu-owner',action='store_true')
    mode.add_argument('--check',action='store_true');a=p.parse_args();c=read(a.config)
    if a.check:verify(c);print(json.dumps(dict(state='pins_verified_not_activated',fallback=c['fallback_command'])));return
    raise SystemExit(allocation_owner(c,a.config) if a.allocation_owner else gpu_owner(c))


if __name__=='__main__':main()
