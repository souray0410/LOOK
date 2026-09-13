"""Retire only a CPU dispatcher while preserving its controlling TTY and children.

A suspended old session leader keeps existing salloc owners alive. The new
manager uses the same claims/account lock and copies the durable request journal.
No GPU worker or allocation is signalled. The old dispatcher is removed only
when every direct child has exited and no allocation owner can be affected.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time
from look.runtime.state import atomic_write_json,file_sha256


def read(p):return json.loads(Path(p).read_text())


def process(pid):
    root=Path('/proc')/str(pid)
    command=root.joinpath('cmdline').read_bytes().split(b'\0')
    raw=root.joinpath('stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,args=[v.decode() for v in command if v],state=raw[0],start=raw[19])


def validate_manager(p,config):
    args=p['args']
    if 'look.runtime.project_dispatch' not in args or str(config) not in args or any(x in args for x in ('--allocation-owner','--gpu-owner','--execute','--profile-project','--profile-mechanism')):
        raise ValueError('Only the exact CPU dispatcher may be handed over')


def handover(previous_config,new_config,previous_pid):
    old=read(previous_config);new=read(new_config);root=Path(new['output']);ready=read(root/'handover_ready.json')
    for key in ('claims','account_submission_lock','maximum_workflow_allocations'):
        if old[key]!=new[key]:raise ValueError('Handover changes resource ownership policy')
    oldp=process(previous_pid);newp=process(ready['pid']);validate_manager(oldp,previous_config);validate_manager(newp,new_config)
    if '--standby' not in newp['args'] or ready['state']!='ready':raise ValueError('Replacement must be validated and waiting')
    if (root/'handover_armed.json').exists():raise ValueError('Handover already recorded')
    oldjournal=Path(old['output'])/'requests.json'
    with Path(new['account_submission_lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=read(oldjournal)
        if any(not str(r.get('job_id','')).isdigit() for r in journal['requests']):raise ValueError('Ambiguous allocation identity')
        if (root/'requests.json').exists():raise ValueError('New journal must not overwrite requests')
        receipt=dict(state='prepared',previous=oldp,replacement=newp,old_journal=str(oldjournal),old_journal_sha256=file_sha256(oldjournal),time=time.time())
        atomic_write_json(receipt,root/'handover_intent.json')
        # A single PID, not its process group; allocated work keeps its TTY.
        os.kill(previous_pid,signal.SIGSTOP)
        try:
            atomic_write_json(journal,root/'requests.json')
            receipt.update(state='armed',new_journal_sha256=file_sha256(root/'requests.json'))
            atomic_write_json(receipt,root/'handover_armed.json')
        except Exception:
            os.kill(previous_pid,signal.SIGCONT)
            raise
    return receipt


def retire_parent(root):
    root=Path(root);path=root/'handover_armed.json'
    if not path.exists() or (root/'previous_retired.json').exists():return
    receipt=read(path);old=receipt['previous'];pid=old['pid']
    try:
        current=process(pid)
        if current['start']!=old['start']:raise ValueError('Previous PID reused; no signal sent')
        if current['state'] not in ('T','t'):raise ValueError('Previous manager unexpectedly resumed')
        children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        for child in children:
            try:
                if process(int(child))['state']!='Z':return
            except FileNotFoundError:pass
        if current['state']=='t':return  # The session guardian owns tracer cleanup.
        os.kill(pid,signal.SIGKILL)
        atomic_write_json(dict(previous_pid=pid,state='retired_after_children_exit',time=time.time()),root/'previous_retired.json')
    except FileNotFoundError:
        atomic_write_json(dict(previous_pid=pid,state='already_exited',time=time.time()),root/'previous_retired.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous-config',required=True);p.add_argument('--new-config',required=True);p.add_argument('--previous-pid',required=True,type=int)
    a=p.parse_args();print(json.dumps(handover(a.previous_config,a.new_config,a.previous_pid)))

if __name__=='__main__':main()
