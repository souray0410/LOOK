"""Reattach lifecycle observation after a login manager disappears.

Never launches or cancels a Slurm step, acquires a training claim, or changes the
scientific source. The old manager lock excludes a second observer/manager.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from look.runtime.state import atomic_write_json, file_sha256


def tick(config, launch, claims, present, verify, publish, end_time, now=None):
    now=time.time() if now is None else now
    task=config['task'];run=Path(task['run_dir']);owner=launch['owner']
    if file_sha256(task['spec'])!=task['spec_sha256']:
        raise ValueError('Scientific specification changed')
    def validate(record):
        if any(record.get(k)!=v for k,v in dict(owner=owner,job_id=str(config['job']),
                generation=launch['generation'],spec_sha256=task['spec_sha256']).items()):
            raise ValueError('Claim identity changed')
        return record
    record=validate(json.loads(claims.path(run).read_text()))
    if record['state'] not in ('running','claimed','liveness_needs_review'):
        return dict(state='claim_already_terminal',claim_state=record['state'])
    alive=present(str(config['job']),str(record['step']))
    if alive is not False:
        if now>=end_time-900:
            atomic_write_json(dict(reason='allocation_expiry',observer='attached_search',time=now),run/'pause.json')
        if alive is True:
            def renew(value):
                validate(value);value['updated_at']=now;return value
            claims.mutate(run,renew)
        return dict(state='observing' if alive else 'liveness_unknown',step=record['step'],time=now)
    # A completion flag is only a trigger for the full scientific verifier.
    if (run/'accepted.json').exists():
        verify(run,json.loads(Path(task['spec']).read_text()));publish()
        terminal='completed'
    else:
        status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        terminal='paused' if status.get('state')=='paused' or now>=end_time else 'failed'
    def finish(value):
        validate(value);value.update(state=terminal,updated_at=now,reattached_observer=True);return value
    claims.mutate(run,finish)
    return dict(state=terminal,time=now,scientific_acceptance=terminal=='completed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--sequence',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    c=json.loads(Path(a.config).read_text());manager=Path(c['manager']);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    launch=json.loads((manager/'launch.json').read_text())
    try:os.kill(int(launch['pid']),0)
    except ProcessLookupError:pass
    else:raise RuntimeError('Original manager PID still exists; reconcile before attachment')
    from scheduling.policy import Claims
    from scheduling.slurm_liveness import step_presence
    from look.studies.search_case import verify_case
    from look.studies.search_delivery_queue import refresh
    info=subprocess.check_output(['scontrol','show','job',str(c['job']),'-o'],text=True,timeout=20)
    attrs=dict(t.split('=',1) for t in info.split() if '=' in t)
    end=datetime.fromisoformat(attrs['EndTime']).timestamp()
    with (manager/'manager.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_write_json(dict(pid=os.getpid(),job=c['job'],step=json.loads((manager/'formal_step.json').read_text())['step'],
            generation=launch['generation'],time=time.time(),scope='observer_only'),out/'launch.json')
        while True:
            try:
                result=tick(c,launch,Claims(c['claims']),step_presence,verify_case,
                            lambda:refresh(a.sequence,out/'delivery'),end)
            except Exception as exc:
                atomic_write_json(dict(state='needs_review',error=repr(exc),time=time.time()),out/'status.json');raise
            atomic_write_json(result,out/'status.json')
            if result['state'] not in ('observing','liveness_unknown'):break
            time.sleep(30)


if __name__=='__main__':main()
