"""Bounded, non-disruptive expiry guard for an existing resource qualification."""
import argparse
import fcntl
import json
from pathlib import Path
import time
from look.runtime.profile_lifecycle import adopt_profile
from look.runtime.state import atomic_write_json, file_sha256


def run(config_path):
    cfg=json.loads(Path(config_path).read_text())
    source=Path(cfg['profile']);run_dir=Path(cfg['run_dir']);out=Path(cfg['output'])
    out.mkdir(parents=True,exist_ok=True)
    if file_sha256(cfg['spec'])!=cfg['spec_sha256']:
        raise ValueError('Qualification specification changed')
    spec=json.loads(Path(cfg['spec']).read_text())
    with (out/'guard.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            state=json.loads((source/'status.json').read_text())
            if state.get('state') in ('paused','completed'):
                try:adopt_profile(source,run_dir/'resource_profile',spec)
                except BlockingIOError:pass  # Writer is still unwinding safely.
                else:
                    atomic_write_json(dict(state='resumable' if state['state']=='paused' else 'qualified',
                        profile=str(source),run_dir=str(run_dir),updated_at=time.time(),
                        scientific_acceptance=False),out/'status.json')
                    return
            elif state.get('state')=='needs_review':
                raise RuntimeError('Qualification failed; no automatic scientific retry')
            if time.time()>=cfg['pause_at']:
                atomic_write_json(dict(reason='qualification_step_expiry_checkpoint',time=time.time()),source/'pause.json')
            if time.time()>cfg['deadline']:
                raise RuntimeError('No closed qualification before deadline; preserve evidence and healthy workers')
            atomic_write_json(dict(state='waiting',updated_at=time.time(),pause_at=cfg['pause_at']),out/'status.json')
            time.sleep(10)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    try:run(a.config)
    except Exception as error:
        atomic_write_json(dict(state='needs_review',error=repr(error),updated_at=time.time()),Path(cfg['output'])/'status.json')
        raise

if __name__=='__main__':main()
