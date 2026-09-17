"""Read-only finite-delivery watchdog. Never owns, restarts or cancels GPU work."""
import argparse,json,time,os,fcntl
from pathlib import Path

def assess(manager, last_progress, now, alive, stall_seconds=600):
    state=manager.get('state','initializing')
    if state in ('failed','needs_review'):
        return 'action_required_failure'
    if state=='accepted':return 'accepted_configuration'
    if state=='paused':return 'action_required_resumable_pause'
    if not alive:return 'action_required_manager_dead'
    if now-last_progress>stall_seconds:return 'action_required_no_experimental_progress'
    return 'progress_observed_not_acceptance'

def write(p,value):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def main():
    a=argparse.ArgumentParser();a.add_argument('--sequence',type=Path,required=True);a.add_argument('--interval',type=float,default=30);args=a.parse_args()
    c=read(args.sequence);root=args.sequence.parent;last_progress=time.time();previous=None
    with (root/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            now=time.time();seq=read(Path(c['output'])/'status.json');launch=read(root/'launch.json')
            if seq.get('state')=='all_configurations_delivered':
                write(root/'health.json',dict(state='finite_sequence_reported_complete',time=now,requires_receipt_verification=True));return
            current=next((read(Path(x['config'])) for x in c['tasks'] if x['run']==seq.get('run')),None)
            if current is None:
                status='initializing';fingerprint=None;detail={}
            else:
                m=Path(current['manager']);detail=read(m/'status.json');run=Path(current['task']['run_dir'])
                files=list(m.glob('*.log'))+list((run/'corrections').glob('*/factors/*/rounds/site_*.json'))+list((run/'corrections').glob('*/factors/*/rounds/*/site_*.json'))
                fingerprint=tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(files))
                if fingerprint!=previous:last_progress=now;previous=fingerprint
                status=assess(detail,last_progress,now,alive(launch['pid']))
            record=dict(state=status,time=now,sequence_state=seq,manager=detail,last_experimental_progress=last_progress,seconds_without_progress=now-last_progress,read_only=True)
            write(root/'health.json',record)
            if status.startswith('action_required'):
                # Owner is the existing maintenance automation, not this observer.
                record['continuation_owner']='existing_ukb_maintenance_automation_15min_during_incident'
                write(root/'open_incident.json',record)
            time.sleep(args.interval)
if __name__=='__main__':main()
