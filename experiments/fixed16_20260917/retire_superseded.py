"""Bounded supervisor: pause only the superseded search after new formal evidence."""
import json,sys,time,fcntl,hashlib,os,subprocess
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def main():
 c=json.load(open(sys.argv[1]));m=Path(c['manager']);r=Path(c['task']['run_dir']);old=Path(c['superseded_search'])
 assert old.name=='2026_09_16_14_28_13_776837' and r.name=='2026_09_17_08_25_08_837397'
 with (m/'retire.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   state=json.load(open(m/'status.json'))
   if state['state'] in ('failed','needs_review','paused'):
    write(m/'retire_status.json',dict(state='replacement_not_ready',manager=state));return
   if (old/'pause.json').exists():write(m/'retire_status.json',dict(state='already_requested'));return
   if state['state'] in ('running_formal','accepted'):
    profile=r/'resource_profile/profile_accepted.json';p=json.load(open(profile));assert p['state']=='accepted'
    for name,digest in p['files'].items():assert sha(profile.parent/name)==digest
    for cache in r.glob('corrections/*/factors/x16/rounds/*/site_*.json'):
     rows=json.load(open(cache))['candidates'];assert rows
     factor_root=cache.parents[2]
     for row in rows:
      artifact=(factor_root/row['artifact']).resolve();assert artifact.is_relative_to(factor_root.resolve())
      assert sha(artifact)==row['sha256']
      assert sha(row['evidence']['prediction'])==row['evidence']['sha256']
     # Bound to the verified replacement step or final accepted execution.
     if state['state']=='running_formal':
      q=subprocess.check_output(['scontrol','show','step',c['job']+'.'+str(state['step']),'-o'],text=True)
      assert 'State=RUNNING' in q
     marker=dict(reason='fixed16_formal_site_replay_artifacts_verified',replacement=str(r),cache=str(cache),time=time.time())
     write(old/'pause.json',marker);write(m/'retire_status.json',dict(state='pause_requested',**marker));return
   write(m/'retire_status.json',dict(state='waiting_formal_site_acceptance',time=time.time()));time.sleep(20)
if __name__=='__main__':main()
