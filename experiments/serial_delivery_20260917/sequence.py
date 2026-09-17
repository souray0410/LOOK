"""One finite delivery sequence. Uses the existing shared claims, no allocations."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path

def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)

def main():
 c=json.load(open(sys.argv[1]));root=Path(c['output']);root.mkdir(parents=True,exist_ok=True)
 env=c['env'];sys.path[:0]=env['PYTHONPATH'].split(':')
 from look.analysis.search_delivery import report
 from look.runtime.state import file_sha256
 with (root/'sequence.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for item in c['tasks']:
   if file_sha256(item['config'])!=item['sha256']:raise ValueError('Sequence configuration changed')
   task=json.load(open(item['config']));run=Path(task['task']['run_dir']);manager=Path(task['manager'])
   if (run/'accepted.json').exists():
    report(run);continue
   if (manager/'launch.json').exists():
    write(root/'status.json',dict(state='reconcile_existing_attempt',run=str(run),time=time.time()));return
   write(root/'status.json',dict(state='executing',run=str(run),time=time.time()))
   result=subprocess.run([c['python'],c['launcher'],item['config']],env=env)
   if result.returncode:
    write(root/'status.json',dict(state='paused' if result.returncode==75 else 'needs_review',returncode=result.returncode,run=str(run),time=time.time()));return
   report(run)
   write(root/'status.json',dict(state='configuration_delivered',run=str(run),time=time.time()))
  write(root/'status.json',dict(state='all_configurations_delivered',time=time.time()))
if __name__=='__main__':main()
