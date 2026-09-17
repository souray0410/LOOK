import json,os,sys,pathlib,hashlib,subprocess,time,fcntl
O=pathlib.Path('/ibex/project/c2377/souray/home/mengh/operations/2026_09_10_11_11_31');old=O/'look_serial_20260917_v2';b=O/'look_serial_20260917_allocator_recovery';b.mkdir(exist_ok=True)
def read(p):return json.load(open(p))
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
c=read(old/'sequence.json');env=c['env'].copy();env.update(MALLOC_ARENA_MAX='1',MALLOC_TRIM_THRESHOLD_='131072',MALLOC_MMAP_THRESHOLD_='131072')
os.environ.update(env);sys.path[:0]=env['PYTHONPATH'].split(':')
from scheduling.policy import Claims
from scheduling.slurm_liveness import step_presence
from look.studies.search_case import dependencies,check_files
from look.runtime.state import stable_hash
with (b/'recovery.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (b/'launch.json').exists():raise RuntimeError('Recovery already launched; inspect it')
 assert step_presence('51909172','9') is False
 prev=read(old/'manager_best_forward_1/status.json');assert prev['state']=='failed'
 assert 'MemoryError: Host RAM reserve breached' in (old/'manager_best_forward_1/formal.log').read_text()
 task=read(c['tasks'][0]['config'])['task'];run=pathlib.Path(task['run_dir'])
 with (run/'run.lock').open('a') as rl:fcntl.flock(rl,fcntl.LOCK_EX|fcntl.LOCK_NB)
 spec=read(task['spec']);dependencies(spec);receipt=read(run/'resource_profile/profile_accepted.json');assert receipt['identity']==stable_hash(spec);check_files(run/'resource_profile',receipt['files'])
 write(b/'repair.json',dict(state='verifying_allocator_hypothesis',cause='RSS grew during full development scan; sampled profile missed growth',previous_step='51909172.9',log_sha256=sha(old/'manager_best_forward_1/formal.log'),previous_failure=prev,environment_delta={k:env[k] for k in ('MALLOC_ARENA_MAX','MALLOC_TRIM_THRESHOLD_','MALLOC_MMAP_THRESHOLD_')},scientific_spec_sha256=sha(task['spec']),run=str(run),source_unchanged=True,claim_review='old terminal step verified',memory_gib=6,resource_limits_unchanged=True,time=time.time()))
 claims=Claims(read(c['tasks'][0]['config'])['claims'])
 def reviewed(x):
  assert x['state']=='failed' and x['step']=='9' and x['spec_sha256']==task['spec_sha256'],x
  x.update(state='paused',failure_review=str(b/'repair.json'),updated_at=time.time());return x
 claims.mutate(run,reviewed)
 items=[]
 for n,item in enumerate(c['tasks']):
  v=read(item['config']);v['env']=env.copy();v['manager']=str(b/('manager_'+str(n)));v['reuse_profile']=(n==0)
  q=b/('task_'+str(n)+'.json');write(q,v);items.append(dict(config=str(q),sha256=sha(q),run=item['run']))
 c.update(env=env,output=str(b/'sequence'),launcher=str(b/'launch_bound.py'),tasks=items);write(b/'sequence.json',c)
 tests=['tests/unit/studies/test_search_policy.py','tests/unit/analysis/test_search_delivery.py','tests/unit/analysis/test_search_report.py','tests/unit/methods/test_independent_greedy.py']
 r=subprocess.run([c['python'],'-m','pytest','-q']+[str(old/'source'/p) for p in tests],env=env,stdout=open(b/'tests.log','w'),stderr=subprocess.STDOUT)
 assert r.returncode==0,(b/'tests.log').read_text()[-2500:]
 p=subprocess.Popen([c['python'],str(old/'source/experiments/serial_delivery_20260917/sequence.py'),str(b/'sequence.json')],env=env,stdout=open(b/'sequence.log','x'),stderr=subprocess.STDOUT,start_new_session=True)
 write(b/'launch.json',dict(pid=p.pid,time=time.time(),manager_sha256=sha(b/'launch_bound.py'),scientific_source='b6bbc2727255f4cd7c63c2108bb19a8073f137c2',tests=(b/'tests.log').read_text()[-500:]));print('launched',p.pid)
