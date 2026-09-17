"""One registered LOOK task, shared claim, profile then execution, no allocator."""
import os,json,pathlib,subprocess,time,fcntl,datetime,sys
C=json.load(open(sys.argv[1]));B=pathlib.Path(C['manager']);B.mkdir(parents=True,exist_ok=True)
env=C['env'];sys.path[:0]=env['PYTHONPATH'].split(':')
from scheduling.policy import Claims
from scheduling.slurm_liveness import step_presence
from look.runtime.state import stable_hash,file_sha256,atomic_write_json as write
from look.studies.search_case import dependencies,check_files,verify_case
R=pathlib.Path(C['task']['run_dir']);S=json.load(open(C['task']['spec']));profile=R/'resource_profile';job=C['job'];owner='look-serial-'+job
threads=S.get('worker_threads',2);memory_gib=C.get('memory_gib',24)
assert int(env['LOOK_WORKER_MEMORY_BYTES'])==memory_gib*1024**3
with (B/'manager.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (B/'launch.json').exists():raise RuntimeError('Reconcile prior launch before restart')
 dependencies(S);assert S['spatial_factors']==[16] and len(S['latent_dims'])==1 and S['host']['seed']==3416
 info=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True);attrs=dict(t.split('=',1) for t in info.split() if '=' in t)
 assert attrs['JobState']=='RUNNING' and int(attrs['NumCPUs'])>=8 and 'mem=128G' in attrs['AllocTRES']
 end=datetime.datetime.fromisoformat(attrs['EndTime']).timestamp();assert end-time.time()>3600
 claims=Claims(C['claims']);token=claims.acquire(R,C['task']['spec_sha256'],owner,job)
 write(dict(pid=os.getpid(),time=time.time(),task=C['task'],owner=owner,generation=token['generation']),B/'launch.json')
 try:
  for phase,out in ([('formal',R)] if C.get('reuse_profile') else [('profile',profile),('formal',R)]):
   stamp=B/(phase+'_step.json');worker=B/(phase+'_worker.py')
   args=[C['python'],'-m','look.studies.search_case','--spec',C['task']['spec'],'--output',str(out)]+(['--profile'] if phase=='profile' else [])
   worker.write_text('import os,json,pathlib\npathlib.Path('+repr(str(stamp))+').write_text(json.dumps({"step":os.environ["SLURM_STEP_ID"]}))\nos.execvpe('+repr(C['python'])+','+repr(args)+',os.environ)\n')
   if phase=='formal':
    receipt=json.load(open(profile/'profile_accepted.json'));assert receipt['identity']==stable_hash(S) and receipt['state']=='accepted'
    check_files(profile,receipt['files']);cost=json.load(open(profile/'costs.json'))
    assert cost['gpu_peak_reserved_bytes']*1.2+2*1024**3<8*1024**3 and cost['rss_bytes']<.85*memory_gib*1024**3
    env['LOOK_SEARCH_PROFILE_RECEIPT']=str(profile/'profile_accepted.json')
   # Count requested co-resident step budgets; overlap is not extra capacity.
   current=subprocess.check_output(['scontrol','show','step',job,'-o'],text=True)
   cpu=0;ram=0
   for line in current.splitlines():
    a=dict(t.split('=',1) for t in line.split() if '=' in t)
    if a.get('State')!='RUNNING':continue
    cpu+=int(a.get('CPUs',0))
    for part in a.get('TRES','').split(','):
     if part.startswith('mem='):
      v=part[4:];ram+=float(v[:-1])*({'G':1,'M':1/1024,'T':1024}[v[-1]])
   assert cpu+threads<=int(attrs['NumCPUs']), 'Co-resident CPU budgets exceed allocation'
   assert ram+memory_gib<=128*.85, 'Co-resident host RAM reserve unavailable'
   remaining=int((end-time.time()-120)//60);assert remaining>15
   cmd=['srun','--jobid='+job,'--overlap','--exact','-N1','-n1','-c'+str(threads),'--gpus=1','--mem='+str(memory_gib)+'G','--time='+str(remaining),'--job-name=look_fixed16',C['python'],str(worker)]
   with (B/(phase+'.log')).open('x') as log:child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
   step=None;released_old=False
   while child.poll() is None:
    if stamp.exists():
     step=json.load(open(stamp))['step'];claims.update(R,owner,state='running',step=step)
    if time.time()>end-900 or (B/'pause.json').exists():write(dict(reason='expiry_or_requested',time=time.time()),out/'pause.json')
    # Retire only superseded three-factor search after fixed16 completes a real
    # formal site fit. The distinct original host-study worker is never touched.
    if C.get('superseded_search') and phase=='formal' and not released_old and list((R/'corrections').glob('*/factors/x16/rounds/*/site_*.json')):
     for cache in (R/'corrections').glob('*/factors/x16/rounds/*/site_*.json'):
      rows=json.load(open(cache))['candidates'];assert rows
      folder=cache.parents[2]
      for row in rows:
       path=(folder/row['artifact']).resolve();assert path.is_relative_to(folder.resolve())
       assert file_sha256(path)==row['sha256']
       assert file_sha256(row['evidence']['prediction'])==row['evidence']['sha256']
     targets=C.get('superseded_searches',[C['superseded_search']])
     for old in targets:write(dict(reason='fixed32_formal_site_verified',replacement=str(R),time=time.time()),pathlib.Path(old)/'pause.json')
     released_old=True
    write(dict(state='running_'+phase,step=step,run=str(R),updated_at=time.time(),scientific_acceptance=False),B/'status.json');time.sleep(10)
   for _ in range(30):
    if step is not None and step_presence(job,step) is False:break
    time.sleep(2)
   else:raise RuntimeError('Step exit not verified; claim retained')
   if child.returncode!=0:
    state='paused' if child.returncode==75 else 'failed';claims.release(R,owner,state,step_dead=True)
    write(dict(state=state,phase=phase,returncode=child.returncode,updated_at=time.time()),B/'status.json');sys.exit(child.returncode)
  verify_case(R,S)
  from look.analysis.search_delivery import report
  report(R)
  claims.release(R,owner,'completed',step_dead=True)
  write(dict(state='accepted',run=str(R),updated_at=time.time()),B/'status.json')
 except Exception as e:
  write(dict(state='needs_review',error=repr(e),updated_at=time.time()),B/'status.json');raise
