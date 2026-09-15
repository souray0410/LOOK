"""Release already approved single-final work at the accepted-host boundary."""
import argparse
import fcntl
from pathlib import Path
import time
from look.runtime.state import atomic_write_json,file_sha256,stable_hash
from look.studies.project_case import read
from look.studies.linear_protocol import matrix
from look.studies.terminal_case import VERSION,protocol,verify_case,dependencies


def tick(config):
    from runtime.run_registry import reserve
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=protocol();path=out/'protocol.json'
        if path.exists() and read(path)!=p:raise ValueError('Stage protocol changed')
        atomic_write_json(p,path)
        feed=read(config['base_feed'])
        if feed.get('schema')!='look_project_work_feed_v1' or feed.get('test_access') is not False:raise ValueError('Unsealed base feed')
        hosts={};errors=[];tasks=[];waiting={};pilots={};completed=0
        for row in feed['tasks']:
            root=Path(row['run_dir'])
            if not (root/'host/accepted.json').exists():continue
            base=read(row['spec']);key=(base['disease'],base['model']['name'],base['position'],base['seed'])
            r=read(root/'host/accepted.json')
            hosts[key]=dict(run_dir=str(root),spec_path=row['spec'],spec_sha256=row['spec_sha256'],
                host_accepted_sha256=file_sha256(root/'host/accepted.json'),best_sha256=r['files']['best.pt'])
        # Stage every predeclared host, first seed before repetitions. No score-based gate.
        for h in sorted(matrix(),key=lambda h:(h['seed'],h['disease'],h['architecture'],{'middle':0,'deep':1,'features':2}[h['position']])):
            key=tuple(h[k] for k in ('disease','architecture','position','seed'));group=key[:3];name='/'.join(map(str,key))
            if key not in hosts:waiting[name]='waiting_host_acceptance';continue
            if h['seed']!=3416 and group not in pilots:waiting[name]='waiting_first_seed_technical_acceptance';continue
            spec=dict(schema=VERSION,protocol=p,host=h,source=hosts[key],source_pins=config['source_pins'],test_access=False)
            if h['seed']!=3416:spec['pilot']=pilots[group]
            try:
                dependencies(spec)
                sp=out/'specs'/(stable_hash(h)+'.json')
                if sp.exists() and read(sp)!=spec:raise ValueError('Stage identity changed')
                atomic_write_json(spec,sp)
                run=Path(reserve(out,VERSION,stable_hash(h),spec,source=hosts[key],refresh_summary=False))
                if (run/'accepted.json').exists():
                    verify_case(run,spec);completed+=1
                    if h['seed']==3416:pilots[group]=dict(run_dir=str(run),sha256=file_sha256(run/'accepted.json'))
                tasks.append(dict(id='terminal/'+name,spec=str(sp),spec_sha256=file_sha256(sp),run_dir=str(run)))
            except Exception as e:errors.append(dict(host=name,error=repr(e)))
        result=dict(schema='look_terminal_work_feed_v1',test_access=False,tasks=tasks,expected_hosts=81,
            accepted_stages=completed,accepted_pilots=len(pilots),waiting=waiting,errors=errors,updated_at=time.time())
        atomic_write_json(result,out/'queue.json');atomic_write_json({k:v for k,v in result.items() if k!='tasks'},out/'status.json')
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--once',action='store_true')
    a=p.parse_args();config=read(a.config);out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    if a.once:tick(config);return
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (out/'stop.json').exists():
            try:tick(config)
            except Exception as e:atomic_write_json(dict(state='needs_review',error=repr(e),updated_at=time.time()),out/'status.json')
            time.sleep(300)

if __name__=='__main__':main()
