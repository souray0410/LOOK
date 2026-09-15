"""Publish dependency-ready work to the existing dispatcher; never allocates GPUs."""
import argparse
import fcntl
from pathlib import Path
import time
from look.runtime.state import atomic_write_json,file_sha256,stable_hash
from look.studies.project_case import read,verify_case as verify_host
from look.studies.linear_case import verify_case
from look.studies.linear_protocol import VERSION,matrix,protocol
from look.runtime.mechanism_receipts import monitored_acceptance


def tick(config):
    from runtime.run_registry import reserve
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=protocol();path=out/'protocol.json'
        if path.exists() and read(path)!=p:raise ValueError('Linear protocol changed')
        atomic_write_json(p,path);atomic_write_json(matrix(),out/'matrix.json')
        feed=read(config['base_feed'])
        if feed.get('schema')!='look_project_work_feed_v1' or feed.get('test_access') is not False:raise ValueError('Invalid base feed')
        hosts={};errors=[];ready=[];wait={};pilots={};completed=0
        for row in feed['tasks']:
            root=Path(row['run_dir'])
            if not (root/'accepted.json').exists():continue
            try:
                s=read(row['spec'])
                if file_sha256(row['spec'])!=row['spec_sha256']:raise ValueError('Host specification changed')
                r=monitored_acceptance(root,s,verify_host,out/'cache')
                key=(s['disease'],s['model']['name'],s['position'],s['seed'])
                hosts[key]=dict(run_dir=str(root),spec_path=row['spec'],spec_sha256=row['spec_sha256'],
                    accepted_sha256=file_sha256(root/'accepted.json'),best_sha256=r['files']['host/best.pt'])
            except Exception as e:errors.append(dict(run=str(root),error=repr(e)))
        # Pilot first for every disease/architecture/position, never top-scoring only.
        for h in sorted(matrix(),key=lambda h:(h['seed'],h['disease'],h['architecture'],h['position'])):
            key=tuple(h[k] for k in ('disease','architecture','position','seed'));group=key[:3];name='/'.join(map(str,key))
            if key not in hosts:wait[name]='waiting_accepted_host';continue
            if h['seed']!=3416 and group not in pilots:wait[name]='waiting_complete_three_arm_pilot';continue
            spec=dict(schema=VERSION,host=h,protocol=p,source=hosts[key],source_pins=config['source_pins'],test_access=False)
            if h['seed']!=3416:spec['pilot']=pilots[group]
            specpath=out/'specs'/(stable_hash(h)+'.json');specpath.parent.mkdir(exist_ok=True)
            if specpath.exists() and read(specpath)!=spec:raise ValueError('Linear task identity changed')
            atomic_write_json(spec,specpath)
            run=Path(reserve(out,VERSION,stable_hash(h),spec,source=hosts[key],refresh_summary=False))
            row=dict(id=name,spec=str(specpath),spec_sha256=file_sha256(specpath),run_dir=str(run),execution='look_linear')
            if (run/'accepted.json').exists():
                try:
                    monitored_acceptance(run,spec,verify_case,out/'cache');completed+=1
                    if h['seed']==3416:pilots[group]=dict(run_dir=str(run),sha256=file_sha256(run/'accepted.json'))
                except Exception as e:errors.append(dict(run=str(run),error=repr(e)));continue
            ready.append(row)
        # Complete matched groups produce a three-seed report automatically.
        report_groups={}
        for row in ready:
            root=Path(row['run_dir'])
            if not (root/'accepted.json').exists():continue
            h=read(row['spec'])['host'];key=(h['disease'],h['architecture'],h['position'])
            report_groups.setdefault(key,[]).append(root)
        for key,runs in report_groups.items():
            if len(runs)!=3:continue
            target=out/'reports'/stable_hash(key)
            inputs={str(r):file_sha256(r/'accepted.json') for r in runs}
            if (target/'inputs.json').exists() and read(target/'inputs.json')==inputs:continue
            from look.runtime.linear_analysis import request
            request(config,runs,target)
        result=dict(schema='look_linear_work_feed_v1',test_access=False,tasks=ready,expected_hosts=81,expected_arm_views=243,
            complete_hosts=completed,accepted_pilots=len(pilots),waiting=wait,errors=errors,updated_at=time.time())
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
