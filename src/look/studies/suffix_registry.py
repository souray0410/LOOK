"""Register independent starts in the existing claim/dispatch system."""
import argparse
import fcntl
from pathlib import Path
import time
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.studies.project_case import read
from look.studies.suffix_protocol import VERSION, protocol, sites, PATTERNS
from look.studies.suffix_case import verify_case
from look.analysis.suffix_report import verify_group
from look.runtime.mechanism_receipts import monitored_acceptance


def tick(config):
    from runtime.run_registry import reserve
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=protocol();path=out/'protocol.json'
        if path.exists() and read(path)!=p:raise ValueError('Suffix protocol changed')
        atomic_write_json(p,path);feed=read(config['base_feed'])
        if feed.get('schema')!='look_project_work_feed_v1' or feed.get('test_access') is not False:raise ValueError('Unsealed host feed')
        waiting={};errors=[];tasks=[];accepted=0;pilots=0;groups=[]
        rows=sorted(feed['tasks'],key=lambda r:(read(r['spec'])['seed'],r['id']))
        for row in rows:
            root=Path(row['run_dir']);base=read(row['spec']);h=dict(disease=base['disease'],architecture=base['model']['name'],position=base['position'],seed=base['seed'])
            name='/'.join(map(str,h.values()));key=stable_hash(h);target=out/'reports'/key
            if not (root/'host/accepted.json').exists():waiting[name]='waiting_frozen_host';continue
            manifests=list(root.glob('pca/*/bank_manifest.json'))
            manifests=[m for m in manifests if read(m)['identity'].get('case')==stable_hash(base)]
            if not manifests:waiting[name]='waiting_complete_train_PCA';continue
            if len(manifests)!=1:errors.append(dict(host=name,error='ambiguous PCA bank'));continue
            try:
                if file_sha256(row['spec'])!=row['spec_sha256']:raise ValueError('Host spec changed')
                host_receipt=read(root/'host/accepted.json')
                if host_receipt.get('state')!='accepted' or host_receipt.get('identity')!=stable_hash(base):raise ValueError('Host receipt mismatch')
                ordered=sites(h['architecture'],h['position']);pilot=None
                if h['seed']!=3416:
                    pilot_dir=out/'reports'/stable_hash(dict(h,seed=3416))
                    if not (pilot_dir/'accepted.json').exists():waiting[name]='waiting_complete_3416_start_comparison';continue
                    r=verify_group(pilot_dir)
                    if r['host']!=dict(h,seed=3416):raise ValueError('Pilot host mismatch')
                    pilot=dict(path=str(pilot_dir),sha256=file_sha256(pilot_dir/'accepted.json'))
                source=dict(run_dir=str(root),spec_path=row['spec'],spec_sha256=row['spec_sha256'],
                    host_accepted_sha256=file_sha256(root/'host/accepted.json'),best_sha256=host_receipt['files']['best.pt'])
                completed=[]
                # Short late-start routes first; all predeclared starts retained.
                for ordinal in reversed(range(2,len(ordered))):
                    spec=dict(schema=VERSION,protocol=p,host=h,start_ordinal=ordinal,candidate_sites=ordered,eligible_sites=ordered[ordinal-1:],
                        source=source,pca=dict(path=str(manifests[0]),sha256=file_sha256(manifests[0])),source_pins=config['source_pins'],test_access=False)
                    if pilot:spec['pilot']=pilot
                    identity=stable_hash(dict(host=h,start=ordinal));sp=out/'specs'/(identity+'.json');sp.parent.mkdir(exist_ok=True)
                    if sp.exists() and read(sp)!=spec:raise ValueError('Suffix task identity changed')
                    atomic_write_json(spec,sp)
                    run=Path(reserve(out,VERSION,identity,spec,source=source,refresh_summary=False))
                    task=dict(id='suffix/'+name+'/start'+str(ordinal),spec=str(sp),spec_sha256=file_sha256(sp),run_dir=str(run),execution='look_suffix')
                    if (run/'accepted.json').exists():
                        monitored_acceptance(run,spec,verify_case,out/'cache');accepted+=1;completed.append(str(run))
                    tasks.append(task)
                # The original earliest and terminal routes are aliases, never refit.
                if len(completed)==len(ordered)-2 and (root/'accepted.json').exists():
                    from look.runtime.suffix_analysis import request
                    request(config,dict(host=h,source=source,runs=completed,output=str(target),test_access=False))
                if (target/'accepted.json').exists():
                    verify_group(target);groups.append(dict(host=h,report=str(target)))
                    if h['seed']==3416:pilots+=1
                else:waiting[name]='fitting_suffixes_or_waiting_original_anchors'
            except Exception as e:errors.append(dict(host=name,error=repr(e)))
        result=dict(schema='look_suffix_work_feed_v1',test_access=False,tasks=tasks,expected_hosts=81,
            expected_new_fit_tasks=567,expected_start_views=729,accepted_fits=accepted,accepted_pilots=pilots,
            accepted_groups=groups,waiting=waiting,errors=errors,updated_at=time.time())
        atomic_write_json(result,out/'queue.json');atomic_write_json({k:v for k,v in result.items() if k!='tasks'},out/'status.json')
        return result


def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--once',action='store_true');args=a.parse_args();c=read(args.config)
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:tick(c)
            except Exception as e:atomic_write_json(dict(state='needs_review',error=repr(e),updated_at=time.time()),out/'status.json')
            if args.once or (out/'stop.json').exists():break
            time.sleep(300)

if __name__=='__main__':main()
