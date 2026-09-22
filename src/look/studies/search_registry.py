"""Finite matched search feed using the existing reservation/claim system."""
import argparse
import fcntl
import time
from pathlib import Path
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.studies.project_case import read
from look.studies.search_protocol import VERSION, protocol, sites, representative_starts
from look.studies.search_case import verify_case
from look.runtime.mechanism_receipts import monitored_acceptance


def tick(config):
    from mhd_models.runtime.run_registry import reserve
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        feed=read(config['base_feed'])
        if feed.get('schema')!='look_project_work_feed_v1' or feed.get('test_access') is not False:
            raise ValueError('Unsealed host feed')
        tasks=[];waiting={};errors=[];accepted=0;complete=[];by_host={}
        rows=sorted(feed['tasks'],key=lambda r:(read(r['spec'])['seed'],r['id']))
        weekly = None
        if config.get('weekly_delivery_policy'):
            from look.runtime.weekly_delivery import release_state
            weekly = release_state(config['weekly_delivery_policy'])
        for row in rows:
            base=read(row['spec']);root=Path(row['run_dir'])
            h=dict(disease=base['disease'],architecture=base['model']['name'],position=base['position'],seed=base['seed'])
            name='/'.join(map(str,h.values()));key=stable_hash(h)
            if weekly and not weekly['released'] and h['seed'] != weekly['first_seed']:
                waiting[name] = weekly['reason']; continue
            if not (root/'host/accepted.json').exists():waiting[name]='waiting_accepted_host';continue
            manifests=[m for m in root.glob('pca/*/bank_manifest.json') if read(m)['identity'].get('case')==stable_hash(base)]
            if not manifests:waiting[name]='waiting_frozen_complete_train_PCA';continue
            try:
                if len(manifests)!=1 or file_sha256(row['spec'])!=row['spec_sha256']:raise ValueError('Ambiguous basis or changed host spec')
                hr=read(root/'host/accepted.json')
                if hr.get('state')!='accepted' or hr.get('identity')!=stable_hash(base):raise ValueError('Host receipt mismatch')
                pilot=[]
                if h['seed']!=3416:
                    pilot=by_host.get(stable_hash(dict(h,seed=3416)),[])
                    if len(pilot)!=(5 if config.get('representative_starts') else 2):waiting[name]='waiting_matched_first_seed_technical_acceptance';continue
                source=dict(run_dir=str(root),spec_path=row['spec'],spec_sha256=row['spec_sha256'],
                    host_accepted_sha256=file_sha256(root/'host/accepted.json'),best_sha256=hr['files']['best.pt'])
                ordered=sites(h['architecture'],h['position']);done=[]
                routes = [('best_forward',1),('greedy',1)]
                if config.get('representative_starts'):
                    routes += [('greedy',i) for i in representative_starts(h['architecture'],h['position'])[1:]]
                for mode,start in routes:
                    spec=dict(schema=VERSION,protocol=protocol(),host=h,mode=mode,candidate_sites=ordered,eligible_sites=ordered,
                        source=source,pca=dict(path=str(manifests[0]),sha256=file_sha256(manifests[0])),
                        source_pins=config['source_pins'],test_access=False)
                    if start != 1:
                        spec.update(start_ordinal=start,eligible_sites=ordered[start-1:],
                                    source_pins=config['representative_source_pins'])
                    if pilot:spec['pilot']=pilot
                    identity=dict(host=h,mode=mode)
                    if config.get('spatial_factors'):
                        spec['spatial_factors']=config['spatial_factors']
                        identity['spatial_factors']=config['spatial_factors']
                    if config.get('latent_dims'):
                        spec['latent_dims']=config['latent_dims']
                        identity['latent_dims']=config['latent_dims']
                    if start != 1:identity['start_ordinal']=start
                    taskid=stable_hash(identity);sp=out/'specs'/(taskid+'.json')
                    if sp.exists() and read(sp)!=spec:raise ValueError('Registered scientific identity changed')
                    atomic_write_json(spec,sp)
                    run=Path(reserve(out,VERSION,taskid,spec,source=source,refresh_summary=False))
                    task=dict(id='search/'+name+'/'+mode+('' if start==1 else '/start'+str(start)),spec=str(sp),spec_sha256=file_sha256(sp),
                        run_dir=str(run),execution='look_search',search_mode=mode)
                    if (run/'accepted.json').exists():
                        monitored_acceptance(run,spec,verify_case,out/'cache');accepted+=1
                        done.append(dict(path=str(run),sha256=file_sha256(run/'accepted.json')))
                    tasks.append(task)
                if len(done)==len(routes):
                    by_host[key]=done;complete.append(dict(host=h,cases=done))
                    if config.get('representative_starts') and config.get('analysis'):
                        from look.runtime.suffix_analysis import request
                        request(config['analysis'],dict(analysis_kind='search',host=h,runs=[r['path'] for r in done],
                            output=str(out/'reports'/key),test_access=False))
            except Exception as e:errors.append(dict(host=name,error=repr(e)))
        result=dict(schema='look_search_work_feed_v1',test_access=False,tasks=tasks,accepted_cases=accepted,
            matched_groups=complete,waiting=waiting,errors=errors,updated_at=time.time(),
            expected_hosts=81,expected_search_trajectories=405 if config.get("representative_starts") else 162,all_starts_required=False)
        atomic_write_json(result,out/'queue.json')
        atomic_write_json({k:v for k,v in result.items() if k!='tasks'},out/'status.json')
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
