"""Dependency registry for both affine scopes; uses the sole project GPU owner."""
import argparse
import fcntl
from pathlib import Path
import time
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.runtime.mechanism_receipts import monitored_acceptance
from look.studies.project_case import read
from look.studies.affine_protocol import VERSION, protocol, matrix, counts
from look.studies.affine_case import verify_case
from look.studies import terminal_case, linear_case


def tick(config):
    from runtime.run_registry import reserve
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        declared={s:protocol(s) for s in ('terminal','progressive')};pp=out/'protocol.json'
        if pp.exists() and read(pp)!=declared:raise ValueError('Registered family changed')
        atomic_write_json(declared,pp);atomic_write_json(counts(),out/'counts.json')
        tasks=[];waiting={};errors=[];pilots={};complete=0
        for scope,feedkey,module in [('terminal','terminal_feed',terminal_case),('progressive','linear_feed',linear_case)]:
            feed=read(config[feedkey]);expected='look_terminal_work_feed_v1' if scope=='terminal' else 'look_linear_work_feed_v1'
            if feed.get('schema')!=expected or feed.get('test_access') is not False:raise ValueError('Unsealed reference feed')
            sources={}
            for row in feed['tasks']:
                root=Path(row['run_dir'])
                if not (root/'accepted.json').exists():continue
                try:
                    prior=read(row['spec'])
                    if file_sha256(row['spec'])!=row['spec_sha256']:raise ValueError('Reference spec changed')
                    monitored_acceptance(root,prior,module.verify_case,out/'verification_cache')
                    sources[stable_hash(prior['host'])]=dict(run_dir=str(root),spec_sha256=file_sha256(root/'spec.json'),accepted_sha256=file_sha256(root/'accepted.json'))
                except Exception as e:errors.append(dict(reference=str(root),error=repr(e)))
            for h in sorted(matrix(),key=lambda h:(h['seed'],h['disease'],h['architecture'],h['position'])):
                group=(scope,h['disease'],h['architecture'],h['position']);name='/'.join(map(str,(*group,h['seed'])))
                source=sources.get(stable_hash(h))
                if not source:waiting[name]='waiting_reference_acceptance';continue
                if h['seed']!=3416 and group not in pilots:waiting[name]='waiting_complete_3416_family';continue
                spec=dict(schema=VERSION,scope=scope,protocol=declared[scope],host=h,source=source,source_pins=config['source_pins'],test_access=False)
                if h['seed']!=3416:spec['pilot']=pilots[group]
                key=stable_hash(dict(host=h,scope=scope));path=out/'specs'/(key+'.json')
                if path.exists() and read(path)!=spec:raise ValueError('Affine identity changed')
                atomic_write_json(spec,path)
                run=Path(reserve(out,VERSION,key,spec,source=source,refresh_summary=False))
                if (run/'accepted.json').exists():
                    try:
                        monitored_acceptance(run,spec,verify_case,out/'verification_cache');complete+=1
                        if h['seed']==3416:pilots[group]=dict(run_dir=str(run),sha256=file_sha256(run/'accepted.json'))
                    except Exception as e:errors.append(dict(run=str(run),error=repr(e)));continue
                tasks.append(dict(id='affine/'+name,scope=scope,spec=str(path),spec_sha256=file_sha256(path),run_dir=str(run)))
        groups={}
        for row in tasks:
            root=Path(row['run_dir'])
            if not (root/'accepted.json').exists():continue
            s=read(row['spec']);h=s['host'];key=(s['scope'],h['disease'],h['architecture'],h['position'])
            groups.setdefault(key,[]).append(root)
        for key,runs in groups.items():
            if len(runs)!=3:continue
            from look.runtime.affine_analysis import request
            request(config,runs,out/'reports'/stable_hash(key))
        result=dict(schema='look_affine_work_feed_v1',test_access=False,tasks=tasks,complete_cases=complete,
            accepted_pilots=len(pilots),expected_cases=162,waiting=waiting,errors=errors,updated_at=time.time())
        atomic_write_json(result,out/'queue.json');atomic_write_json({k:v for k,v in result.items() if k!='tasks'},out/'status.json');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    config=read(a.config);out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    if a.once:tick(config);return
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (out/'stop.json').exists():
            try:tick(config)
            except Exception as e:atomic_write_json(dict(state='needs_review',error=repr(e),updated_at=time.time()),out/'status.json')
            time.sleep(300)

if __name__=='__main__':main()
