"""Content-bound, dependency-driven supplement feed using the shared registrar."""
import argparse
import fcntl
import json
from pathlib import Path
import time
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.studies.mechanism_protocol import matrix, protocol, VERSION


def read(path): return json.loads(Path(path).read_text())


def tick(config):
    from runtime.run_registry import reserve
    from look.studies.project_case import verify_case
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks=matrix();p=protocol()
        frozen=root/'protocol.json'
        if frozen.exists() and read(frozen)!=p:raise ValueError('Locked protocol changed')
        atomic_write_json(p,frozen)
        atomic_write_json(tasks,root/'matrix.json')
        base=read(config['base_feed'])
        if base.get('test_access') is not False or base.get('schema')!='look_project_work_feed_v1':
            raise ValueError('Unsealed base feed')
        accepted={};errors=[]
        cache=read(root/'base_acceptance_cache.json') if (root/'base_acceptance_cache.json').exists() else {}
        for item in base['tasks']:
            run=Path(item['run_dir']);receipt=run/'accepted.json'
            if not receipt.exists():continue
            try:
                if file_sha256(item['spec'])!=item['spec_sha256']:raise ValueError('Base specification changed')
                s=read(item['spec']);digest=file_sha256(receipt)
                # A changed receipt invalidates cached acceptance. Workers still
                # verify every actually consumed parent/basis file at use time.
                if cache.get(str(run))!=digest:verify_case(run,s);cache[str(run)]=digest
                key=(s['disease'],s['model']['name'],s['position'],s['seed'])
                accepted[key]=dict(run_dir=str(run),spec_path=item['spec'],spec_sha256=item['spec_sha256'],
                    accepted_sha256=digest,best_sha256=read(receipt)['files']['host/best.pt'])
            except Exception as exc:errors.append(dict(run=str(run),error=repr(exc)))
        ready=[];waiting=0
        for task in tasks:
            h=task['host'];position=h.get('position','middle')
            key=(h['disease'],h['architecture'],position,h['seed'])
            if key not in accepted:waiting+=1;continue
            source=accepted[key]
            spec=dict(schema=VERSION,task=task,protocol=p,source=source,
                source_pins=config['source_pins'],test_access=False)
            path=root/'specs'/(task['id']+'.json');path.parent.mkdir(exist_ok=True)
            if path.exists() and read(path)!=spec:raise ValueError('Registered task provenance changed')
            atomic_write_json(spec,path)
            run=reserve(root,VERSION,task['id'],spec,source=source,refresh_summary=False)
            ready.append(dict(id=task['id'],spec=str(path),spec_sha256=file_sha256(path),
                              run_dir=str(run),execution='look_mechanism'))
        atomic_write_json(cache,root/'base_acceptance_cache.json')
        feed=dict(schema='look_mechanism_work_feed_v1',test_access=False,tasks=ready,
            expected=len(tasks),waiting_dependencies=waiting,protocol_sha256=file_sha256(frozen))
        atomic_write_json(feed,root/'queue.json')
        from look.analysis.mechanism_report import progress
        summary=progress(feed,root/'report')
        if summary['complete']:
            from look.runtime.mechanism_analysis import request
            summary['statistics']=request(config)
        from look.evaluation.mechanism_test import prepare
        prepare(feed,base,root/'report'/'test_preparation.json')
        summary.update(accepted_base_hosts=len(accepted),errors=errors,time=time.time())
        atomic_write_json(summary,root/'status.json')
        return summary


def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--once',action='store_true')
    args=a.parse_args();config=read(args.config)
    if args.once:print(json.dumps(tick(config)));return
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'stop.json').exists():
            try:tick(config)
            except Exception as exc:atomic_write_json(dict(state='needs_review',error=repr(exc),time=time.time()),root/'status.json')
            time.sleep(1800)

if __name__=='__main__':main()
