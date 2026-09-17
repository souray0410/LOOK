"""Explicit immutable-source migration of verified best-forward candidate evidence.

This is an offline conversion, never a fallback in the current runtime reader.
The original execution remains untouched. Only source pins may change.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.studies.search_protocol import validate


def migrate(source, destination, new_spec, equivalence_receipt):
    source=Path(source).resolve(); destination=Path(destination).resolve()
    if destination.exists():raise ValueError('Migration destination must be new')
    old_spec=json.loads((source/'spec.json').read_text());validate(old_spec);validate(new_spec)
    a=copy.deepcopy(old_spec);b=copy.deepcopy(new_spec)
    a.pop('source_pins');b.pop('source_pins')
    if a!=b or old_spec['mode']!='best_forward':raise ValueError('Only source-only best-forward migration is permitted')
    receipt=json.loads(Path(equivalence_receipt).read_text())
    if receipt.get('state')!='accepted' or receipt.get('test_access') is not False:
        raise ValueError('Accepted equivalence evidence required')
    if not receipt.get('records') or not all(all(r.get(k) is True for k in ('exact_moments','exact_parameters','exact_logits')) for r in receipt['records']):
        raise ValueError('Incomplete equivalence evidence')
    if receipt.get('source_identity')!=stable_hash(new_spec):raise ValueError('Equivalence probe used a different execution')
    tested=receipt.get('tested_source_sha256',{})
    for pin in new_spec['source_pins']:
        suffix='src/look/'+pin['path'].split('/src/look/')[-1]
        if '/src/look/' in pin['path'] and tested.get(suffix)!=pin['sha256']:raise ValueError('New source lacks exact probe acceptance')
        if file_sha256(pin['path'])!=pin['sha256']:raise ValueError('New source changed')
    with (source/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (source/'accepted.json').exists():raise ValueError('Completed runs must be reused, not migrated')
        destination.mkdir(parents=True)
        audit=[]
        for old_root in sorted((source/'corrections').glob('*/factors/*')):
            if not (old_root/'contract.json').exists():continue
            new_root=destination/old_root.relative_to(source);new_root.mkdir(parents=True)
            old_contract=json.loads((old_root/'contract.json').read_text());new_contract=copy.deepcopy(old_contract)
            if old_contract['identity']['case']!=stable_hash(old_spec):raise ValueError('Old contract identity mismatch')
            new_contract['identity']['case']=stable_hash(new_spec)
            atomic_write_json(new_contract,new_root/'contract.json')
            old_decisions=[];new_decisions=[];start=0
            def copy_row(row):
                converted=copy.deepcopy(row)
                artifact=(old_root/row['artifact']).resolve()
                if not artifact.is_relative_to(old_root) or file_sha256(artifact)!=row['sha256']:raise ValueError('Old artifact mismatch')
                target=new_root/row['artifact'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(artifact,target)
                e=converted['evidence'];prediction=Path(e['prediction'])
                if e.get('role')!='development' or file_sha256(prediction)!=e['sha256']:raise ValueError('Old prediction mismatch')
                target=new_root/'predictions'/prediction.name;target.parent.mkdir(exist_ok=True);shutil.copy2(prediction,target)
                e['prediction']=str(target)
                return converted
            def copy_baseline(e):
                e=copy.deepcopy(e);prediction=Path(e['prediction'])
                if e.get('role')!='development' or file_sha256(prediction)!=e['sha256']:raise ValueError('Old baseline mismatch')
                target=new_root/'predictions'/prediction.name;target.parent.mkdir(exist_ok=True);shutil.copy2(prediction,target);e['prediction']=str(target)
                return e
            for folder in sorted((old_root/'rounds').glob('*')):
                if not folder.is_dir():continue
                rid=stable_hash(dict(contract=old_contract,upstream=old_decisions,start=start))
                nrid=stable_hash(dict(contract=new_contract,upstream=new_decisions,start=start))
                new_folder=new_root/'rounds'/folder.name
                for cache in sorted(folder.glob('site_*.json')):
                    index=int(cache.stem.split('_')[1]);saved=json.loads(cache.read_text())
                    if saved['identity']!=stable_hash(dict(round=rid,index=index)):raise ValueError('Old candidate identity mismatch')
                    converted=dict(identity=stable_hash(dict(round=nrid,index=index)),candidates=[copy_row(r) for r in saved['candidates']])
                    atomic_write_json(converted,new_folder/cache.name)
                    audit.append(dict(source=str(cache),source_sha256=file_sha256(cache),target=str(new_folder/cache.name),target_sha256=file_sha256(new_folder/cache.name)))
                dp=folder/'decision.json'
                if not dp.exists():break
                d=json.loads(dp.read_text())
                if d['identity']!=rid:raise ValueError('Old decision identity mismatch')
                nd=copy.deepcopy(d);nd.update(identity=nrid,baseline=copy_baseline(d['baseline']),candidates=[copy_row(r) for r in d['candidates']],winner=copy_row(d['winner']))
                atomic_write_json(nd,new_folder/'decision.json');old_decisions.append(d);new_decisions.append(nd)
                if not d['enabled']:break
                start=d['winner']['index']+1
        atomic_write_json(new_spec,destination/'spec.json')
        record=dict(schema='look_source_only_migration_v1',state='converted_pending_replay',source=str(source),
            old_identity=stable_hash(old_spec),new_identity=stable_hash(new_spec),
            equivalence_receipt=str(equivalence_receipt),equivalence_sha256=file_sha256(equivalence_receipt),
            reused_candidates=len(audit),files=audit,test_access=False)
        atomic_write_json(record,destination/'migration.json');return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--spec',required=True);p.add_argument('--equivalence-receipt',required=True);a=p.parse_args()
    print(json.dumps(migrate(a.source,a.destination,json.loads(Path(a.spec).read_text()),a.equivalence_receipt)))

if __name__=='__main__':main()
