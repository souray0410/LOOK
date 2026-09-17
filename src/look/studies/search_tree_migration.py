"""Explicit root-only evidence conversion to a new positive-tree identity.

Committed zero-upstream evidence can be read while the old execution continues.
All copied files are checked before and after reading. No old run is modified;
no corrected-prefix statistics or old winning path is imported.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import tempfile
from look.runtime.state import atomic_write_json,file_sha256,stable_hash
from look.methods.positive_forward_tree import tree_contract,prefix_identity,candidate_identity
from look.studies.search_protocol import validate


def read(path):return json.loads(Path(path).read_text())


def numerical_pins(spec):
    result={}
    for p in spec['source_pins']:
        if '/src/look/' not in p['path']:continue
        name=p['path'].split('/src/look/',1)[1]
        if (name.startswith(('data/','models/','evaluation/')) or
                name in ('methods/operator.py','methods/joint.py','methods/shared_latent.py',
                         'methods/linear_operator.py')):
            if name in result:raise ValueError('Duplicate numerical source pin')
            if file_sha256(p['path'])!=p['sha256']:raise ValueError('Numerical source changed')
            result[name]=p['sha256']
    if not {'methods/operator.py','methods/shared_latent.py'}.issubset(result):
        raise ValueError('Incomplete numerical source pins')
    return result


def migrate(source,destination,new_spec):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination:raise ValueError('New scientific run required')
    old_spec=read(source/'spec.json');validate(old_spec);validate(new_spec)
    a=copy.deepcopy(old_spec);b=copy.deepcopy(new_spec)
    for key in ('mode','protocol','source_pins'):a.pop(key);b.pop(key)
    if a!=b or old_spec['mode']!='best_forward' or new_spec['mode']!='positive_forward_tree':
        raise ValueError('Root migration changed host/data/operator/settings')
    if numerical_pins(old_spec)!=numerical_pins(new_spec):raise ValueError('Root numerical implementation changed')
    for p in new_spec['source_pins']:
        if file_sha256(p['path'])!=p['sha256']:raise ValueError('New source changed')
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (destination/'migration.json').exists() or (destination/'corrections').exists():raise ValueError('Destination already imported')
        if (destination/'spec.json').exists() and read(destination/'spec.json')!=new_spec:raise ValueError('Destination identity differs')
        stage=Path(tempfile.mkdtemp(prefix='.root-import-',dir=destination));audit=[];imports=[]
        def copied(path,where,digest):
            path=Path(path)
            if file_sha256(path)!=digest:raise ValueError('Source evidence changed')
            where.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,where)
            if file_sha256(path)!=digest or file_sha256(where)!=digest:raise ValueError('Evidence changed while copying')
            audit.append(dict(source=str(path),sha256=digest))
        try:
            for old_root in sorted((source/'corrections').glob('*/factors/*')):
                decision=old_root/'rounds/000/decision.json'
                if not decision.exists():continue
                decision_sha=file_sha256(decision);old_contract=read(old_root/'contract.json');d=read(decision)
                if old_contract['identity']['case']!=stable_hash(old_spec) or old_contract['mode']!='best_forward':raise ValueError('Original root contract mismatch')
                rid=stable_hash(dict(contract=old_contract,upstream=[],start=0))
                if d['identity']!=rid:raise ValueError('Not a zero-upstream root')
                identity=dict(old_contract['identity'],case=stable_hash(new_spec))
                contract=tree_contract(identity,old_contract['sites']);relative=old_root.relative_to(source)
                target=stage/relative;final=destination/relative
                atomic_write_json(contract,target/'contract.json')
                def evidence(e):
                    e=copy.deepcopy(e)
                    if e.get('role')!='development' or e.get('data_role')!='development':raise ValueError('Only formal development evidence')
                    old=Path(e['prediction']);rel=Path('predictions')/old.name
                    copied(old,target/rel,e['sha256']);e['prediction']=str(final/rel);return e
                baseline=evidence(d['baseline'])
                atomic_write_json(dict(identity=prefix_identity(contract,[]),evidence=baseline),target/'prefixes/root/baseline.json')
                candidates=[]
                for index,site in enumerate(old_contract['sites']):
                    cache=old_root/'rounds/000'/f'site_{index:03d}.json';cache_sha=file_sha256(cache);saved=read(cache)
                    if saved['identity']!=stable_hash(dict(round=rid,index=index)):raise ValueError('Original candidate identity differs')
                    converted=[]
                    for row in saved['candidates']:
                        if row['index']!=index or row['node']!=site or row['score']!=row['evidence']['score']:raise ValueError('Original candidate descriptor differs')
                        artifact=(old_root/row['artifact']).resolve()
                        if not artifact.is_relative_to(old_root):raise ValueError('Artifact outside original factor')
                        rel=Path('prefixes/root/artifacts')/artifact.name;copied(artifact,target/rel,row['sha256'])
                        nr=copy.deepcopy(row);nr.update(artifact=str(rel),evidence=evidence(row['evidence']));converted.append(nr)
                    if not converted or saved['candidates']!=[r for r in d['candidates'] if r['index']==index]:raise ValueError('Original committed root rows differ')
                    if file_sha256(cache)!=cache_sha:raise ValueError('Candidate metadata changed during copy')
                    atomic_write_json(dict(identity=candidate_identity(contract,[],index),candidates=converted),target/'prefixes/root'/cache.name)
                    candidates.extend(converted);audit.append(dict(source=str(cache),sha256=cache_sha))
                if file_sha256(decision)!=decision_sha:raise ValueError('Root decision changed during copy')
                audit.append(dict(source=str(decision),sha256=decision_sha))
                imports.append(dict(pattern=identity['pattern'],factor=identity['factor'],candidates=len(candidates),
                    positive_sites=sum(max(r['score'] for r in candidates if r['index']==i)>baseline['score'] for i in range(len(old_contract['sites'])))))
            if not imports:raise ValueError('No complete committed root to import')
            (stage/'corrections').rename(destination/'corrections')
            atomic_write_json(new_spec,destination/'spec.json')
            record=dict(schema='look_positive_tree_root_import_v1',state='converted_pending_tree_execution',
                source=str(source),old_identity=stable_hash(old_spec),new_identity=stable_hash(new_spec),
                imported=imports,source_evidence=audit,numerical_sources=numerical_pins(new_spec),
                nonroot_imported=False,old_winner_not_transferred=True,test_access=False)
            atomic_write_json(record,destination/'migration.json');return record
        finally:shutil.rmtree(stage,ignore_errors=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--spec',required=True);a=p.parse_args();print(json.dumps(migrate(a.source,a.destination,read(a.spec))))


if __name__=='__main__':main()
