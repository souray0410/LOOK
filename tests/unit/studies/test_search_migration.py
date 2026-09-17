import copy
import fcntl
import json
from pathlib import Path
import pytest
from look.methods.independent_greedy import fit_trajectory,SelectionPaused
from look.runtime.state import atomic_write_json,file_sha256,stable_hash
from look.studies.search_protocol import protocol,sites,VERSION
from look.studies.search_migration import migrate


def fixture(tmp_path):
    ordered=sites('resnet50','deep')
    source=tmp_path/'old';source.mkdir()
    p=tmp_path/'old.py';p.write_text('old');q=tmp_path/'new.py';q.write_text('new')
    spec=dict(schema=VERSION,protocol=protocol(),test_access=False,
        host=dict(disease='cataract',architecture='resnet50',position='deep',seed=3416),
        mode='best_forward',candidate_sites=ordered,eligible_sites=ordered,
        source_pins=[dict(path=str(p),sha256=file_sha256(p))])
    atomic_write_json(spec,source/'spec.json')
    new=copy.deepcopy(spec);new['source_pins']=[dict(path=str(q),sha256=file_sha256(q))]
    receipt=tmp_path/'receipt.json';atomic_write_json(dict(state='accepted',test_access=False,
        source_identity=stable_hash(new),records=[dict(exact_moments=True,exact_parameters=True,exact_logits=True)]),receipt)
    return source,spec,new,receipt


def trajectory(root,spec,limit=None):
    folder=root/'corrections/oct_missing/factors/x16';calls=[]
    def fit(site,bank,path):
        if limit is not None and len(calls)>=limit:raise SelectionPaused()
        calls.append(site);return [('q32',dict(node=site))]
    def evaluate(bank):
        score=0 if not bank else (1. if bank[0]['node']==spec['eligible_sites'][0] else .5)-.1*(len(bank)-1)
        p=root/'predictions'/f'{stable_hash(bank)}.json';atomic_write_json(dict(score=score),p)
        return dict(role='development',score=score,prediction=str(p),sha256=file_sha256(p))
    b,r=fit_trajectory(identity=dict(case=stable_hash(spec),pattern='oct_missing',factor=16,latent_dims=[32],max_rank=32),
        sites=spec['eligible_sites'],mode='best_forward',output=folder,fit_candidates=fit,evaluate=evaluate,
        save_artifact=lambda a,p:atomic_write_json(a,p),load_artifact=lambda p:json.loads(p.read_text()))
    return b,r,calls


def test_partial_conversion_reuses_verified_candidates(tmp_path):
    old,spec,new,receipt=fixture(tmp_path)
    with pytest.raises(SelectionPaused):trajectory(old,spec,limit=3)
    before={str(p.relative_to(old)):file_sha256(p) for p in old.rglob('*') if p.is_file()}
    target=tmp_path/'new';r=migrate(old,target,new,receipt)
    assert r['reused_candidates']==3
    b,result,calls=trajectory(target,new)
    assert calls[:6]==new['eligible_sites'][3:]
    assert b==[dict(node=new['eligible_sites'][0])]
    assert result['final']['score']==1.
    for name,sha in before.items():assert file_sha256(old/name)==sha


def test_completed_round_identity_chain_migrates(tmp_path):
    old,spec,new,receipt=fixture(tmp_path)
    with pytest.raises(SelectionPaused):trajectory(old,spec,limit=11)
    target=tmp_path/'new';migrate(old,target,new,receipt)
    b,result,calls=trajectory(target,new)
    assert calls==new['eligible_sites'][3:]
    assert len(result['decisions'])==2 and result['final']['score']==1.


def test_reject_active_source_and_scientific_change(tmp_path):
    old,spec,new,receipt=fixture(tmp_path)
    with (old/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):migrate(old,tmp_path/'new',new,receipt)
    changed=copy.deepcopy(new);changed['worker_threads']=1
    with pytest.raises(ValueError,match='source-only'):migrate(old,tmp_path/'new',changed,receipt)
