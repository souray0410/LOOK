import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from look_core.start_study import (make_cases, sites_for_fusion, choose_start, verify_parent,
    file_record, verify_file, SuffixRunner, audit_case, lock_case_identity)
from look_core.start_report import report_rows, three_seed_rows
from look_core import look
from look_core.state import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def spec():
    return json.loads((ROOT/'configs/unified_study.json').read_text())


def test_grid_exact_counts_and_reuse_semantics():
    cases = make_cases(spec(), ['layer3','feature','layer2'])
    assert len(cases) == len({c['case_id'] for c in cases}) == 243
    assert sum(c['reuse_stage'] is None for c in cases) == 213
    assert sum(c['start_ordinal'] == 1 for c in cases) == 27
    for c in cases:
        assert len(c['candidate_sites']) == 9
        assert c['eligible_sites'] == c['candidate_sites'][c['start_ordinal']-1:]
        assert c['eligible_sites'][0] == c['allowed_start']
    assert cases == make_cases(spec(), ['layer3','feature','layer2'])


@pytest.mark.parametrize('fusion', ['layer3','feature','layer2'])
def test_canonical_sites_equal_actual_graph(fusion):
    from look_core.graph import build_resnet50_mhd_graph
    from look_core.joint import correction_sites
    graph = build_resnet50_mhd_graph(fusion, batch_size=1, image_size=224, pretrained=False, device='cpu')
    assert sites_for_fusion(fusion) == correction_sites(graph)


def test_choose_start_ties_earliest_and_requires_complete_grid():
    rows = [dict(start_ordinal=i, macro_f1=.7) for i in range(9,0,-1)]
    assert choose_start(rows)['start_ordinal'] == 1
    rows[3]['macro_f1'] = .71
    assert choose_start(rows)['start_ordinal'] == rows[3]['start_ordinal']
    with pytest.raises(ValueError): choose_start(rows[:-1])
    rows[3]['macro_f1'] = float('nan')
    with pytest.raises(ValueError): choose_start(rows)


def test_parent_gate_cannot_bypass_incomplete_failed_or_test_access():
    p = ROOT/'configs/unified_study.json'
    valid = dict(protocol=spec()['protocol'],spec_sha256=file_sha256(p),test_access=False,status='complete',
        completed_stages={str(i):dict(status='complete') for i in range(52)})
    verify_parent(valid, spec(), p)
    for change in [dict(status='running'),dict(status='failed'),dict(test_access=True),dict(completed_stages={}),dict(spec_sha256='wrong')]:
        with pytest.raises(RuntimeError): verify_parent(dict(valid,**change),spec(),p)


def test_pinned_file_changes_fail_without_replacement(tmp_path):
    p = tmp_path/'best.pt'; p.write_bytes(b'original')
    record=file_record(p); assert verify_file(record)==p
    p.write_bytes(b'modified')
    with pytest.raises(RuntimeError,match='replacement forbidden'): verify_file(record)
    assert p.read_bytes()==b'modified'
    p.unlink()
    with pytest.raises(RuntimeError): verify_file(record)


def test_suffix_resource_hooks_never_fit_backbone_or_generator(tmp_path):
    r=SuffixRunner.__new__(SuffixRunner)
    p=tmp_path/'frozen.pt'; p.write_bytes(b'frozen')
    r.source=dict(checkpoint=file_record(p),filling={'strategy':'raw_zero'},generators={})
    r.selection=SimpleNamespace(filling_strategy='raw_zero')
    assert r._train_or_resume()==p
    filler,_=r._prepare_filler({})
    assert filler.name=='raw_zero'
    r.selection=SimpleNamespace(filling_strategy='paired_cgan')
    with pytest.raises(KeyError):r._prepare_filler({})
    p.unlink()
    with pytest.raises(RuntimeError):r._train_or_resume()


def artifact(node):
    return look.LOOKArtifact(node,'oct_missing','normalized_mean',1,1,(2,),(2,),torch.zeros(2),torch.ones(2),
        torch.zeros(2),torch.ones(1,2),torch.zeros(1,1),torch.ones(1),1.,0.,0.)


def test_suffix_refit_upstream_optional_off_and_cross_start_resume_rejected(monkeypatch,tmp_path):
    import look_core.evaluate as ev
    import look_core.matrix_analysis as ma
    calls=[]
    def fit(**kw):
        calls.append((kw['node_name'],[a.node_name for a in kw['upstream_artifacts']]))
        return {1:artifact(kw['node_name'])}
    def evaluate(*args,artifact_banks,**kwargs):
        nodes=[a.node_name for a in artifact_banks['oct_missing']]
        score=.5 + .1*('b' in nodes) + .1*('c' in nodes)
        return dict(metrics={'macro_f1':score}, labels=np.array([0,1]), probabilities=np.eye(2),
            logits=np.eye(2), scores=np.array([-1,1]),participant_ids=np.array(['a','b']),patterns=np.array(['x','x']))
    monkeypatch.setattr(look,'fit_look_node',fit)
    monkeypatch.setattr(ev,'evaluate_missing',evaluate)
    monkeypatch.setattr(ma,'analyze_look_bank',lambda *a: [])
    pca={(n,1):SimpleNamespace(feature_shape=(2,),source_id=n) for n in ['a','b','c']}
    args=(None,None,None,'oct_missing',['a','b','c'],4,[1],2,torch.device('cpu'),tmp_path/'first')
    bank,_,_=look._fit_factor_bank(*args,pca_bank=pca)
    assert calls==[('a',[]),('b',[]),('c',['b'])]  # Start a is allowed but OFF on a tie.
    assert [a.node_name for a in bank]==['b','c']
    calls.clear()
    suffix=(None,None,None,'oct_missing',['b','c'],4,[1],2,torch.device('cpu'),tmp_path/'second')
    look._fit_factor_bank(*suffix,pca_bank=pca)
    assert calls==[('b',[]),('c',['b'])]
    calls.clear(); look._fit_factor_bank(*suffix,pca_bank=pca)
    assert calls==[]
    path=tmp_path/'second'/'suffix_identity.json'
    lock_case_identity(path, {'eligible_sites':['b','c']}, {'source':'fixed'})
    with pytest.raises(ValueError,match='Cross-start'):
        lock_case_identity(path, {'eligible_sites':['a','b','c']}, {'source':'fixed'})


def test_report_requires_all_seeds_and_preserves_negative_delta():
    rows=[dict(fusion='feature',filling='raw_zero',pattern='oct_missing',start_ordinal=2,
        seed=3407,macro_f1=.6,delta_f1_pp=-2.),dict(fusion='feature',filling='raw_zero',pattern='oct_missing',start_ordinal=2,
        seed=3408,macro_f1=.7,delta_f1_pp=1.)]
    assert three_seed_rows(rows)==[]
    rows.append(dict(rows[0],seed=3409,macro_f1=.8,delta_f1_pp=-2.))
    got=three_seed_rows(rows)[0]
    assert got['mean_macro_f1']==pytest.approx(.7)
    assert got['sd_macro_f1']==pytest.approx(.1)
    assert got['mean_delta_f1_pp']==pytest.approx(-1.)


def test_selected_directions_are_combined_without_refitting_or_ratio_selection(monkeypatch,tmp_path):
    import look_core.start_study as module
    records=[]
    for i in range(1,10):
        banks={}
        for pattern in module.PATTERNS:
            banks[pattern]=dict(macro_f1=.8 if i == (2 if pattern=='oct_missing' else 5) else .5,
                bank_root=f'{pattern}/{i}',first_enabled=None,selected_factor=16,
                selected_manifest={'path':f'{pattern}/{i}/selected_manifest.json'},artifacts=[])
        records.append(dict(case_id=f'start{i}',start_ordinal=i,allowed_start=f'n{i}',banks=banks,result={'path':f'result{i}'}))
    class FakeRunner:
        def __init__(self,*a):pass
        def _build_loaders(self):return {'validation':'validation_only'},{}
        def _train_or_resume(self):return 'pinned'
        def _load_frozen_graph(self,path):assert path=='pinned';return None,None
        def _prepare_filler(self,d):return None,None
        def _evaluate_split(self,graph,loader,split,filler,banks):
            assert loader=='validation_only' and split=='validation'
            assert banks=={'oct_missing':'oct_missing/2','cfp_missing':'cfp_missing/5'}
            return {f'look_after_fill_{p}':{'metrics':{'macro_f1':.8}} for p in module.PATTERNS}
    monkeypatch.setattr(module,'SuffixRunner',FakeRunner)
    monkeypatch.setattr(module,'load_selected_bank',lambda p:str(p))
    got=module.evaluate_selected('context',records,{'selection':{'seed':3407}},tmp_path,torch.device('cpu'),(0,1))
    frozen=json.loads((tmp_path/'selected_starts/context/frozen_selection.json').read_text())
    assert frozen['directions']['oct_missing']['start_ordinal']==2
    assert frozen['directions']['cfp_missing']['start_ordinal']==5
    monkeypatch.setattr(FakeRunner,'_evaluate_split',lambda *a:pytest.fail('completed frozen evaluation repeated'))
    assert module.evaluate_selected('context',records,{'selection':{'seed':3407}},tmp_path,torch.device('cpu'),(0,1))==got
