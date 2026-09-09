from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from look_core.self_input import neutralize_retained,apply_self_input,POLICY
from look_core.self_input_study import make_cases,verify_predecessor
from look_core.method_report import self_input_comparisons
from look_core import method_kernels as k
from test_joint_protocol import artifact


def basis(pattern='oct_missing',node='joint_stem'):
    torch.manual_seed(15)
    return replace(artifact(node,2),missing_pattern=pattern,feature_shape=(5,2,2),downsample_shape=(5,2,2),
        mean=torch.randn(20),std=torch.rand(20)+.5,pca_mean=torch.randn(20),
        components=torch.randn(2,20),weight=torch.randn(2,2),bias=torch.randn(2),
        member_names=('oct_'+node[6:],'cfp_'+node[6:]),
        member_shapes=((2,2,2),(3,2,2)) if node!='joint_input' else ((2,2,2,2),(2,3,2,2)))


@pytest.mark.parametrize('pattern',['oct_missing','cfp_missing'])
@pytest.mark.parametrize('node',['joint_input','joint_stem'])
def test_missing_predictor_is_invariant_to_retained_sample(pattern,node):
    a=basis(pattern,node);x=torch.randn(4,5,2,2);other=x.clone()
    retained=slice(2,None) if pattern=='oct_missing' else slice(0,2)
    missing=slice(0,2) if pattern=='oct_missing' else slice(2,None)
    other[:,retained]=torch.randn_like(other[:,retained])*100
    assert torch.equal(neutralize_retained(x.flatten(1),a,pattern),neutralize_retained(other.flatten(1),a,pattern))
    first=apply_self_input(x,a);second=apply_self_input(other,a)
    assert torch.equal(first[:,missing],second[:,missing])
    changed=x.clone();changed[:,missing]+=1
    assert not torch.equal(apply_self_input(changed,a)[:,missing],first[:,missing])


def test_fused_site_exactly_uses_original_kernel():
    a=replace(basis(),node_name='fusion_layer3');x=torch.randn(4,5,2,2)
    from look_core.look import apply_artifact
    assert torch.equal(apply_self_input(x,a),apply_artifact(x,a))


def test_explicit_member_metadata_and_geometry_required():
    a=basis()
    for b in [replace(a,member_names=()),replace(a,member_shapes=((8,2,2),(3,2,2)))]:
        with pytest.raises(ValueError):neutralize_retained(torch.randn(4,20),b,'oct_missing')
    with pytest.raises(ValueError):neutralize_retained(torch.randn(4,20),a,'complete')


@pytest.mark.parametrize('pattern',['oct_missing','cfp_missing'])
def test_fitting_uses_only_missing_sample_and_inherits_upstream(monkeypatch,pattern):
    a=basis(pattern);pca=SimpleNamespace(**vars(a),explained_variance_ratio=torch.ones(2),fit_seconds=0,peak_rss_bytes=0,source_id='same-pca')
    gen=torch.Generator().manual_seed(44)
    full=torch.randn(6,5,2,2,generator=gen);miss=full.clone()
    missing=slice(0,2) if pattern=='oct_missing' else slice(2,None)
    retained=slice(2,None) if pattern=='oct_missing' else slice(0,2)
    miss[:,missing]=0
    calls=[]
    def forward(*args,**kwargs):
        calls.append((len(args)>3 and args[3],kwargs.get('policy')))
        return full if len(args)==3 else miss
    monkeypatch.setattr(k,'forward_control',forward)
    graph=SimpleNamespace(eval=lambda:None)
    filler=SimpleNamespace(name='normalized_mean',fill=lambda o,c,p:(o,c))
    batch={'oct':torch.zeros(6,1),'cfp':torch.zeros(6,1)}
    upstream=[artifact()]
    def fit():return k.fit_candidates(graph,[batch],'joint_stem',pattern,1,[1,2],2,'cpu',pca,upstream,filler,POLICY)
    first=fit()
    full[:,retained]+=123;miss[:,retained]-=55
    second=fit()
    assert calls[1]==(upstream,POLICY)
    for d in first:
        assert torch.equal(first[d].weight,second[d].weight)
        assert torch.equal(first[d].bias,second[d].bias)


def test_exact_three_case_scope_and_predecessor_gate():
    spec=json.loads((Path(__file__).resolve().parents[2]/'configs/self_input_evidence.json').read_text())
    cases=make_cases(spec)
    assert len(cases)==len({r['case_id'] for r in cases})==3
    assert make_cases(spec)==cases
    for key,value in [('seeds',[3407]),('fusion_position','feature'),('methods',['missing_only']),('test_access',True)]:
        with pytest.raises(ValueError):make_cases(dict(spec,**{key:value}))
    with pytest.raises(RuntimeError):verify_predecessor(dict(status='running'))


def test_primary_pairing_has_the_correct_baseline_and_negative_sign():
    row=dict(fusion='layer3',filling='normalized_mean',seed=3407,scenario='oct_missing',metric='macro_f1',source_path='x',source_sha256='y')
    rows=[dict(row,method='missing_only',value=.6),dict(row,method=POLICY,value=.5),dict(row,method='original_LOOK',value=.8)]
    pair=self_input_comparisons(rows)
    assert len(pair)==1 and pair[0]['delta']==pytest.approx(-.1)
    assert pair[0]['reference_method']=='missing_only'


from test_method_evidence import graph

@pytest.mark.parametrize('pattern',['oct_missing','cfp_missing'])
def test_actual_graph_writes_only_missing_branch_and_propagates(graph,pattern):
    from look_core.look import fit_complete_pca
    o=torch.randn(2,2,3,224,224);c=torch.randn_like(o)
    pca=fit_complete_pca(graph,[{'oct':o,'cfp':c}],'joint_stem',16,2,torch.device('cpu'),'self-input-writeback')
    a=replace(artifact('joint_stem',2),missing_pattern=pattern,factor=16,
        feature_shape=pca.feature_shape,downsample_shape=pca.downsample_shape,
        mean=pca.mean,std=pca.std,pca_mean=pca.pca_mean,components=pca.components[:2],
        member_names=pca.member_names,member_shapes=pca.member_shapes,
        weight=torch.zeros(2,2),bias=torch.ones(2))
    with torch.no_grad():
        base=k.forward_control(graph,o,c,stop_node='joint_stem').clone()
        after=k.forward_control(graph,o,c,[a],policy=POLICY,pattern=pattern,stop_node='joint_stem').clone()
        kept=slice(64,None) if pattern=='oct_missing' else slice(0,64)
        changed=slice(0,64) if pattern=='oct_missing' else slice(64,None)
        assert torch.equal(after[:,kept],base[:,kept])
        assert not torch.equal(after[:,changed],base[:,changed])
        plain_next=k.forward_control(graph,o,c,stop_node='joint_layer1').clone()
        next_state=k.forward_control(graph,o,c,[a],policy=POLICY,pattern=pattern,stop_node='joint_layer1').clone()
        assert not torch.equal(plain_next,next_state)
        with pytest.raises(ValueError,match='direction'):
            k.forward_control(graph,o,c,[a],policy=POLICY,pattern='cfp_missing' if pattern=='oct_missing' else 'oct_missing')
