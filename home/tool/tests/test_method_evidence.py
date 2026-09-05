import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from look_core import method_kernels as k
from look_core.method_study import make_method_cases,verify_predecessors
from look_core.method_report import aggregate,paired_comparisons
from look_core.method_ssf import SSFAdapter,train_ssf
from look_core.graph import build_resnet50_mhd_graph,reset_and_forward
from look_core.look import forward_with_look,fit_complete_pca,fit_look_node
from look_core.filling import NormalizedMeanFiller
from look_core.distributed import module_state_sha256
from test_joint_protocol import artifact


@pytest.fixture(scope='module')
def graph():
    torch.manual_seed(7)
    g=build_resnet50_mhd_graph('layer3',batch_size=1,pretrained=False,device='cpu').eval()
    for p in g.parameters():p.requires_grad_(False)
    return g


def test_exact_bounded_scope_and_fail_closed():
    spec=json.loads((Path(__file__).resolve().parents[2]/'configs/method_evidence.json').read_text())
    cases=make_method_cases(spec)
    assert len(cases)==len({c['case_id'] for c in cases})==9
    assert make_method_cases(spec)==cases
    for key,value in [('fusion_position','feature'),('seeds',[3407]),('test_access',True)]:
        with pytest.raises(ValueError):make_method_cases(dict(spec,**{key:value}))


def test_predecessor_incomplete_fails(monkeypatch):
    import look_core.method_study as study
    monkeypatch.setattr(study,'verify_parent',lambda *a:None)
    with pytest.raises(RuntimeError):verify_predecessors({},dict(status='running',test_access=False),{},Path('x'))


@pytest.mark.parametrize('pattern',['oct_missing','cfp_missing'])
def test_forward_identity_and_joint_writeback(graph,monkeypatch,pattern):
    o=torch.randn(1,2,3,224,224);c=torch.randn_like(o)
    a=replace(artifact(),feature_shape=(6,224,224),downsample_shape=(6,224,224))
    with torch.no_grad():
        before=forward_with_look(graph,o,c).clone()
        for policy in k.POLICIES:assert torch.equal(k.forward_control(graph,o,c,policy=policy,pattern=pattern),before)
        monkeypatch.setattr(k,'apply_artifact',lambda x,a:x+.125)
        corrected=k.forward_control(graph,o,c,[a],policy='missing_only',pattern=pattern,stop_node='joint_stem').clone()
        expected=forward_with_look(graph,o+.125 if pattern=='oct_missing' else o,c+.125 if pattern=='cfp_missing' else c,stop_node='joint_stem')
        assert torch.equal(corrected,expected)
        corrected=k.forward_control(graph,o,c,[a],stop_node='joint_stem').clone()
        assert torch.equal(corrected,forward_with_look(graph,o+.125,c+.125,stop_node='joint_stem'))


def test_ridge_equivalence_and_independent_conditioning(graph,monkeypatch):
    batch={'oct':torch.randn(3,2,3,224,224),'cfp':torch.randn(3,2,3,224,224)}
    node='fusion_participant_feature';device=torch.device('cpu');filler=NormalizedMeanFiller()
    pca=fit_complete_pca(graph,[batch],node,1,2,device,'test-method-pca')
    a=k.fit_candidates(graph,[batch],node,'oct_missing',1,[1,2],2,device,pca,[],filler,'joint')
    b=fit_look_node(graph,[batch],node,'oct_missing',1,[1,2],2,device,pca,[],filler)
    for d in a:
        assert torch.equal(a[d].weight,b[d].weight)
        assert torch.equal(a[d].bias,b[d].bias)
    calls=[];original=k.forward_control
    def wrapped(*args,**kwargs):
        if len(args)>3:calls.append(list(args[3]))
        return original(*args,**kwargs)
    monkeypatch.setattr(k,'forward_control',wrapped)
    c=k.fit_candidates(graph,[batch],node,'oct_missing',1,[1],2,device,pca,[artifact()],filler,'independent_fit')
    assert calls==[[]] and torch.equal(a[1].weight,c[1].weight)


@pytest.mark.parametrize('candidate,enabled',[(.5,False),(.4,False),(.6,True)])
def test_strict_acceptance_ties_and_resume(tmp_path,monkeypatch,candidate,enabled):
    def prediction(*args,**kw):
        bank=args[3]['oct_missing'];value=candidate if bank else .5
        return dict(metrics={'macro_f1':value},labels=np.array([0,1]),probabilities=np.eye(2),logits=np.eye(2),
            scores=np.array([-1,1]),participant_ids=np.array(['a','b']),patterns=np.array(['oct_missing']*2))
    monkeypatch.setattr(k,'evaluate_control',prediction)
    monkeypatch.setattr(k,'fit_candidates',lambda *a,**kw:{2:replace(artifact('a',2),factor=4),1:replace(artifact('a',1),factor=4)})
    config=SimpleNamespace(downsample_factors=[4],correction_nodes=['a'],latent_dims=[1,2],max_pca_rank=2)
    pcas={('a',4):SimpleNamespace(feature_shape=(6,1,1),source_id='pca')}
    args=(None,None,None,'cpu',pcas,config,'oct_missing','independent_fit',tmp_path,'source')
    bank,record=k.fit_control(*args)
    assert bool(bank)==enabled
    d=record['factor_banks'][0]['decisions'][0]
    assert d['chosen_dimension']==(1 if enabled else None)
    monkeypatch.setattr(k,'fit_candidates',lambda *a:pytest.fail('resume refit'))
    bank2,_=k.fit_control(*args);assert len(bank2)==len(bank)
    with pytest.raises(ValueError,match='resume'):k.fit_control(*args[:-3],'missing_only',tmp_path,'source')


def test_ssf_identity_gradient_and_frozen_graph(graph):
    o=torch.randn(1,2,3,224,224);c=torch.randn_like(o)
    with torch.no_grad():base=forward_with_look(graph,o,c).clone()
    frozen=module_state_sha256(graph);adapter=SSFAdapter(graph,'oct_missing')
    try:
        assert all(not s['edge'].startswith('oct_') and 'classifier' not in s['edge'] for s in adapter.slots)
        adapter.enabled=True
        with torch.no_grad():assert torch.equal(forward_with_look(graph,o,c),base)
        optimizer=torch.optim.SGD(adapter.parameters(),lr=.01)
        reset_and_forward(graph,o,c,torch.tensor([1]));graph.backward(levels=graph.backward_levels)
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in adapter.parameters())
        optimizer.step()
        assert module_state_sha256(graph)==frozen
        with torch.no_grad():assert not torch.equal(forward_with_look(graph,o,c),base)
    finally:adapter.close()
    with torch.no_grad():assert torch.equal(forward_with_look(graph,o,c),base)


def test_report_never_aggregates_partial_seeds_and_preserves_negative():
    base=dict(fusion='layer3',filling='normalized_mean',method='ssf',scenario='oct_missing',metric='macro_f1',source_path='p',source_sha256='h')
    rows=[dict(base,seed=s,value=.4) for s in [3407,3408]]
    assert aggregate(rows)==[]
    rows.append(dict(base,seed=3409,value=.7));assert aggregate(rows)[0]['mean']==pytest.approx(.5)
    rows.append(dict(base,method='original_LOOK',seed=3407,value=.6))
    assert paired_comparisons(rows)[0]['delta']==pytest.approx(-.2)


def test_report_verifies_predictions_and_rejects_changed_metrics(tmp_path,monkeypatch):
    import look_core.method_report as report
    from look_core.stable_metrics import logit_metrics
    pred=tmp_path/'predictions';pred.mkdir()
    y=np.array([0,1]);z=np.array([[2.,0.],[0.,2.]])
    np.savez(pred/'validation__fill_oct_missing.npz',labels=y,logits=z)
    result=dict(status='complete',test={},validation={'fill_oct_missing':logit_metrics(y,z)})
    path=tmp_path/'validation_result.json';path.write_text(json.dumps(result))
    parent=tmp_path/'runs'/'unified'/'study'/'summary.json';parent.parent.mkdir(parents=True)
    parent.write_text(json.dumps(dict(completed_stages={'normalized_mean_layer3_3407':{}})))
    suffix=tmp_path/'suffix.json';suffix.write_text(json.dumps(dict(completed_cases={},selected_contexts={})))
    monkeypatch.setattr(report,'source_result',lambda *a:path)
    summary=dict(parent_summary=str(parent),suffix_summary=str(suffix),completed_cases={})
    rows,*_=report.collect(summary)
    assert len(rows)==6 and rows[0]['method']=='filling' and rows[0]['scenario']=='oct_missing'
    result['validation']['fill_oct_missing']['macro_f1']=.4;path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='does not reproduce'):report.collect(summary)


def test_ssf_epoch_resume_is_exact(graph,tmp_path,monkeypatch):
    from torch.utils.data import Dataset,DataLoader
    import look_core.method_ssf as ssf
    class Data(Dataset):
        participant_ids=['a','b','c']
        def __init__(self):
            gen=torch.Generator().manual_seed(44)
            self.o=torch.randn(3,2,3,224,224,generator=gen);self.c=torch.randn(3,2,3,224,224,generator=gen)
        def __len__(self):return 3
        def __getitem__(self,i):return dict(oct=self.o[i],cfp=self.c[i],label=i%2,participant_id=self.participant_ids[i])
    data=Data()
    def loader():return DataLoader(data,batch_size=1,shuffle=True,generator=torch.Generator().manual_seed(9))
    val=DataLoader(data,batch_size=1)
    spec=dict(learning_rates=[1e-5],weight_decay=.01,epochs=2,patience=15,effective_batch_size=2,micro_batch_size=1)
    full=tmp_path/'full';resume=tmp_path/'resume'
    ssf.train_ssf(graph,loader(),val,torch.device('cpu'),'oct_missing',spec,full,'source',3407)
    original=ssf.save_torch
    def interrupt(value,path):
        original(value,path)
        if Path(path).name=='last.pt' and value['epoch']==1:raise KeyboardInterrupt('test interruption')
    monkeypatch.setattr(ssf,'save_torch',interrupt)
    with pytest.raises(KeyboardInterrupt):ssf.train_ssf(graph,loader(),val,torch.device('cpu'),'oct_missing',spec,resume,'source',3407)
    monkeypatch.setattr(ssf,'save_torch',original)
    ssf.train_ssf(graph,loader(),val,torch.device('cpu'),'oct_missing',spec,resume,'source',3407)
    a=torch.load(full/'lr_00/last.pt',weights_only=False);b=torch.load(resume/'lr_00/last.pt',weights_only=False)
    assert a['epoch']==b['epoch']==2
    assert all(torch.equal(a['adapter'][k],b['adapter'][k]) for k in a['adapter'])
    ssf.train_ssf(graph,loader(),val,torch.device('cpu'),'oct_missing',spec,resume,'source',3407)
