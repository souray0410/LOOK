import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from look_core.method_logit import fit_affine,transform_logits,freeze_direction,frozen_random
from look_core.stable_metrics import logit_metrics
from look_core.method_study import make_method_cases
from look_core import method_kernels as kernels
from look_core.look import LatentSufficientStatistics

SPEC=dict(identity_l2=1e-6,minimum_slope=1e-8,max_iterations=2000)


def bundle(y,s,pattern='oct_missing'):
    return dict(labels=np.asarray(y),logits=np.column_stack((-np.asarray(s)/2,np.asarray(s)/2)),
        participant_ids=np.array([str(i) for i in range(len(y))]),patterns=np.array([pattern]*len(y)))


def test_train_affine_ranking_identity_and_finite_extreme_scores():
    y=np.array([0,1,0,1,0,1,0,1]);s=np.array([-2,-1,0,1,2,3,4,5.])*1000
    b=bundle(y,s);p=fit_affine(y,b['logits'],SPEC)
    corrected=transform_logits(b['logits'],p)
    assert p['a']>0 and np.isfinite(corrected).all()
    assert np.array_equal(transform_logits(b['logits'],dict(a=1,c=0)),b['logits'])
    before=logit_metrics(y,b['logits']);after=logit_metrics(y,corrected)
    assert after['macro_auroc_ovr']==before['macro_auroc_ovr']
    assert after['macro_auprc_ovr']==before['macro_auprc_ovr']
    assert after['negative_log_likelihood']<before['negative_log_likelihood']
    assert fit_affine(y,b['logits'],SPEC)==p


def test_validation_labels_do_not_change_fitted_parameters_and_ties_off(monkeypatch):
    import look_core.method_logit as mod
    train=bundle([0,1,0,1],[-2,1,2,3]);val=bundle([0,0,1,1],[-2,-1,1,2])
    a=freeze_direction(train,val,SPEC)
    b=freeze_direction(train,dict(val,labels=1-val['labels']),SPEC)
    assert a['fitted']==b['fitted']
    monkeypatch.setattr(mod,'fit_affine',lambda *args:dict(a=2.,c=0.))
    d=freeze_direction(train,val,SPEC)
    assert not d['enabled'] and d['selected']==dict(a=1.,c=0.)


def test_frozen_random_no_fit_and_complete_rows_unchanged(monkeypatch):
    import look_core.method_logit as mod
    y=[0,1]*5
    bundles={p:bundle(y,np.arange(10)+i,p) for i,p in enumerate(('complete','oct_missing','cfp_missing'))}
    parameters={'oct_missing':dict(a=2,c=1),'cfp_missing':dict(a=.5,c=-2)}
    monkeypatch.setattr(mod,'fit_affine',lambda *a:pytest.fail('random ratios must not fit'))
    a=frozen_random(bundles,parameters,.2,3407);b=frozen_random(bundles,parameters,.4,3407)
    missing=a['patterns']!='complete'
    assert missing.sum()==2 and (b['patterns']!='complete').sum()==4
    assert np.array_equal(a['patterns'][missing],b['patterns'][missing])
    assert np.array_equal(a['logits'][~missing],bundles['complete']['logits'][~missing])
    with pytest.raises(ValueError,match='order'):
        frozen_random(dict(bundles,oct_missing=dict(bundles['oct_missing'],participant_ids=np.arange(10)[::-1])),parameters,.2,3407)


def test_bias_fit_is_train_mean_residual_and_inherits_upstream(monkeypatch):
    # Replace only graph transport with observed latent tensors; regression is real.
    x=torch.tensor([[1.,2.],[3.,8.],[7.,4.]])
    full=x+torch.tensor([[2.,1.],[4.,3.],[6.,5.]])
    pca=SimpleNamespace(node_name='fusion_feature',factor=1,components=torch.eye(2),mean=torch.zeros(2),
        std=torch.ones(2),pca_mean=torch.zeros(2),downsample_shape=(2,),feature_shape=(2,),
        explained_variance_ratio=torch.ones(2)/2,fit_seconds=0.,peak_rss_bytes=0,source_id='p',member_names=(),member_shapes=())
    calls=[];upstream=[object()]
    def forward(graph,o,c,*args,**kw):
        calls.append(list(args[0]) if args else [])
        return x if args else full
    monkeypatch.setattr(kernels,'forward_control',forward)
    filler=SimpleNamespace(name='normalized_mean',fill=lambda o,c,p:(o,c))
    graph=SimpleNamespace(eval=lambda:None)
    out=kernels.fit_candidates(graph,[dict(oct=x,cfp=x)],'fusion_feature','oct_missing',1,[1,2],2,'cpu',pca,upstream,filler,'bias_only')
    assert calls==[[],upstream]
    for d,a in out.items():
        assert torch.count_nonzero(a.weight)==0
        assert torch.equal(a.bias,(full-x).mean(0)[:d])


def test_exact_six_new_cases_and_existing_nine_preserved():
    spec=json.loads((Path(__file__).resolve().parents[2]/'configs/method_evidence.json').read_text())
    cases=make_method_cases(spec)
    old=dict(spec,protocol='frozen_look_method_evidence_v1',methods=['ssf','independent_fit','missing_only'])
    prior=make_method_cases(old)
    assert [c for c in cases if c['method'] not in ('bias_only','logit_affine')]==prior
    assert len({c['case_id'] for c in cases}-{c['case_id'] for c in prior})==6


def test_invalid_fit_fails_closed():
    with pytest.raises(ValueError):fit_affine([0,0],np.zeros((2,2)),SPEC)
    with pytest.raises(ValueError):transform_logits(np.zeros((2,2)),dict(a=-1,c=0))
