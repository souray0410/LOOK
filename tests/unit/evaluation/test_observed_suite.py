import numpy as np
from look.analysis.observed_report import f1_from_confusion,simultaneous_bootstrap
from look.evaluation.observed_suite import assemble_mixed,fit_logit_controls,apply_control
from look.evaluation.stability import logit_metrics


def result(z):
    z=np.asarray(z,dtype=float);n=len(z)
    return dict(logits=z,labels=np.arange(n)%2,participant_ids=np.asarray([str(i) for i in range(n)]),patterns=np.asarray(['complete']*n))


def test_fixed_bias_and_affine_fit_train_target():
    x=result([[1,0],[0,1],[2,0],[0,2]])
    y=dict(x,logits=x['logits']+np.array([.5,-.5]))
    c=fit_logit_controls(y,x)
    for method in ('bias','affine'):
        np.testing.assert_allclose(apply_control(x,c,method)['logits'],y['logits'],atol=1e-12)


def test_masks_are_shared_and_nested_and_only_modify_selected_people():
    full=result(np.ones((10,2)))
    missing={p:dict(full,logits=np.zeros((10,2))) for p in ('oct_missing','cfp_missing')}
    first=assemble_mixed(full,missing,.2,5);second=assemble_mixed(full,missing,.8,5)
    assert np.sum(first['patterns']!='complete')==2
    assert np.all(second['patterns'][first['patterns']!='complete']==first['patterns'][first['patterns']!='complete'])
    np.testing.assert_array_equal(first['logits'][first['patterns']=='complete'],1)


def test_bootstrap_preserves_paired_identity_and_zero_variance():
    y=np.array([0,0,1,1]);p=np.array([y,y,1-y])
    r=simultaneous_bootstrap(y,p,[[1,-1,0],[1,0,-1]],1000,9)
    assert r['contrasts'][0]['difference']==0
    assert r['contrasts'][0]['zero_variance']
    assert r['contrasts'][0]['simultaneous_95'] is None
    assert r['contrasts'][1]['difference']==1
    np.testing.assert_allclose(f1_from_confusion(np.array([2,0,0,2])),1)
