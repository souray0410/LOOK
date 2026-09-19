import numpy as np
import pytest

from look.analysis import probability_diagnostic as p
from look.evaluation.stability import logit_metrics, probabilities_from_logits


def bundle(labels,logits,ids=None):
    labels=np.asarray(labels,dtype=np.int64);logits=np.asarray(logits,dtype=float)
    return dict(labels=labels,logits=logits,probabilities=probabilities_from_logits(logits),
        participant_ids=np.asarray(ids if ids is not None else [f"id{i}" for i in range(len(labels))]))


def test_group_contributions_sum_to_total_and_public_small_groups_suppress():
    y=np.array([0,1]*148)
    h=np.tile([[2.,0.],[0.,2.]],(148,1))
    c=h.copy();c[:12]=c[:12,::-1];c[12:40]*=.5
    host=bundle(y,h);corr=bundle(y,c,host["participant_ids"])
    private,public=p.analyze_pair(host,corr,logit_metrics(y,h),logit_metrics(y,c))
    total=private["delta_metrics"]["nll"]
    assert sum(v["contribution_nll"] for v in private["by_class"].values())==pytest.approx(total,abs=1e-12)
    assert sum(v["contribution_nll"] for v in private["by_transition"].values())==pytest.approx(total,abs=1e-12)
    assert sum(v["contribution_nll"] for v in private["by_host_confidence_bin"].values())==pytest.approx(total,abs=1e-12)
    assert public["by_transition"]["host_wrong_to_wrong"]["status"] in ("未观察","分层细节不公开")


def test_fixed_confidence_bin_boundaries_cover_once():
    conf=np.array([.5,.599999999,.6,.6999999,.7,.8,.9,1.0])
    masks=p._bin_masks(conf)
    cover=sum(mask.astype(int) for mask in masks)
    assert np.array_equal(cover,np.ones(len(conf),dtype=int))
    assert masks[0][0] and masks[0][1] and masks[1][2] and masks[-1][-1]


def test_extreme_logits_use_stable_nll_without_clipping():
    y=np.array([0,1])
    z=np.array([[1000.,-1000.],[-1000.,1000.]])
    nll,brier=p._per_person_losses(y,z)
    assert np.all(np.isfinite(nll)) and np.allclose(nll,0)
    assert np.all(np.isfinite(brier)) and np.allclose(brier,0)
    metrics=logit_metrics(y,z)
    assert metrics["negative_log_likelihood"]==pytest.approx(0.0)


def test_pair_rejects_participant_reordering():
    y=np.array([0,1,0,1])
    z=np.array([[2.,0.],[0.,2.],[2.,0.],[0.,2.]])
    host=bundle(y,z,["a","b","c","d"]);corr=bundle(y,z,["b","a","c","d"])
    with pytest.raises(ValueError,match="Participant order mismatch"):
        p.analyze_pair(host,corr,logit_metrics(y,z),logit_metrics(y,z))


def test_metric_mismatch_rejected():
    y=np.array([0,1,0,1])
    z=np.array([[2.,0.],[0.,2.],[2.,0.],[0.,2.]])
    host=bundle(y,z);corr=bundle(y,z,host["participant_ids"])
    wrong=dict(logit_metrics(y,z));wrong["negative_log_likelihood"]+=1e-4
    with pytest.raises(ValueError,match="Accepted metric mismatch"):
        p.analyze_pair(host,corr,wrong,logit_metrics(y,z))


def test_probability_sum_tolerance_is_machine_epsilon_scaled():
    assert p.PROBABILITY_SUM_ATOL == 8*np.finfo(np.float64).eps
    assert p.PROBABILITY_SUM_ATOL > 1.5543122344752192e-15
    assert p.PROBABILITY_SUM_ATOL < 2e-15


def test_prediction_sha_corruption_rejected(tmp_path):
    path=tmp_path/'p.npz'
    y=np.array([0,1]);z=np.array([[2.,0.],[0.,2.]])
    probs=probabilities_from_logits(z)
    np.savez(path,labels=y,probabilities=probs,logits=z,scores=z[:,1]-z[:,0],
        participant_ids=np.array(['a','b']),patterns=np.array(['x','x']))
    with pytest.raises(ValueError,match='Prediction SHA changed'):
        p._finite_binary_bundle(path,'0'*64)


def test_saved_probability_corruption_rejected_even_with_matching_file_sha(tmp_path):
    from look.runtime.state import file_sha256
    path=tmp_path/'p.npz'
    y=np.array([0,1]*148);z=np.tile(np.array([[2.,0.],[0.,2.]]),(148,1))
    probs=probabilities_from_logits(z);probs[0]=[.5,.5]
    np.savez(path,labels=y,probabilities=probs,logits=z,scores=z[:,1]-z[:,0],
        participant_ids=np.array([f'id{i}' for i in range(296)]),patterns=np.array(['x']*296))
    with pytest.raises(ValueError,match='stable softmax'):
        p._finite_binary_bundle(path,file_sha256(path))
