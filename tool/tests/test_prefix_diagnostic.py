import numpy as np
import pytest
from look_core.prefix_diagnostic import chosen_candidates, summarize_logits, read_prediction


def test_only_accepted_candidate_not_largest_unused_dimension():
    off=dict(state='off',node='joint_stem')
    on=dict(state='on',node='joint_input',chosen_dimension=8,selected_score=.6,baseline_score=.5,
            candidates=[dict(latent_dim=8,prediction_path='accepted'),dict(latent_dim=512,prediction_path='unused')])
    assert [x['prediction_path'] for x in chosen_candidates([on,off])]==['accepted']


def test_tied_candidate_cannot_be_on():
    d=dict(state='on',node='x',chosen_dimension=8,selected_score=.5,baseline_score=.5,candidates=[dict(latent_dim=8)])
    with pytest.raises(ValueError):chosen_candidates([d])


def test_f1_gain_can_coexist_with_extreme_nll_without_numeric_overflow():
    y=np.array([0,0,1,1])
    before=summarize_logits(y,np.array([[0,1],[0,1],[0,1],[0,1]]))
    after=summarize_logits(y,np.array([[1000,0],[0,1000],[0,1000],[0,1000]]))
    assert after['metrics']['macro_f1']>before['metrics']['macro_f1']
    assert after['metrics']['negative_log_likelihood']==250.
    assert after['max_abs_margin']==1000


def test_reject_nonfinite_logits():
    with pytest.raises(ValueError):summarize_logits([0,1],[[0,float('inf')],[0,1]])


def test_prediction_hash_guards_reuse(tmp_path):
    p=tmp_path/'pred.npz';np.savez(p,labels=[0,1],logits=[[1,0],[0,1]])
    with pytest.raises(ValueError):read_prediction(p,'not_the_hash')
