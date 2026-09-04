import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from look_core.missingness import missingness_plan
from look_core.unified_study import rank_fusions, correction_stages
from look_core.train import evaluate_fp32


def spec():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/unified_study.json').read_text())


def test_missingness_nested_fixed_direction_exact_counts_and_order_independence():
    ids=[str(i) for i in range(296)]
    previous={}
    for ratio, count in [(0,0),(.2,59),(.4,118),(.6,178),(.8,237),(1,296)]:
        plan,meta=missingness_plan(ids,ratio)
        assert plan==missingness_plan(list(reversed(ids)),ratio)[0]
        missing={pid:p for pid,p in plan.items() if p!='complete'}
        assert len(missing)==count and all(missing[pid]==p for pid,p in previous.items())
        assert abs(meta['counts']['oct_missing']-meta['counts']['cfp_missing'])<=1
        assert set(plan.values()) <= {'complete','oct_missing','cfp_missing'}
        previous=missing
    with pytest.raises(ValueError):missingness_plan(['a','a'],.2)
    with pytest.raises(ValueError):missingness_plan(ids,float('nan'))


def evidence():
    return [dict(fusion_position=fusion,seed=seed,macro_f1=.5,backbone_id=f'{fusion}:{seed}')
            for fusion in spec()['fusion_positions'] for seed in spec()['seeds']]


def test_fusion_ranking_uses_all_seeds_and_declared_ties():
    rows=evidence()
    # A high best seed must not win against a better three-seed mean.
    for row in rows:
        if row['fusion_position']=='input':row['macro_f1']=1. if row['seed']==3407 else .25
        if row['fusion_position']=='layer3':row['macro_f1']=.6
    ranking=rank_fusions(rows,spec())
    assert ranking[0]['fusion_position']=='layer3'
    assert ranking[1]['fusion_position']=='stem' # same mean, lower SD than input
    assert ranking[-1]['fusion_position']=='input'
    with pytest.raises(RuntimeError):rank_fusions(rows[:-1],spec())
    with pytest.raises(RuntimeError):rank_fusions(rows+[rows[0]],spec())


def test_fusion_only_sites_follow_selected_architecture():
    stages=dict(correction_stages(spec(),['input','layer3','feature']))
    assert stages['fusion_only_feature_3407'].look_profiles[0]['correction_nodes']==['fusion_feature','fusion_participant_feature']
    assert stages['fusion_only_input_3407'].look_profiles[0]['correction_nodes'][0]=='fusion_input'
    assert len(stages)==33


def test_validation_uses_fp32_and_restores_training_precision_on_error():
    trainer=SimpleNamespace(precision='fp16')
    def evaluate(loader,epoch):
        assert trainer.precision=='fp32'
        return .7
    trainer.eval_epoch=evaluate
    assert evaluate_fp32(trainer,None,1)==.7 and trainer.precision=='fp16'
    def fail(*args):raise RuntimeError('interrupted')
    trainer.eval_epoch=fail
    with pytest.raises(RuntimeError):evaluate_fp32(trainer,None,1)
    assert trainer.precision=='fp16'


def test_prediction_bundle_preserves_shared_mask_provenance(tmp_path):
    import numpy as np
    from look_core.pipeline import ExperimentRunner
    plan,meta=missingness_plan(['a','b'],1.)
    result=dict(labels=np.array([0,1]),logits=np.array([[1.,0.],[0.,1.]]),probabilities=np.array([[.7,.3],[.3,.7]]),
                scores=np.array([-1.,1.]),participant_ids=np.array(['a','b']),patterns=np.array([plan['a'],plan['b']]),missingness=meta)
    runner=ExperimentRunner.__new__(ExperimentRunner);runner.prediction_dir=tmp_path
    runner._save_prediction(result,'validation','random_1.0')
    assert json.loads((tmp_path/'validation__random_1.0.missingness.json').read_text())==meta
    with np.load(tmp_path/'validation__random_1.0.npz') as stored:
        assert list(stored['patterns'])==[plan['a'],plan['b']]


def test_missing_or_modified_accepted_checkpoint_never_requests_retraining(tmp_path):
    from look_core.unified_study import verify_locked_checkpoint
    from look_core.state import file_sha256
    checkpoint=tmp_path/'best.pt';checkpoint.write_bytes(b'accepted')
    record=dict(path=str(checkpoint),sha256=file_sha256(checkpoint),backbone_id='accepted')
    def forbidden():raise AssertionError('must reject before any training or reuse call')
    runner=SimpleNamespace(_train_or_resume=forbidden)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='automatic replacement'):verify_locked_checkpoint(runner,record)
    checkpoint.unlink()
    with pytest.raises(RuntimeError,match='automatic replacement'):verify_locked_checkpoint(runner,record)
