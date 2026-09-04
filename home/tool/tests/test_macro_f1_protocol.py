import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from look_core.train import selection_criterion
from look_core.metrics import validation_macro_f1,validation_binary_auroc
from look_core.pipeline import ExperimentRunner,PipelineOptions
from look_core.macro_f1_study import study_stages,verify_f1_checkpoint
from look_core.study_grid import expand_study_grid,_ranking_key
from test_configuration import _paths,_config


def test_macro_f1_default_and_criteria_dispatch():
    assert selection_criterion('macro_f1') is validation_macro_f1
    assert selection_criterion('macro_auroc_ovr') is validation_binary_auroc
    with pytest.raises(ValueError):selection_criterion('accuracy')


def test_metric_changes_checkpoint_identity(tmp_path):
    from look_core.config import ExperimentSelection,ExperimentConfig
    config=_config(tmp_path)
    assert config.primary_metric=='macro_f1'
    first=ExperimentRunner(config,ExperimentSelection(),PipelineOptions(),torch.device('cpu'))
    other=ExperimentConfig.from_dict(config.as_dict());other.primary_metric='macro_auroc_ovr'
    second=ExperimentRunner(other,ExperimentSelection(),PipelineOptions(),torch.device('cpu'))
    assert first._backbone_id()!=second._backbone_id()


def test_entire_new_study_uses_macro_f1_and_same_backbones(tmp_path):
    spec=json.loads((Path(__file__).resolve().parents[2]/'configs/macro_f1_study.json').read_text())
    baselines,looks=study_stages(spec);paths=_paths(tmp_path)
    ids={}
    for _,grid in baselines:
        case=expand_study_grid(grid,paths,gpu_devices=(0,1))[0]
        runner=ExperimentRunner(case.config,case.selection,case.options,torch.device('cpu'))
        ids[case.selection.seed]=runner._backbone_id()
        assert case.config.primary_metric=='macro_f1' and not case.options.fit_look
    count=0
    for _,grid in looks:
        for case in expand_study_grid(grid,paths,gpu_devices=(0,1)):
            count+=1
            runner=ExperimentRunner(case.config,case.selection,case.options,torch.device('cpu'))
            assert runner._backbone_id()==ids[case.selection.seed]
            assert case.config.primary_metric=='macro_f1' and case.config.evaluate_all_factors
            assert case.options.phase=='validation'
    assert count==11 and len(ids)==3


def test_factor_evaluation_covers_all_random_ratios(monkeypatch,tmp_path):
    import look_core.pipeline as p
    runner=ExperimentRunner.__new__(ExperimentRunner)
    runner.config=SimpleNamespace(missing_patterns=['oct_missing','cfp_missing'],missing_ratios=[.2,.4,.6,.8,1.],evaluate_all_factors=True)
    runner.options=SimpleNamespace(evaluate_random_missing=True,evaluate_missing_baselines=True)
    runner.selection=SimpleNamespace(seed=3407);runner.device='cpu'
    runner.factor_look_banks={f:{'oct_missing':[],'cfp_missing':[]} for f in (4,8,16)}
    runner._save_prediction=lambda *a:None
    monkeypatch.setattr(p,'evaluate_missing',lambda *a,**k: {'metrics':{'macro_f1':.3}})
    result=runner._evaluate_split(None,None,'validation',None,{'oct_missing':[],'cfp_missing':[]})
    for f in (4,8,16):
        assert f'look_x{f}_after_fill_oct_missing' in result
        assert f'look_x{f}_after_fill_cfp_missing' in result
        for ratio in (.2,.4,.6,.8,1.):assert f'look_x{f}_after_fill_random_{ratio:.1f}' in result


def test_checkpoint_verifier_checks_macro_f1_best_epoch(tmp_path):
    checkpoint=tmp_path/'best.pt';checkpoint.write_bytes(b'test')
    history=[dict(epoch=1,criteria_value=.5,validation={'macro_f1':.5}),dict(epoch=2,criteria_value=.7,validation={'macro_f1':.7}),dict(epoch=3,criteria_value=.6,validation={'macro_f1':.6})]
    (tmp_path/'history.json').write_text(json.dumps(history))
    complete=dict(criteria='validation_macro_f1',primary_metric='macro_f1',best_epoch=2,best_score=.7)
    path=tmp_path/'training_complete.json';path.write_text(json.dumps(complete))
    runner=SimpleNamespace(_train_or_resume=lambda:checkpoint,_backbone_id=lambda:'new')
    assert verify_f1_checkpoint(runner)['best_epoch']==2
    complete['best_epoch']=3;path.write_text(json.dumps(complete))
    with pytest.raises(RuntimeError,match='best validation Macro-F1'):verify_f1_checkpoint(runner)


def test_f1_ranking_takes_precedence_over_auc():
    a={'macro_f1':.6,'macro_auroc_ovr':.9}
    b={'macro_f1':.7,'macro_auroc_ovr':.8}
    assert sorted([a,b],key=_ranking_key)[0] is b


def test_look_acceptance_uses_f1_even_when_auc_disagrees(monkeypatch,tmp_path):
    from test_joint_protocol import artifact
    import look_core.look as look
    import look_core.evaluate as evaluate
    import look_core.matrix_analysis as analysis
    def evaluate_bank(*a,artifact_banks,**kw):
        bank=artifact_banks['oct_missing']
        f1,auc=(1/3,.7) if not bank else ((.6,.55) if bank[-1].latent_dim==1 else (.5,.8))
        return dict(metrics={'macro_f1':f1,'macro_auroc_ovr':auc},labels=np.array([0,1]),
            logits=np.eye(2),probabilities=np.eye(2),scores=np.array([-1,1]),participant_ids=np.array(['a','b']),patterns=np.array(['oct_missing']*2))
    monkeypatch.setattr(evaluate,'evaluate_missing',evaluate_bank)
    monkeypatch.setattr(look,'fit_look_node',lambda **kw:{d:replace(artifact('a',d),factor=4) for d in (1,2)})
    monkeypatch.setattr(analysis,'analyze_look_bank',lambda *a:[])
    pca={('a',4):SimpleNamespace(feature_shape=(6,1,1),source_id='new')}
    bank,_,record=look._fit_factor_bank(None,None,None,'oct_missing',['a'],4,[1,2],512,'cpu',tmp_path,pca,primary_metric='macro_f1')
    assert bank[0].latent_dim==1 and record['decisions'][0]['enabled']
    assert record['primary_score']==.6
