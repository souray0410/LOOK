import json
import os
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from look.methods import operator as look
from look.methods.joint import correction_sites, read_site, write_site
from look.evaluation.stability import logit_metrics, probabilities_from_logits, participant_cluster_bootstrap
from look.runtime.cleanup import build_cleanup, execute_cleanup, validate_target, TARGET_STUDY


def artifact(name='joint_input', dimension=1):
    return look.LOOKArtifact(name, 'oct_missing', 'normalized_mean', 1, dimension,
        (6, 1, 1), (6, 1, 1), torch.zeros(6), torch.ones(6), torch.zeros(6),
        torch.ones(dimension, 6), torch.zeros(dimension, dimension), torch.ones(dimension), 1., 0., 0.)


def test_joint_input_eye_order_writeback_and_downstream_fit(monkeypatch):
    from look.models.graph import build_resnet50_mhd_graph
    graph = build_resnet50_mhd_graph('layer3', batch_size=2, pretrained=False, device='cpu').eval()
    assert correction_sites(graph) == ['joint_input', 'joint_stem', 'joint_layer1', 'joint_layer2', 'joint_layer3', 'fusion_layer3', 'fusion_layer4', 'fusion_feature', 'fusion_participant_feature']
    oct_x = torch.randn(2, 2, 3, 224, 224)
    cfp_x = torch.randn_like(oct_x)
    with torch.no_grad():
        look._reset_inputs(graph, oct_x, cfp_x)
        for level in graph.model_levels:
            graph.forward(levels=[level])
        expected = graph.get_node_by_name('fusion_logits').feature_message.current_state.clone()
        assert torch.equal(look.forward_with_look(graph, oct_x, cfp_x, []), expected)
        a = replace(artifact(), feature_shape=(6,224,224), downsample_shape=(6,224,224))
        # A visible residual tests propagation without an enormous identity PCA.
        monkeypatch.setattr(look, 'apply_artifact', lambda x, a: x + 0.125)
        after = look.forward_with_look(graph, oct_x, cfp_x, [a], stop_node='joint_stem').clone()
        manual = look.forward_with_look(graph, oct_x + .125, cfp_x + .125, stop_node='joint_stem')
        assert torch.equal(after, manual)
        look._reset_inputs(graph, oct_x, cfp_x)
        joined = read_site(graph, 'joint_input')
        assert torch.equal(joined[:, :3], oct_x.flatten(0,1))
        assert torch.equal(joined[:, 3:], cfp_x.flatten(0,1))
        write_site(graph, 'joint_input', joined+1)
        assert torch.equal(graph.get_node_by_name('cfp_input').feature_message.current_state, cfp_x+1)
        batch = {'oct': oct_x, 'cfp': cfp_x}
        full, missing, _, _ = next(look.iter_feature_pairs(graph, [batch], 'joint_stem', 'oct_missing', 4, torch.device('cpu'), [a]))
        manual_full = look.forward_with_look(graph, oct_x, cfp_x, stop_node='joint_stem')
        manual_missing = look.forward_with_look(graph, torch.zeros_like(oct_x)+.125, cfp_x+.125, stop_node='joint_stem')
        assert torch.equal(full, look.downsample_flatten(manual_full,4)[0])
        assert torch.equal(missing, look.downsample_flatten(manual_missing,4)[0])


def test_channel_stats_interpolation_and_tail(monkeypatch):
    source = torch.arange(19*2*8*8).reshape(19,2,8,8).float() / 100
    state = SimpleNamespace(feature_message=SimpleNamespace(current_state=source))
    graph = SimpleNamespace(get_node_by_name=lambda _: state)
    def features(graph, loader, node, factor, device):
        for batch in source.split(4):
            flat, shape = look.downsample_flatten(batch, factor)
            yield flat, (2,8,8), shape
    monkeypatch.setattr(look,'iter_complete_features',features)
    pca = look.fit_complete_pca(graph, None, 'spatial', 2, 3, torch.device('cpu'), 'test')
    down = F.interpolate(source, size=(4,4), mode='bilinear', align_corners=False, antialias=False)
    expected_mean = down.mean((0,2,3)).repeat_interleave(16)
    expected_std = down.double().var((0,2,3), unbiased=False).clamp_min(1e-12).sqrt().float().repeat_interleave(16)
    assert torch.allclose(pca.mean, expected_mean)
    assert torch.allclose(pca.std, expected_std)
    assert pca.sample_count == 19 and len(pca.components) == 3
    a = replace(artifact(), pca_mean=torch.full((6,), 999.), mean=torch.full((6,), 42.), std=torch.full((6,), 2.))
    # Decoder must not add PCA mean or feature mean to the predicted residual.
    assert torch.equal(look.apply_artifact(torch.zeros(1,6,1,1),a), torch.full((1,6,1,1),2.))


def test_gcv_matches_explicit_centered_hat_matrix():
    torch.manual_seed(18)
    x, y = torch.randn(31,4,dtype=torch.float64), torch.randn(31,4,dtype=torch.float64)
    x -= x.mean(0); y -= y.mean(0)
    cxx, cxy = x.T@x, x.T@y
    actual = look._gcv_lambda(cxx,cxy,float(y.square().sum()),len(x),4)
    def objective(log_lambda):
        hat = x @ torch.linalg.solve(cxx+np.exp(log_lambda)*torch.eye(4),x.T)
        sse = (y-hat@y).square().sum().item()
        return sse/(len(x)*4)/(1-(1+hat.trace().item())/len(x))**2
    expected = np.exp(minimize_scalar(objective,bounds=(-13.8,4.6),method='bounded').x)
    assert actual == pytest.approx(expected, rel=1e-5)


def test_saturated_scores_and_bootstrap_keep_ranking():
    y = np.array([0,0,1,1])
    logits = np.array([[0,80],[0,81],[0,82],[0,83]],dtype=np.float32)
    assert (probabilities_from_logits(logits)[:,1] == 1).all()
    metrics = logit_metrics(y,logits)
    assert metrics['macro_auroc_ovr'] == 1
    assert metrics['macro_auprc_ovr'] == 1
    assert metrics['probability_saturation']['rows_float64'] == 4
    assert metrics['confusion_matrix'] == [[0,2],[0,2]]
    ci = participant_cluster_bootstrap(y,logits,np.array(['a','a','b','b']),metric='macro_auroc_ovr',iterations=20)
    assert ci['estimate'] == 1


@pytest.mark.parametrize('scores, expected', [({'a':.7,'b':.7},['a']), ({'a':.5,'b':.4},[])])
def test_optional_decisions_atomic_resume_and_all_off(monkeypatch,tmp_path,scores,expected):
    import look.evaluation.evaluator as ev
    import look.analysis.matrices as ma
    calls=[]
    def evaluate(*args,artifact_banks,**kwargs):
        bank=artifact_banks['oct_missing']
        value=scores[bank[-1].node_name] if bank else .5
        return dict(metrics={'macro_auroc_ovr':value}, labels=np.array([0,1]), probabilities=np.eye(2), logits=np.eye(2), scores=np.array([-1,1]), participant_ids=np.array(['a','b']), patterns=np.array(['x','x']))
    def fit(**kw):
        calls.append((kw['node_name'],[a.node_name for a in kw['upstream_artifacts']]))
        return {2: replace(artifact(kw['node_name'],2), factor=4), 1:replace(artifact(kw['node_name']),factor=4)}
    monkeypatch.setattr(ev,'evaluate_missing',evaluate)
    monkeypatch.setattr(look,'fit_look_node',fit)
    monkeypatch.setattr(ma,'analyze_look_bank',lambda *a: [])
    pca={ (n,4):SimpleNamespace(feature_shape=(6,1,1),source_id=n) for n in ['a','b'] }
    args=(None,None,None,'oct_missing',['a','b'],4,[1,2],512,torch.device('cpu'),tmp_path)
    bank,history,complete=look._fit_factor_bank(*args,pca_bank=pca,primary_metric='macro_auroc_ovr')
    assert [a.node_name for a in bank] == expected
    decisions=complete['decisions']
    assert [d['best_dimension'] for d in decisions] == [1,1]
    assert calls[1][1] == expected
    assert len(look.load_selected_bank(tmp_path)) == len(expected)
    # Simulate interruption after all decision commits but before bank completion.
    (tmp_path/'bank_complete.json').unlink()
    monkeypatch.setattr(look,'fit_look_node',lambda **kw: pytest.fail('completed decisions were refit'))
    resumed,_,_=look._fit_factor_bank(*args,pca_bank=pca,primary_metric='macro_auroc_ovr')
    assert [a.node_name for a in resumed] == expected
    assert json.loads((tmp_path/'decisions/02_b.json').read_text()) == decisions[1]


def make_old_study(tmp_path):
    runs,cache=tmp_path/'runs',tmp_path/'cache'
    def put(relative,value):
        p=runs/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    put(f'reviewed_look/{TARGET_STUDY}/summary.json',dict(status='failed',active_plan_id='0123456789ab'))
    put('sweeps/validation__0123456789ab/study_plan.json',dict(phase='validation',grid={'look_profiles':[{'enabled':True}]},experiment_ids=['old']))
    put('experiments/old/validation_result.json',{'validation':{'negative':{'macro_auroc_ovr':.4}}})
    put('backbones/keep.json',{'keep':True})
    return runs,cache


def test_cleanup_scoped_audit_retains_negative_result(tmp_path):
    runs,cache=make_old_study(tmp_path)
    report=build_cleanup(runs,cache)
    path=execute_cleanup(report,runs,cache)
    assert (runs/'backbones/keep.json').exists()
    assert not (runs/'experiments/old').exists()
    audit=json.loads(path.read_text())
    assert audit['status']=='complete' and '.4' in json.dumps(audit['prior_results'])


@pytest.mark.parametrize('kind',['symlink','live_lock','other_study','test','baseline','external_reference'])
def test_cleanup_rejects_unsafe_scope(tmp_path,kind):
    runs,cache=make_old_study(tmp_path)
    if kind=='symlink':
        (runs/'experiments/old/escape').symlink_to(tmp_path)
    elif kind=='live_lock':
        (runs/'experiments/old/stage.lock').write_text(json.dumps({'pid':os.getpid()}))
    elif kind=='test':
        (runs/'experiments/old/test_result.json').write_text('{}')
    elif kind=='external_reference':
        p=runs/'sweeps/validation__eeeeeeeeeeee/study_plan.json';p.parent.mkdir();p.write_text(json.dumps({'experiment_ids':['old']}))
    with pytest.raises((ValueError,RuntimeError)):
        if kind=='other_study': build_cleanup(runs,cache,'other')
        elif kind=='baseline': validate_target(runs/'backbones/keep.json',(runs,))
        else: build_cleanup(runs,cache)


def test_factor_ties_prefer_fewer_sites_then_larger_factor(monkeypatch,tmp_path):
    def fit(*args,**kwargs):
        factor=args[5]
        bank=[replace(artifact('a'),factor=factor)] if factor==8 else []
        return bank,[],dict(primary_score=.7,factor=factor)
    monkeypatch.setattr(look,'_fit_factor_bank',fit)
    result,_=look.greedy_fit_look(None,None,None,'oct_missing',['a'],[4,8,16],[1],512,torch.device('cpu'),tmp_path,{})
    assert result==[]
    assert json.loads((tmp_path/'factor_selection.json').read_text())['selected_factor']==16
    assert look.load_selected_bank(tmp_path)==[]


def test_strict_checkpoint_failure_preserves_original(tmp_path):
    from look.studies.pipeline import ExperimentRunner
    runner=ExperimentRunner.__new__(ExperimentRunner)
    runner.config=SimpleNamespace(output_root=tmp_path,cache_root=tmp_path/'cache')
    runner.options=SimpleNamespace(restart=False,resume=True,train_if_missing=False)
    runner._backbone_id=lambda:'fixed'
    runner.data_hash='expected'
    path=tmp_path/'backbones/fixed';path.mkdir(parents=True)
    (path/'best.pt').write_bytes(b'original')
    (path/'training_complete.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Strict checkpoint'):
        runner._train_or_resume()
    assert (path/'best.pt').read_bytes()==b'original'
    assert (path/'training_complete.json').read_text()=='{}'
    assert not (tmp_path/'cache').exists()


def test_cleanup_rejects_externally_shared_pca(tmp_path):
    runs,cache=make_old_study(tmp_path)
    source=runs/f'reviewed_look/{TARGET_STUDY}/summary.json'
    summary=json.loads(source.read_text());summary['shared_pca_sources']={'3407':['a'*16+'/feature_x1']};source.write_text(json.dumps(summary))
    bank=runs/'pca'/('a'*16);bank.mkdir(parents=True)
    (bank/'bank_manifest.json').write_text(json.dumps({'bank_id':'a'*16,'identity':{}}))
    other=runs/'experiments/other';other.mkdir()
    (other/'manifest.json').write_text(json.dumps({'pca_source_id':'a'*16+'/feature_x1'}))
    with pytest.raises(ValueError,match='externally referenced'):
        build_cleanup(runs,cache)
    assert bank.exists()


def test_backbone_reuse_does_not_disable_independent_generator(tmp_path,monkeypatch):
    from look.studies.pipeline import ExperimentRunner, PipelineOptions
    import look.studies.pipeline as pipeline
    options=PipelineOptions(train_if_missing=True,strict_backbone_reuse=True)
    runner=ExperimentRunner.__new__(ExperimentRunner)
    runner.options=options
    runner.config=SimpleNamespace(output_root=tmp_path,cache_root=tmp_path/'cache')
    runner._backbone_id=lambda:'missing_backbone'
    with pytest.raises(FileNotFoundError):
        runner._train_or_resume()
    # Reach the generator launcher while strictly forbidding backbone training.
    runner.selection=SimpleNamespace(filling_strategy='paired_cgan',seed=3407)
    runner._gan_id=lambda:'independent'
    runner.project_root=tmp_path
    runner.config.as_dict=lambda:{}
    class ReachedGenerator(Exception): pass
    def launch(*args):
        assert args[2]=='gan'
        raise ReachedGenerator
    monkeypatch.setattr(pipeline,'make_gan_loaders',lambda *a,**k:(None,None,{}))
    monkeypatch.setattr(pipeline,'launch_ddp_stage',launch)
    with pytest.raises(ReachedGenerator):
        runner._prepare_filler({'train':None,'look_train':None})
    runner.options=replace(options,phase='test',train_if_missing=False)
    with pytest.raises(FileNotFoundError,match='Frozen test generator'):
        runner._prepare_filler({'train':None,'look_train':None})
