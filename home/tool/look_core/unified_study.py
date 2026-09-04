"""Uniform Macro-F1 fusion comparison followed by three selected LOOK branches."""
from __future__ import annotations
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import torch
from .pipeline import ExperimentRunner
from .study_grid import StudyGrid, expand_study_grid, run_study_grid, FUSION_POSITIONS, _rows
from .state import PipelineState, atomic_write_json, file_sha256, stable_hash, utc_now
from .reproducibility import implementation_sha256
from .task_selection import validate_selected_task
from .missingness import PROTOCOL as MISSINGNESS_PROTOCOL


def validate_spec(spec):
    if spec['primary_metric'] != 'macro_f1' or spec['classifier_profile']['primary_metric'] != 'macro_f1':
        raise ValueError('Backbone, fusion and LOOK must select by Macro-F1')
    if spec['fusion_positions'] != list(FUSION_POSITIONS) or spec['seeds'] != [3407, 3408, 3409]:
        raise ValueError('All seven fusion positions and the three declared seeds are required')
    if spec['top_k'] != 3 or spec['reference_positions'] != ['oct_only', 'cfp_only']:
        raise ValueError('Three selected fusion positions and both unimodal controls are required')
    if spec.get('test_access') is not False or spec['validation_precision'] != 'fp32':
        raise ValueError('FP32 validation and sealed tests are required')
    if spec['missingness_protocol'] != MISSINGNESS_PROTOCOL or spec['classifier_profile']['missingness_seed'] != spec['missingness_seed']:
        raise ValueError('A shared fixed-direction missingness protocol is required')


def backbone_stages(spec):
    validate_spec(spec)
    return [(f'backbone_{fusion}_{seed}', StudyGrid(fusion_positions=[fusion], seeds=[seed],
        classifier_profiles=[spec['classifier_profile']], filling_strategies=['normalized_mean'],
        look_profiles=[dict(name='unified_backbone', enabled=False, evaluate_missing_baselines=False,
                            evaluate_random_missing=False, primary_metric='macro_f1')]))
        for fusion in spec['reference_positions'] + spec['fusion_positions'] for seed in spec['seeds']]


def rank_fusions(rows, spec):
    """Require every replicate; do not rank architectures by their best seed."""
    validate_spec(spec)
    ranking = []
    for fusion in spec['fusion_positions']:
        group = [row for row in rows if row['fusion_position'] == fusion]
        if sorted(row['seed'] for row in group) != sorted(spec['seeds']):
            raise RuntimeError(f'{fusion}: missing or duplicate seed evidence')
        scores = [float(row['macro_f1']) for row in group]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in scores):
            raise RuntimeError(f'{fusion}: invalid Macro-F1')
        ranking.append(dict(fusion_position=fusion, seeds=spec['seeds'],
            mean_macro_f1=statistics.mean(scores), sd_macro_f1=statistics.stdev(scores),
            per_seed_macro_f1={str(row['seed']): row['macro_f1'] for row in group},
            backbone_ids=[row['backbone_id'] for row in group]))
    ranking.sort(key=lambda row: (-row['mean_macro_f1'], row['sd_macro_f1'], spec['fusion_positions'].index(row['fusion_position'])))
    return ranking


def correction_stages(spec, selected):
    validate_spec(spec)
    if len(selected) != spec['top_k'] or len(set(selected)) != len(selected) or set(selected) - set(spec['fusion_positions']):
        raise ValueError('Exactly three distinct ranked fusion positions are required')
    look = dict(name='joint_macro_f1', enabled=True, evaluate_missing_baselines=True, evaluate_random_missing=True,
        missing_patterns=['oct_missing', 'cfp_missing'], missing_ratios=[.2,.4,.6,.8,1.0],
        correction_nodes=['all_available'], downsample_factors=spec['factors'], latent_dims=spec['latent_dims'],
        max_pca_rank=spec['max_pca_rank'], primary_metric='macro_f1', evaluate_all_factors=True)
    stages = []
    # Complete the same filling for all selected branches before the next filling.
    for filling in ('normalized_mean', 'raw_zero', 'paired_cgan'):
        for fusion in selected:
            for seed in spec['seeds']:
                stages.append((f'{filling}_{fusion}_{seed}', StudyGrid(fusion_positions=[fusion], seeds=[seed],
                    classifier_profiles=[spec['classifier_profile']], filling_strategies=[filling], look_profiles=[copy.deepcopy(look)])))
    for fusion in selected:
        fusion_nodes = [f'fusion_{stage}' for stage in list(FUSION_POSITIONS)[list(FUSION_POSITIONS).index(fusion):]] + ['fusion_participant_feature']
        for name, nodes in [('input_only', ['joint_input']), ('fusion_only', fusion_nodes)]:
            profile = copy.deepcopy(look); profile.update(name=name, correction_nodes=nodes)
            stages.append((f'{name}_{fusion}_3407', StudyGrid(fusion_positions=[fusion], seeds=[3407],
                classifier_profiles=[spec['classifier_profile']], filling_strategies=['normalized_mean'], look_profiles=[profile])))
    return stages


def verify_f1_checkpoint(runner):
    path = runner._train_or_resume()
    completion = json.loads((path.parent/'training_complete.json').read_text())
    history = json.loads((path.parent/'history.json').read_text())
    if completion.get('criteria') != 'validation_macro_f1' or completion.get('primary_metric') != 'macro_f1' or completion.get('validation_precision') != 'fp32':
        raise RuntimeError('Backbone was not selected by FP32 validation Macro-F1')
    best = max(history, key=lambda row: row['criteria_value'])
    if completion['best_epoch'] != best['epoch'] or abs(completion['best_score']-best['validation']['macro_f1']) > 1e-10:
        raise RuntimeError('Saved checkpoint does not match the best validation Macro-F1 epoch')
    return dict(backbone_id=runner._backbone_id(), path=str(path), sha256=file_sha256(path),
        best_epoch=completion['best_epoch'], macro_f1=completion['best_score'], criteria=completion['criteria'], validation_precision='fp32')


def verify_locked_checkpoint(runner, recorded):
    path = Path(recorded['path'])
    if not path.is_file() or file_sha256(path) != recorded['sha256']:
        raise RuntimeError('Previously accepted checkpoint is missing or changed; automatic replacement is forbidden')
    verified = verify_f1_checkpoint(runner)
    if verified['backbone_id'] != recorded['backbone_id'] or verified['sha256'] != recorded['sha256']:
        raise RuntimeError('Previously accepted checkpoint identity changed')
    return verified


def run_unified_study(paths, spec_path, devices, execute):
    validate_selected_task(paths)
    spec = json.loads(Path(spec_path).read_text()); baselines = backbone_stages(spec)
    identity = dict(protocol=spec['protocol'], spec_sha256=file_sha256(spec_path),
        labels_sha256=file_sha256(paths.labels_csv), implementation_sha256=implementation_sha256(paths.project_root),
        gpus=list(devices), primary_metric='macro_f1')
    output = paths.runs_root/'unified_study'/stable_hash(identity)[:12]
    summary = dict(identity, status='planned', output=str(output), completed_stages={}, backbones={}, test_access=False,
        backbone_stages=[name for name,_ in baselines], expected_backbones=27, expected_main_look_cases=27,
        expected_ablation_cases=6, selection_rule='mean_macro_f1, then lower sample SD, then declared position order')
    if not execute:
        return summary
    output.mkdir(parents=True, exist_ok=True)
    with PipelineState(paths.cache_root/'pipeline_state', 'unified_study', identity, [spec_path, paths.labels_csv]) as state:
        if (output/'summary.json').exists():
            old = json.loads((output/'summary.json').read_text())
            if any(old.get(key) != value for key, value in identity.items()):
                raise RuntimeError('Resume identity differs from the committed study')
            summary.update(completed_stages=old.get('completed_stages', {}), backbones=old.get('backbones', {}))
        def report(stage, **updates):
            updates.setdefault('status', 'running')
            summary.update(stage=stage, updated_at_utc=utc_now(), **updates)
            atomic_write_json(summary, output/'summary.json')
            state.checkpoint(stage=stage, summary=str(output/'summary.json'))
            print(f"{utc_now()} stage={stage} status={summary['status']}", flush=True)
        try:
            evidence = []
            for name, grid in baselines:
                case = expand_study_grid(grid, paths, gpu_devices=devices)[0]
                runner = ExperimentRunner(case.config, case.selection, case.options, torch.device('cuda:0'))
                report(name, active_seed=case.selection.seed, active_fusion=case.selection.fusion_position, active_backbone_id=runner._backbone_id())
                key = f'{case.selection.fusion_position}:{case.selection.seed}'
                strict = ExperimentRunner(case.config, case.selection, replace(case.options, strict_backbone_reuse=True), torch.device('cpu'))
                recorded = summary['backbones'].get(key)
                if recorded is not None:
                    verify_locked_checkpoint(strict, recorded)
                result = run_study_grid(grid, paths, torch.device('cuda:0'), execute=True,
                    gpu_devices=devices, strict_backbone_reuse=recorded is not None)
                verified = verify_f1_checkpoint(strict)
                summary['backbones'][key] = verified
                rows = _rows(paths, result)
                if len(rows) != 1 or abs(float(rows[0]['macro_f1']) - verified['macro_f1']) > 1e-10:
                    raise RuntimeError('Frozen FP32 Macro-F1 differs from selected checkpoint validation; investigate before ranking')
                evidence.extend(rows)
                summary['completed_stages'][name] = dict(plan_id=result['plan_id'], status=result['status'])
                atomic_write_json(evidence, output/'backbone_evidence.json')
                report(name+'_complete')
            ranking = rank_fusions(evidence, spec)
            selected = [row['fusion_position'] for row in ranking[:spec['top_k']]]
            atomic_write_json(dict(primary_metric='macro_f1', ranking=ranking, selected=selected,
                selection_rule=summary['selection_rule'], test_access=False), output/'fusion_selection.json')
            stages = correction_stages(spec, selected)
            report('fusion_selection_complete', selected_fusions=selected, look_stages=[name for name,_ in stages])
            expected = {row['backbone_id'] for key,row in summary['backbones'].items() if key.split(':')[0] in selected}
            for name, grid in stages:
                case = expand_study_grid(grid, paths, gpu_devices=devices)[0]
                runner = ExperimentRunner(case.config, case.selection, replace(case.options, strict_backbone_reuse=True), torch.device('cuda:0'))
                if runner._backbone_id() not in expected:
                    raise RuntimeError('LOOK case does not match the newly selected F1 backbone')
                verify_locked_checkpoint(runner, summary['backbones'][f'{case.selection.fusion_position}:{case.selection.seed}'])
                report('prepare_complete_pca', next_stage=name, active_seed=case.selection.seed,
                    active_fusion=case.selection.fusion_position, active_backbone_id=runner._backbone_id())
                sources = runner.prepare_shared_pca()
                summary.setdefault('pca_sources', {})[f'{case.selection.fusion_position}:{case.selection.seed}'] = sources
                torch.cuda.empty_cache()
                plan = run_study_grid(grid, paths, torch.device('cuda:0'), execute=False, gpu_devices=devices, strict_backbone_reuse=True)
                report(name, active_plan_id=plan['plan_id'])
                result = run_study_grid(grid, paths, torch.device('cuda:0'), execute=True, gpu_devices=devices, strict_backbone_reuse=True)
                summary['completed_stages'][name] = dict(plan_id=result['plan_id'], status=result['status'])
                report(name+'_complete')
            report('complete', status='complete')
            state.complete([output/'summary.json', output/'fusion_selection.json', output/'backbone_evidence.json'])
        except BaseException as error:
            report('failed', status='failed', error=repr(error)); raise
    return summary
