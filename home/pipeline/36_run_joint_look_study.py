#!/usr/bin/env python3
"""Run a reviewed fixed-backbone LOOK development study; never open test."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path


def latent_candidates(explicit, start, stop, step, max_rank):
    if explicit is not None and any(v is not None for v in (start, stop, step)):
        raise ValueError('Choose either a latent list or a range, not both')
    if any(v is not None for v in (start, stop, step)):
        if any(v is None for v in (start, stop, step)) or step < 1 or start < 1 or stop < start:
            raise ValueError('A range requires positive min, max and step')
        values = list(range(start, stop + 1, step))
    else:
        values = explicit if explicit is not None else [8, 16, 32, 64, 96, 128, 192, 256, 384, 512]
    values = sorted(set(values))
    if max_rank < 1 or not values or values[0] < 1 or values[-1] > max_rank:
        raise ValueError('Latent candidates must be positive and not exceed PCA Dmax')
    return values


def study_stages(candidate, factors, latent_dims, pca_max_rank=512, quick_dims=None):
    from look_core.study_grid import StudyGrid
    common = dict(
        fusion_positions=[candidate['winner']['fusion_position']],
        classifier_profiles=[candidate['classifier_profile']],
        look_profiles=[dict(
            name='joint_sequential_optional_v1', enabled=True, evaluate_random_missing=True,
            evaluate_missing_baselines=True,
            missing_patterns=['oct_missing', 'cfp_missing'],
            missing_ratios=[0.2, 0.4, 0.6, 0.8, 1.0],
            correction_nodes=['all_available'], downsample_factors=factors,
            latent_dims=latent_dims, max_pca_rank=pca_max_rank,
            primary_metric='macro_auroc_ovr',
        )],
    )
    stages = [
        ('main_mean_3407', StudyGrid(**common, seeds=[3407], filling_strategies=['normalized_mean'])),
        ('main_mean_remaining', StudyGrid(**common, seeds=[3408, 3409], filling_strategies=['normalized_mean'])),
        ('main_black_all', StudyGrid(**common, seeds=[3407, 3408, 3409], filling_strategies=['raw_zero'])),
        ('main_cgan_all', StudyGrid(**common, seeds=[3407, 3408, 3409], filling_strategies=['paired_cgan'])),
    ]
    for name, nodes in (
        ('input_only', ['joint_input']),
        ('fusion_only', ['fusion_layer3', 'fusion_layer4', 'fusion_feature', 'fusion_participant_feature']),
    ):
        ablation = copy.deepcopy(common)
        ablation['look_profiles'][0].update(name=name, correction_nodes=nodes)
        stages.append((name, StudyGrid(**ablation, seeds=[3407], filling_strategies=['normalized_mean'])))
    return stages


def main():
    from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
    from look_core.state import PipelineState, atomic_write_json, file_sha256, stable_hash, utc_now
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reviewer-note', required=True)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--factors', nargs='+', type=int, default=[4, 8, 16])
    parser.add_argument('--pca-max-rank', type=int, default=512)
    parser.add_argument('--latent-dims', nargs='+', type=int)
    parser.add_argument('--latent-min', type=int)
    parser.add_argument('--latent-max', type=int)
    parser.add_argument('--latent-step', type=int)
    parser.add_argument('--quick-latent-dims', nargs='+', type=int)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.reviewer_note.strip():
        parser.error('A genuine review decision is required')
    if not args.factors or min(args.factors) < 1:
        parser.error('Factors must be positive')
    try:
        dims = latent_candidates(args.latent_dims, args.latent_min, args.latent_max,
                                 args.latent_step, args.pca_max_rank)
        quick_dims = (latent_candidates(args.quick_latent_dims, None, None, None, args.pca_max_rank)
                      if args.quick_latent_dims else None)
    except ValueError as error:
        parser.error(str(error))
    paths = resolve_runtime_arguments(args)
    from look_core.distributed import parse_gpu_devices
    devices = parse_gpu_devices(args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
    import torch
    from look_core.artifacts import validate_file_manifest
    from look_core.pipeline import ExperimentRunner
    from look_core.reproducibility import implementation_sha256
    from look_core.study_grid import expand_study_grid, run_study_grid
    from look_core.task_selection import validate_selected_task
    validate_selected_task(paths)
    candidate = json.loads(args.candidate.read_text())
    if candidate['winner']['fusion_position'] != 'layer3':
        raise ValueError('Step 36 strictly reuses the selected layer3 backbone')
    if sorted(candidate['winner']['seeds']) != [3407, 3408, 3409]:
        raise ValueError('The reviewed configuration must have all three seed checkpoints')
    if not candidate.get('artifacts'):
        raise ValueError('Missing baseline evidence manifest')
    errors = validate_file_manifest(candidate['artifacts'])
    if errors:
        raise RuntimeError(f'Baseline evidence failed verification: {errors[:5]}')
    stages = study_stages(candidate, args.factors, dims, args.pca_max_rank, quick_dims)
    identity = dict(candidate_sha256=file_sha256(args.candidate),
        labels_sha256=file_sha256(paths.labels_csv),
        implementation_sha256=implementation_sha256(paths.project_root),
        script_sha256=file_sha256(Path(__file__)), gpus=args.gpus,
        reviewer_note=args.reviewer_note, stages=[(name, asdict(grid)) for name, grid in stages])
    output = paths.runs_root / 'joint_look' / stable_hash(identity)[:12]
    summary = dict(identity, status='planned', output=str(output),
        approval_scope='fixed_backbone_validation_development_only', test_access=False,
        automatic_quality_gate_passed=bool(candidate.get('quality_gate_passed')),
        automatic_quality_gate_checks=candidate.get('quality_gate_checks', {}),
        selection_rule='reviewed three-seed mean validation AUROC; no best-seed selection',
        winner=candidate['winner'], completed_stages={})
    # Check exact scientific IDs before starting any expensive LOOK or GAN work.
    expected = set(candidate['winner']['backbone_ids'])
    for name, grid in stages:
        cases = expand_study_grid(grid, paths, gpu_devices=devices)
        for case in cases:
            probe = ExperimentRunner(case.config, case.selection,
                replace(case.options, train_if_missing=False), torch.device('cpu'))
            if probe._backbone_id() not in expected:
                raise RuntimeError('Study config does not match the reviewed backbone')
            probe._train_or_resume()
    if not args.execute:
        print(json.dumps(summary, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this full study')
    output.mkdir(parents=True, exist_ok=True)
    with PipelineState(paths.cache_root / 'pipeline_state', 'joint_look', identity,
                       [args.candidate, paths.labels_csv]) as state:
        previous = output / 'summary.json'
        if previous.exists():
            summary['completed_stages'] = json.loads(previous.read_text()).get('completed_stages', {})

        def report(stage, **values):
            summary.update(stage=stage, updated_at_utc=utc_now(), **values)
            atomic_write_json(summary, output / 'summary.json')
            state.checkpoint(stage=stage, summary=str(output / 'summary.json'))
            print(f"{utc_now()} stage={stage} status={summary['status']}", flush=True)

        approval = dict(candidate=candidate, reviewer_note=args.reviewer_note,
            scope=summary['approval_scope'], original_gate_passed=summary['automatic_quality_gate_passed'],
            decision='proceed_without_further_backbone_search', test_access=False)
        atomic_write_json(approval, output / 'reviewed_baseline.json')
        try:
            for name, grid in stages:
                report('prepare_shared_complete_pca', status='running', next_stage=name)
                prepared = set()
                for case in expand_study_grid(grid, paths, gpu_devices=devices):
                    runner = ExperimentRunner(case.config, case.selection,
                        replace(case.options, train_if_missing=False), torch.device('cuda:0'))
                    if runner._backbone_id() in prepared:
                        continue
                    report('prepare_shared_complete_pca', active_seed=case.selection.seed)
                    sources = runner.prepare_shared_pca()
                    prepared.add(runner._backbone_id())
                    summary.setdefault('shared_pca_sources', {})[str(case.selection.seed)] = sources
                    report('prepare_shared_complete_pca')
                    torch.cuda.empty_cache()
                report(name, status='running', active_grid=asdict(grid))
                # This writes a resumable plan even if interrupted before the first result.
                plan = run_study_grid(grid, paths, torch.device('cuda:0'), execute=False,
                                      phase='validation', gpu_devices=devices, strict_backbone_reuse=True)
                report(name, active_plan_id=plan['plan_id'])
                result = run_study_grid(grid, paths, torch.device('cuda:0'), execute=True,
                                        phase='validation', gpu_devices=devices, strict_backbone_reuse=True)
                summary['completed_stages'][name] = dict(plan_id=result['plan_id'], status=result['status'])
                comparisons = []
                for completed in summary['completed_stages'].values():
                    progress_path = paths.runs_root / 'sweeps' / f"validation__{completed['plan_id']}" / 'progress.json'
                    progress = json.loads(progress_path.read_text())
                    for item in progress.get('completed', []):
                        record = json.loads(Path(item['result_file']).read_text())
                        comparisons.append(dict(experiment_id=item['experiment_id'],
                            validation=record.get('validation', {}),
                            result_file=item['result_file']))
                atomic_write_json(comparisons, output / 'look_comparisons.json')
                report(name + '_complete')
            report('complete', status='complete', reason='9 full filling/seed cases and 2 ablations completed; test remains sealed')
            state.complete([output / 'summary.json', output / 'reviewed_baseline.json'])
        except BaseException as error:
            report('failed', status='failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
