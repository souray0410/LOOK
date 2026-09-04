"""New-release fixed-layer3 study: Macro-F1 checkpointing and Macro-F1 LOOK."""
from __future__ import annotations
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import torch
from .pipeline import ExperimentRunner
from .study_grid import StudyGrid, expand_study_grid, run_study_grid
from .state import PipelineState, atomic_write_json, file_sha256, stable_hash, utc_now
from .reproducibility import implementation_sha256
from .task_selection import validate_selected_task


def study_stages(spec):
    if spec['primary_metric'] != 'macro_f1' or spec['classifier_profile']['primary_metric'] != 'macro_f1':
        raise ValueError('Both backbone and LOOK must select by Macro-F1')
    if spec['fusion_position'] != 'layer3' or spec['seeds'] != [3407,3408,3409] or spec.get('test_access') is not False:
        raise ValueError('Fixed layer3, three seeds and sealed tests required')
    common = dict(fusion_positions=['layer3'], classifier_profiles=[spec['classifier_profile']])
    backbone = []
    for seed in spec['seeds']:
        backbone.append((f'backbone_{seed}', StudyGrid(**common,seeds=[seed],filling_strategies=['normalized_mean'],
            look_profiles=[dict(name='macro_f1_backbone',enabled=False,evaluate_missing_baselines=False,
                                evaluate_random_missing=False,primary_metric='macro_f1')])))
    look = dict(name='joint_macro_f1',enabled=True,evaluate_missing_baselines=True,evaluate_random_missing=True,
        missing_patterns=['oct_missing','cfp_missing'],missing_ratios=[.2,.4,.6,.8,1.0],
        correction_nodes=['all_available'],downsample_factors=spec['factors'],latent_dims=spec['latent_dims'],
        max_pca_rank=spec['max_pca_rank'],primary_metric='macro_f1',evaluate_all_factors=True)
    correction = []
    for name,seeds,filling in [('mean_3407',[3407],'normalized_mean'),('mean_remaining',[3408,3409],'normalized_mean'),
                               ('black_all',spec['seeds'],'raw_zero'),('cgan_all',spec['seeds'],'paired_cgan')]:
        correction.append((name,StudyGrid(**common,seeds=seeds,filling_strategies=[filling],look_profiles=[copy.deepcopy(look)])))
    for name,nodes in [('input_only',['joint_input']),('fusion_only',['fusion_layer3','fusion_layer4','fusion_feature','fusion_participant_feature'])]:
        profile=copy.deepcopy(look);profile.update(name=name,correction_nodes=nodes)
        correction.append((name,StudyGrid(**common,seeds=[3407],filling_strategies=['normalized_mean'],look_profiles=[profile])))
    return backbone,correction


def verify_f1_checkpoint(runner):
    path=runner._train_or_resume()
    completion=json.loads((path.parent/'training_complete.json').read_text())
    history=json.loads((path.parent/'history.json').read_text())
    if completion.get('criteria')!='validation_macro_f1' or completion.get('primary_metric')!='macro_f1':
        raise RuntimeError('Backbone was not selected by Macro-F1')
    best=max(history,key=lambda row:row['criteria_value'])
    if completion['best_epoch']!=best['epoch'] or abs(completion['best_score']-best['validation']['macro_f1'])>1e-6:
        raise RuntimeError('Saved checkpoint does not match the best validation Macro-F1 epoch')
    return dict(backbone_id=runner._backbone_id(),path=str(path),sha256=file_sha256(path),
        best_epoch=completion['best_epoch'],macro_f1=completion['best_score'],criteria=completion['criteria'])


def run_macro_f1_study(paths,spec_path,devices,execute):
    validate_selected_task(paths)
    spec=json.loads(Path(spec_path).read_text())
    backbone,correction=study_stages(spec)
    identity=dict(protocol=spec['protocol'],spec_sha256=file_sha256(spec_path),
        labels_sha256=file_sha256(paths.labels_csv),implementation_sha256=implementation_sha256(paths.project_root),
        gpus=list(devices),primary_metric='macro_f1')
    output=paths.runs_root/'macro_f1_study'/stable_hash(identity)[:12]
    summary=dict(identity,status='planned',output=str(output),completed_stages={},backbones={},test_access=False,
                 backbone_stages=[n for n,_ in backbone],look_stages=[n for n,_ in correction])
    if not execute:return summary
    output.mkdir(parents=True,exist_ok=True)
    with PipelineState(paths.cache_root/'pipeline_state','macro_f1_study',identity,[spec_path,paths.labels_csv]) as state:
        if (output/'summary.json').exists():
            old=json.loads((output/'summary.json').read_text())
            summary.update(completed_stages=old.get('completed_stages',{}),backbones=old.get('backbones',{}))
        def report(stage,**updates):
            updates.setdefault('status','running')
            summary.update(stage=stage,updated_at_utc=utc_now(),**updates)
            atomic_write_json(summary,output/'summary.json');state.checkpoint(stage=stage,summary=str(output/'summary.json'))
            print(f"{utc_now()} stage={stage} status={summary['status']}",flush=True)
        try:
            for name,grid in backbone:
                case=expand_study_grid(grid,paths,gpu_devices=devices)[0]
                runner=ExperimentRunner(case.config,case.selection,case.options,torch.device('cuda:0'))
                report(name,active_seed=case.selection.seed,active_backbone_id=runner._backbone_id())
                result=run_study_grid(grid,paths,torch.device('cuda:0'),execute=True,gpu_devices=devices)
                strict=ExperimentRunner(case.config,case.selection,replace(case.options,strict_backbone_reuse=True),torch.device('cpu'))
                summary['backbones'][str(case.selection.seed)]=verify_f1_checkpoint(strict)
                summary['completed_stages'][name]=dict(plan_id=result['plan_id'],status=result['status'])
                report(name+'_complete')
            expected={r['backbone_id'] for r in summary['backbones'].values()}
            for name,grid in correction:
                for case in expand_study_grid(grid,paths,gpu_devices=devices):
                    runner=ExperimentRunner(case.config,case.selection,replace(case.options,strict_backbone_reuse=True),torch.device('cuda:0'))
                    if runner._backbone_id() not in expected:raise RuntimeError('LOOK case does not match the new F1 backbone')
                    report('prepare_complete_pca',next_stage=name,active_seed=case.selection.seed)
                    sources=runner.prepare_shared_pca()
                    summary.setdefault('pca_sources',{})[str(case.selection.seed)]=sources
                    torch.cuda.empty_cache()
                plan=run_study_grid(grid,paths,torch.device('cuda:0'),execute=False,gpu_devices=devices,strict_backbone_reuse=True)
                report(name,active_plan_id=plan['plan_id'])
                result=run_study_grid(grid,paths,torch.device('cuda:0'),execute=True,gpu_devices=devices,strict_backbone_reuse=True)
                summary['completed_stages'][name]=dict(plan_id=result['plan_id'],status=result['status'])
                report(name+'_complete')
            report('complete',status='complete')
            state.complete([output/'summary.json'])
        except BaseException as error:
            report('failed',status='failed',error=repr(error));raise
    return summary
