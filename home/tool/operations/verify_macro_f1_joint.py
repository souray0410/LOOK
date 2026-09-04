#!/usr/bin/env python3
"""Bounded real-checkpoint validation; train/validation only, separate maintenance artifacts."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import runpy
import torch
from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.data import UKBBilateralVisitDataset, make_loader
from look_core.evaluate import evaluate_missing, save_prediction_bundle
from look_core.filling import NormalizedMeanFiller
from look_core.joint import correction_sites, PROTOCOL
from look_core.look import _reset_inputs, forward_with_look, prepare_complete_pca_bank, greedy_fit_look, load_selected_bank
from look_core.pipeline import ExperimentRunner
from look_core.reproducibility import sha256, seed_everything, implementation_sha256, backbone_implementation_sha256
from look_core.state import atomic_write_json, stable_hash
from look_core.study_grid import expand_study_grid
from look_core.macro_f1_study import study_stages
from look_core.config import ExperimentConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    args=parser.parse_args()
    paths=resolve_runtime_arguments(args)
    project=Path(__file__).resolve().parents[2]
    spec=json.loads((project/'configs/macro_f1_study.json').read_text())
    case=expand_study_grid(study_stages(spec)[0][0][1],paths,gpu_devices=(0,1),smoke_limit=8)[0]
    proof=json.loads((paths.runs_root/'maintenance/macro_f1_training_verification.json').read_text())
    checkpoint=Path(proof['checkpoint']['path'])
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=ExperimentConfig.from_dict(payload['config'])
    runner=ExperimentRunner(config,case.selection,replace(case.options,strict_backbone_reuse=True),torch.device('cuda:0'))
    assert runner._train_or_resume()==checkpoint
    before=sha256(checkpoint)
    graph,_=runner._load_frozen_graph(checkpoint)
    seed_everything(3407)
    identity=dict(protocol=PROTOCOL,implementation_sha256=implementation_sha256(project),checkpoint_sha256=before,train_limit=9,validation_limit=12,Dmax=4,factors=[4,8,16],dimensions=[1,2])
    output=paths.runs_root/'maintenance'/'macro_f1_joint_smoke'/stable_hash(identity)[:12]
    output.mkdir(parents=True,exist_ok=True)
    loaders={}
    for split,limit in [('train',9),('validation',12)]:
        dataset=UKBBilateralVisitDataset(paths.labels_csv,paths.image_root,split,augment=False,limit=limit,preprocess_cache_root=paths.preprocess_cache_root)
        dataset.split=split
        loaders[split]=make_loader(dataset,3,0,train=False,seed=3407)
    batch=next(iter(loaders['validation']))
    device=torch.device('cuda:0')
    oct_x,cfp_x=batch['oct'].to(device),batch['cfp'].to(device)
    filler=NormalizedMeanFiller()
    with torch.no_grad():
        for pattern in ('complete','oct_missing','cfp_missing'):
            x,y=filler.fill(oct_x,cfp_x,pattern)
            _reset_inputs(graph,x,y)
            for level in graph.model_levels:
                graph.forward(levels=[level])
            expected=graph.get_node_by_name('fusion_logits').feature_message.current_state.clone()
            assert torch.equal(expected,forward_with_look(graph,x,y,[])), pattern
    sites=correction_sites(graph)
    bank=prepare_complete_pca_bank(graph,loaders['train'],sites,[4,8,16],4,device,output/'pca',identity,output/'quarantine')
    matrices={}
    for pattern in ('oct_missing','cfp_missing'):
        selected,_=greedy_fit_look(graph,loaders['train'],loaders['validation'],pattern,sites,[4,8,16],[1,2],4,device,output/pattern,bank,primary_metric='macro_f1')
        restored=load_selected_bank(output/pattern)
        result=evaluate_missing(graph,loaders['validation'],device,fixed_pattern=pattern,artifact_banks={pattern:restored})
        baseline=evaluate_missing(graph,loaders['validation'],device,fixed_pattern=pattern)
        assert result['metrics']['macro_f1'] >= baseline['metrics']['macro_f1']
        save_prediction_bundle(result,output/f'{pattern}.npz')
        matrices[pattern]=dict(enabled=[a.node_name for a in selected],metrics=result['metrics'],baseline_metrics=baseline['metrics'])
    assert before==sha256(checkpoint)
    report=dict(identity,status='passed',checkpoint=str(checkpoint),checkpoint_sha256_after=sha256(checkpoint),backbone_implementation_sha256=backbone_implementation_sha256(project),test_access=False,all_off_logits_exact=True,sites=sites,results=matrices,
        purpose='technical verification on bounded train/validation subsets; not a scientific result')
    atomic_write_json(report,output/'verification.json')
    print(json.dumps(dict(status='passed',report=str(output/'verification.json')),indent=2))

if __name__=='__main__':
    main()
