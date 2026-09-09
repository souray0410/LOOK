#!/usr/bin/env python3
"""Check the entire post-training evaluation path using an existing, immutable checkpoint."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
from look_core.cli import add_runtime_arguments,resolve_runtime_arguments
from look_core.config import ExperimentConfig,ExperimentSelection
from look_core.pipeline import ExperimentRunner,PipelineOptions
from look_core.graph import reset_and_forward
from look_core.reproducibility import implementation_sha256
from look_core.state import atomic_write_json,file_sha256
from look_core.unified_study import verify_f1_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__);add_runtime_arguments(parser)
    parser.add_argument('--checkpoint',type=Path,required=True);args=parser.parse_args()
    paths=resolve_runtime_arguments(args);checkpoint=args.checkpoint.resolve()
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=ExperimentConfig.from_dict(payload['config'])
    position=config.fusion_positions[0]
    if position not in ('oct_only','cfp_only'):raise ValueError('Expected a unimodal checkpoint')
    selection=ExperimentSelection(fusion_position=position,seed=config.seeds[0],filling_strategy='normalized_mean',run_label='post_training_evaluation_regression')
    options=PipelineOptions(strict_backbone_reuse=True,train_if_missing=False,fit_look=False,
        evaluate_random_missing=False,evaluate_missing_baselines=False,gpu_devices=(0,1))
    original=ExperimentRunner(config,selection,options,torch.device('cpu'))
    verified=verify_f1_checkpoint(original)
    assert Path(verified['path'])==checkpoint
    before=file_sha256(checkpoint)
    output=paths.runs_root/'maintenance'/'unimodal_evaluation_fix'/implementation_sha256(paths.project_root)[:12]
    config.output_root=output/'pipeline';config.cache_root=output/'cache'
    runner=ExperimentRunner(config,selection,options,torch.device('cuda:0'))
    assert runner._backbone_id()==verified['backbone_id']
    def preserved_checkpoint():
        # Evaluation outputs/registry stay in maintenance, with explicit read-only weights.
        if file_sha256(checkpoint)!=before:raise RuntimeError('Checkpoint changed during regression verification')
        return checkpoint
    runner._train_or_resume=preserved_checkpoint
    result=runner.run()
    assert result['status']=='complete' and result['test']=={}
    actual=result['validation']['complete']['macro_f1']
    assert abs(actual-verified['macro_f1'])<1e-10,(actual,verified['macro_f1'])
    graph,_=runner._load_frozen_graph(checkpoint)
    loaders,_=runner._build_loaders()
    expected=[]
    with torch.no_grad():
        for batch in loaders['validation']:
            expected.append(reset_and_forward(graph,batch['oct'].cuda(),batch['cfp'].cuda()).cpu().numpy())
    prediction=runner.prediction_dir/'validation__complete.npz'
    with np.load(prediction,allow_pickle=False) as saved:
        assert np.array_equal(saved['logits'],np.concatenate(expected))
        count=len(saved['labels'])
    assert file_sha256(checkpoint)==before
    report=dict(status='passed',test_access=False,position=position,seed=selection.seed,validation_count=count,
        checkpoint=verified,checkpoint_sha256_after=before,original_forward_logits_exact=True,
        full_pipeline_evaluation_complete=True,training_invocations=0,metrics=result['validation']['complete'],
        implementation_sha256=implementation_sha256(paths.project_root),prediction_sha256=file_sha256(prediction))
    atomic_write_json(report,output/'verification.json')
    print(json.dumps(dict(status='passed',report=str(output/'verification.json'),macro_f1=actual,validation_count=count),indent=2))

if __name__=='__main__':main()
