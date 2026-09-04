#!/usr/bin/env python3
"""Two-epoch, two-GPU real-data verification of Macro-F1 best checkpoint saving."""
import argparse,json
from pathlib import Path
from dataclasses import replace
import torch
from look_core.cli import add_runtime_arguments,resolve_runtime_arguments
from look_core.config import ExperimentConfig
from look_core.study_grid import expand_study_grid
from look_core.pipeline import ExperimentRunner
from look_core.unified_study import screening_stages,verify_f1_checkpoint
from look_core.reproducibility import implementation_sha256
from look_core.state import atomic_write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);add_runtime_arguments(p);args=p.parse_args()
    paths=resolve_runtime_arguments(args)
    spec=json.loads((paths.project_root/'configs/unified_study.json').read_text())
    grid=dict(screening_stages(spec))["screen_layer3_3407"]
    case=expand_study_grid(grid,paths,gpu_devices=(0,1),smoke_limit=8)[0]
    config=ExperimentConfig.from_dict(case.config.as_dict())
    config.epochs=2;config.patience=2;config.micro_batch_size=2;config.effective_batch_size=4;config.num_workers=0;config.warmup_epochs=0
    config.output_root=paths.runs_root/'maintenance'/'macro_f1_training_smoke'
    config.cache_root=paths.cache_root/'macro_f1_training_smoke'
    options=replace(case.options,smoke_limit=8)
    runner=ExperimentRunner(config,case.selection,options,torch.device('cuda:0'))
    checkpoint=runner._train_or_resume()
    strict=ExperimentRunner(config,case.selection,replace(options,strict_backbone_reuse=True),torch.device('cpu'))
    verified=verify_f1_checkpoint(strict)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['primary_metric']=='macro_f1' and payload['criteria']=='validation_macro_f1'
    history=json.loads((checkpoint.parent/'history.json').read_text())
    assert len(history)==2 and all(row['criteria']=='validation_macro_f1' for row in history)
    report=dict(status='passed',test_access=False,world_size=2,epochs=2,participants_per_split=8,
        checkpoint=verified,implementation_sha256=implementation_sha256(paths.project_root),
        history=[dict(epoch=x['epoch'],macro_f1=x['validation']['macro_f1'],criterion=x['criteria_value']) for x in history])
    path=paths.runs_root/'maintenance'/'macro_f1_training_verification.json';atomic_write_json(report,path)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
