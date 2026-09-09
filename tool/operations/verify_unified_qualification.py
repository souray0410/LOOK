#!/usr/bin/env python3
"""Intentional train-subset overfit and saved-prediction checks; never a scientific result."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.data import UKBBilateralVisitDataset, make_loader, validate_reference_table
from look_core.graph import build_resnet50_mhd_graph, optimizer_parameter_groups, reset_and_forward
from look_core.evaluate import evaluate_missing, save_prediction_bundle
from look_core.reproducibility import seed_everything, implementation_sha256
from look_core.state import atomic_write_json, durable_replace, file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__);add_runtime_arguments(parser);args=parser.parse_args()
    paths=resolve_runtime_arguments(args);device=torch.device('cuda:0');seed_everything(3407)
    output=paths.runs_root/'maintenance'/'unified_qualification';output.mkdir(parents=True,exist_ok=True)
    audit=validate_reference_table(paths.labels_csv,paths.image_root,check_paths=False)
    loaders={}
    for split,limit in [('train',8),('validation',12)]:
        ds=UKBBilateralVisitDataset(paths.labels_csv,paths.image_root,split,augment=False,limit=limit,preprocess_cache_root=paths.preprocess_cache_root)
        loaders[split]=make_loader(ds,8,0,train=False,seed=3407)
    graph=build_resnet50_mhd_graph('layer3',batch_size=8,device=device,pretrained=True)
    optimizer=torch.optim.AdamW(optimizer_parameter_groups(graph,3e-4,3e-3),weight_decay=1e-4)
    batch=next(iter(loaders['train']));x=batch['oct'].to(device);y=batch['cfp'].to(device);labels=batch['label'].to(device)
    trace=[];passed=False
    for step in range(1,101):
        graph.train();optimizer.zero_grad(set_to_none=True)
        result=reset_and_forward(graph,x,y,labels)
        loss=float(result['loss'].detach());graph.backward(levels=graph.backward_levels);optimizer.step()
        if step % 5==0:
            train=evaluate_missing(graph,loaders['train'],device,fixed_pattern='complete')
            trace.append(dict(step=step,train_loss=loss,train_macro_f1=train['metrics']['macro_f1']))
            print(json.dumps(trace[-1]),flush=True)
            if train['metrics']['macro_f1']==1.0:
                passed=True;break
    if not passed:raise RuntimeError('Intentional small-subset overfit gate not met; diagnose before architecture sweep')
    temporary=output/'overfit.pt.partial';checkpoint=output/'overfit.pt'
    torch.save(graph.state_dict(),temporary);durable_replace(temporary,checkpoint)
    graph.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    validation=evaluate_missing(graph,loaders['validation'],device,fixed_pattern='complete')
    prediction_path=output/'validation.npz';save_prediction_bundle(validation,prediction_path)
    with np.load(prediction_path,allow_pickle=False) as stored:
        f1=f1_score(stored['labels'],stored['logits'].argmax(1),average='macro',zero_division=0)
        auc=roc_auc_score(stored['labels'],stored['scores'])
        assert abs(f1-validation['metrics']['macro_f1'])<1e-12
        assert abs(auc-validation['metrics']['macro_auroc_ovr'])<1e-12
    # A complete bundle round trip must preserve logits exactly.
    with np.load(prediction_path,allow_pickle=False) as stored:assert np.array_equal(stored['logits'],validation['logits'])
    report=dict(status='passed',purpose='technical overfit check, not formal model selection',test_access=False,
        implementation_sha256=implementation_sha256(paths.project_root),data_audit=audit,
        train_count=8,validation_count=12,trace=trace,train_macro_f1=1.0,validation_metrics=validation['metrics'],
        independent_saved_prediction_macro_f1=f1,independent_saved_prediction_auroc=auc,
        checkpoint_sha256=file_sha256(checkpoint),prediction_sha256=file_sha256(prediction_path))
    atomic_write_json(report,output/'verification.json');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
