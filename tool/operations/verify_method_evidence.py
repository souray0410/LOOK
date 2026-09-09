#!/usr/bin/env python3
"""Bounded CPU acceptance using pinned real train/validation resources; never test."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import numpy as np
from torch.utils.data import DataLoader,Subset
from look_core.start_study import read_json,pin_source,source_result,SuffixRunner,verify_source,file_record
from look_core.method_study import make_resource_case
from look_core.method_kernels import fit_control,evaluate_control
from look_core.method_ssf import train_ssf
from look_core.method_analysis import representation_evidence,measure_inference
from look_core.filling import NormalizedMeanFiller
from look_core.state import atomic_write_json,utc_now
from look_core import pipeline

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-summary',type=Path,required=True);p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    parent=read_json(a.parent_summary);runs=a.parent_summary.parents[2]
    source=pin_source(source_result(parent,runs,'normalized_mean_layer3_3407'),parent,runs,a.project_root)
    device=torch.device('cpu');runner=SuffixRunner(source,make_resource_case(3407),a.output/'resources',a.output/'cache',device)
    _,datasets=runner._build_loaders()
    def small(name):
        indices=sorted([int(i) for label in (0,1) for i in np.flatnonzero(datasets[name].labels==label)[:3]])
        if len(indices)!=6:raise ValueError('Acceptance requires three participants from each class')
        subset=Subset(datasets[name],indices)
        subset.participant_ids=[datasets[name].participant_ids[i] for i in indices]
        return DataLoader(subset,batch_size=2,shuffle=False,num_workers=0)
    train,val=small('look_train'),small('validation')
    builder=pipeline.build_resnet50_mhd_graph
    def small_graph(*args,**kw):
        args=list(args);args[2]=2
        return builder(*args,**kw)
    with patch.object(pipeline,'build_resnet50_mhd_graph',small_graph):
        graph,_=runner._load_frozen_graph(runner._train_or_resume())
    pcas=runner._prepare_shared_pca(graph,train,Path(source['checkpoint']['path']))
    config=SimpleNamespace(correction_nodes=['fusion_feature','fusion_participant_feature'],downsample_factors=[8],latent_dims=[8],max_pca_rank=512)
    outputs={};filler=NormalizedMeanFiller()
    for method in ('independent_fit','missing_only'):
        bank,decision=fit_control(graph,train,val,device,pcas,config,'oct_missing',method,a.output/method,source['checkpoint']['sha256'])
        again,_=fit_control(graph,train,val,device,pcas,config,'oct_missing',method,a.output/method,source['checkpoint']['sha256'])
        assert [x.node_name for x in bank]==[x.node_name for x in again]
        evaluated=evaluate_control(graph,val,device,{'oct_missing':bank},policy=method,filler=filler,fixed_pattern='oct_missing')
        assert evaluated['metrics']['macro_f1']==decision['macro_f1']
        random=evaluate_control(graph,val,device,{'oct_missing':bank},policy=method,filler=filler,random_ratio=.4)
        evidence=representation_evidence(graph,val,device,'oct_missing',filler,bank,method,a.output/method/'analysis')
        cost=measure_inference(graph,next(iter(val)),device,'oct_missing',filler,bank,method,repeats=2,warmup=1)
        outputs[method]=dict(decision=decision,diagnostics=evidence,cost=cost,random_metrics=random['metrics'])
    spec=dict(learning_rates=[1e-5],weight_decay=.01,epochs=1,patience=15,effective_batch_size=4,micro_batch_size=2)
    outputs['ssf']=train_ssf(graph,train,val,device,'oct_missing',spec,a.output/'ssf',source['checkpoint']['sha256'],3407)
    verify_source(source)
    atomic_write_json(dict(status='passed',test_access=False,completed_at_utc=utc_now(),scope='acceptance_only; 6 train / 6 validation; CPU; two final nodes; one dimension; SSF one epoch; not formal metrics',
        source=source,outputs=outputs),a.output/'verification.json')
    print(json.dumps(dict(status='passed',verification=str(a.output/'verification.json'))))

if __name__=='__main__':main()
