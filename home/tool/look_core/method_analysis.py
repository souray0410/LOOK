"""Paired representation/decision evidence and measured inference overhead."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch

from .method_kernels import forward_control
from .state import atomic_write_json
from .start_report import write_csv


@torch.no_grad()
def representation_evidence(graph, loader, device, pattern, filler, artifacts, policy,
                            output, adapter=None):
    graph.eval()
    head = graph.get_edge_by_name('fusion_classifier_edge').edge_operations[0].function
    if not hasattr(head, 'linear') or head.linear.out_features != 2:
        raise ValueError('Margin bound requires the frozen binary linear prediction head')
    wnorm = float(torch.linalg.vector_norm(head.linear.weight[1]-head.linear.weight[0]))
    rows = []
    for batch in loader:
        o,c = batch['oct'].to(device),batch['cfp'].to(device)
        if adapter is not None: adapter.enabled=False
        full = forward_control(graph,o,c,stop_node='fusion_participant_feature').detach().clone()
        fo,fc = filler.fill(o,c,pattern)
        missing = forward_control(graph,fo,fc,stop_node='fusion_participant_feature').detach().clone()
        if adapter is not None: adapter.enabled=True
        corrected = forward_control(graph,fo,fc,artifacts,policy=policy,pattern=pattern,
                                    stop_node='fusion_participant_feature').detach().clone()
        z_full,z_before,z_after = head(full),head(missing),head(corrected)
        before_error = torch.linalg.vector_norm(missing-full,dim=1)
        after_error = torch.linalg.vector_norm(corrected-full,dim=1)
        margin_full = z_full[:,1]-z_full[:,0]
        margin_after = z_after[:,1]-z_after[:,0]
        bound = wnorm*after_error
        observed = torch.abs(margin_after-margin_full)
        # Allow only numerical floating-point error in this exact linear-head identity.
        if bool((observed > bound+1e-4*(1+bound)).any()): raise RuntimeError('Margin bound violated')
        truth = batch['label'].to(device)
        for i,pid in enumerate(batch['participant_id']):
            rows.append(dict(participant_id=str(pid),pattern=pattern,label=int(truth[i]),
                complete_prediction=int(z_full[i].argmax()),before_prediction=int(z_before[i].argmax()),
                after_prediction=int(z_after[i].argmax()),before_l2=float(before_error[i]),
                after_l2=float(after_error[i]),complete_norm=float(full[i].norm()),
                margin_complete=float(margin_full[i]),margin_before=float(z_before[i,1]-z_before[i,0]),
                margin_after=float(margin_after[i]),margin_change=float(observed[i]),margin_bound=float(bound[i]),
                complete_decision_guaranteed_preserved=bool(bound[i] < abs(margin_full[i]))))
    if adapter is not None: adapter.enabled=False
    write_csv(rows,Path(output)/'participant_representation.csv')
    wrong_to_right = sum(r['before_prediction']!=r['label'] and r['after_prediction']==r['label'] for r in rows)
    right_to_wrong = sum(r['before_prediction']==r['label'] and r['after_prediction']!=r['label'] for r in rows)
    closer_harmed = sum(r['after_l2']<r['before_l2'] and r['before_prediction']==r['label'] and r['after_prediction']!=r['label'] for r in rows)
    result=dict(n_participants=len(rows),mean_l2_before=float(np.mean([r['before_l2'] for r in rows])),
        mean_l2_after=float(np.mean([r['after_l2'] for r in rows])),wrong_to_right=wrong_to_right,
        right_to_wrong=right_to_wrong,closer_representation_but_harmed=closer_harmed,
        bound_scope='final participant feature; preserves complete-model decision, not correctness or F1',
        interpretation='paired validation diagnostics, not causal node contributions')
    atomic_write_json(result,Path(output)/'representation_summary.json')
    return result


@torch.no_grad()
def measure_inference(graph, batch, device, pattern, filler, artifacts, policy, adapter=None,
                      repeats=10, warmup=3):
    """Same resident graph and batch. CPU loading/metric calculation excluded."""
    o,c = batch['oct'].to(device),batch['cfp'].to(device)
    if adapter is not None: adapter.enabled = pattern != 'complete'
    def run():
        po,pc = filler.fill(o,c,pattern)
        return forward_control(graph,po,pc,artifacts,policy=policy,pattern=pattern)
    for _ in range(warmup): run()
    if device.type=='cuda':
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.memory_allocated(device) if device.type=='cuda' else None
    samples=[]
    for _ in range(repeats):
        if device.type=='cuda': torch.cuda.synchronize(device)
        start=time.perf_counter();run()
        if device.type=='cuda': torch.cuda.synchronize(device)
        samples.append(time.perf_counter()-start)
    result=dict(device=str(device),participants_per_batch=len(o),eyes_per_participant=2,
        warmup=warmup,repeats=repeats,seconds_per_batch=samples,
        median_ms_per_participant=1000*float(np.median(samples))/len(o),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        resident_allocated_before_bytes=base_alloc,
        scope='warm model-only inference including filling and correction; excludes disk IO',
        artifact_tensor_bytes=sum(t.numel()*t.element_size() for a in artifacts
            for t in (a.mean,a.std,a.pca_mean,a.components,a.weight,a.bias)))
    if adapter is not None:
        result['adapter_parameter_bytes']=sum(p.numel()*p.element_size() for p in adapter.parameters())
        adapter.enabled=False
    return result
