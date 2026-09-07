"""Frozen train/validation replay and matched inference benchmark. No fitting."""
from pathlib import Path
import gc
import os
import time
from unittest.mock import patch
import numpy as np
import torch
from .dual_queue import read, record, verify, atomic, digest
from .supplement_diagnostics import PATTERNS, validate_scope, save_npz
from .stable_metrics import logit_metrics
from .test_runtime import load_graph, compare_replay
from .test_evaluation import job_artifacts
from .data import UKBBilateralVisitDataset, make_loader
from .look import load_selected_bank, downsample_flatten
from .joint import read_site, correction_sites
from .method_ssf import SSFAdapter
from .filling import NormalizedMeanFiller
from .distributed import module_state_sha256
from .reproducibility import seed_everything
from . import method_kernels as kernels


def loader_for(model, split, microbatch, limit=None):
    if split not in ('train','validation'):raise ValueError('Diagnostic loaders prohibit test')
    c=model['base']['config']
    ds=UKBBilateralVisitDataset(labels_csv=verify(model['base']['labels']),data_root=c['image_root'],
        image_size=c['image_size'],split=split,augment=False,limit=limit,
        preprocess_cache_root=c['preprocess_cache_root'])
    if limit is None and len(ds)!={'train':1264,'validation':296}[split]:raise ValueError('Incomplete diagnostic split')
    return make_loader(ds,microbatch,0,False,model['seed'],c['sampling_strategy'])


def tensor_rows(before, after, target, artifact):
    if before.shape!=after.shape or before.shape!=target.shape:raise ValueError('Paired feature geometry differs')
    b,a,t=[v.flatten(1).double() for v in (before,after,target)]
    delta=(a-b).norm(dim=1);bn=b.norm(dim=1)
    result=dict(before_l2=bn,after_l2=a.norm(dim=1),target_l2=t.norm(dim=1),delta_l2=delta,
                relative_delta=delta/(bn+1e-12),mse_before=(b-t).square().mean(1),mse_after=(a-t).square().mean(1))
    # Same complete-training PCA coordinates for all three states. No fitting.
    vectors=[]
    for value in (before,after,target):
        flat,shape=downsample_flatten(value,artifact.factor)
        if tuple(shape)!=tuple(artifact.downsample_shape):raise ValueError('Artifact geometry changed')
        z=((flat-artifact.mean.to(flat.device))/artifact.std.to(flat.device)-artifact.pca_mean.to(flat.device)) @ artifact.components.to(flat.device).T
        vectors.append(z.double())
    zb,za,zt=vectors
    result.update(latent_mse_before=(zb-zt).square().mean(1),latent_mse_after=(za-zt).square().mean(1))
    if any(not torch.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite diagnostic features')
    return {k:v.detach().cpu().numpy() for k,v in result.items()}


def distribution(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(a.max()))


@torch.no_grad()
def prefix_replay(model, graph, bank, pattern, split, loader, output, identity):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);results=[];outputs=[]
    available=correction_sites(graph)
    names=[a.node_name for a in bank]
    if names!=[n for n in available if n in names]:raise ValueError('Bank is not an ordered prefix path')
    filler=NormalizedMeanFiller()
    for k in range(len(bank)+1):
        path=out/f'prefix_{k:02d}.json'
        if path.exists():
            old=read(path)
            if old['identity']!=identity:raise ValueError('Cross-model/split/batch prefix resume')
            for rec in old['outputs']:verify(rec)
            results.append(old);outputs.extend([record(path),*old['outputs']]);continue
        ys=[];zs=[];ids=[];final_norm=[];traces={};node=bank[k-1].node_name if k else 'all_off'
        for bi,batch in enumerate(loader):
            o,c=batch['oct'].to(graph.device),batch['cfp'].to(graph.device)
            target=None
            if k:
                # Target captured at the exact same stop site as the original fit.
                target=kernels.forward_control(graph,o,c,stop_node=node).detach().clone()
            mo,mc=filler.fill(o,c,pattern)
            actual_write=kernels.write_site
            def traced(g,name,value):
                if name==node:
                    values=tensor_rows(read_site(g,name),value,target,bank[k-1])
                    for key,array in values.items():traces.setdefault(key,[]).append(array)
                return actual_write(g,name,value)
            with patch.object(kernels,'write_site',traced):
                logits=kernels.forward_control(graph,mo,mc,bank[:k],policy=model['policy'],pattern=pattern)
            if bi==0:
                plain=kernels.forward_control(graph,mo,mc,bank[:k],policy=model['policy'],pattern=pattern)
                if not torch.allclose(logits,plain,atol=1e-6,rtol=1e-6):raise ValueError('Tracing changed inference')
            zs.append(logits.detach().cpu().numpy());ys.append(batch['label'].numpy());ids.extend(map(str,batch['participant_id']))
            final_norm.extend(read_site(graph,'fusion_participant_feature').flatten(1).norm(dim=1).cpu().tolist())
            if bi%10==0:print(f'{model["id"]} {pattern} {split} prefix={k}/{len(bank)} batch={bi}/{len(loader)}',flush=True)
        labels,logits=np.concatenate(ys),np.concatenate(zs).astype(np.float64)
        if len(set(ids))!=len(ids) or len(ids)!=len(loader.dataset):raise ValueError('Incomplete/duplicate diagnostic participants')
        metrics=logit_metrics(labels,logits);trace={key:np.concatenate(a) for key,a in traces.items()}
        npz=path.with_suffix('.npz');save_npz(npz,labels=labels,logits=logits,participant_ids=np.asarray(ids),final_feature_l2=np.asarray(final_norm))
        saved=[record(npz)];norms={key:distribution(a) for key,a in trace.items()}
        if trace:
            tp=out/f'prefix_{k:02d}_features.npz';save_npz(tp,**trace);saved.append(record(tp))
        drift=None
        if split=='validation' and k==len(bank):
            scenario='look_after_fill_'+pattern if model['policy']=='joint' else pattern
            drift=compare_replay(dict(participant_ids=np.asarray(ids),labels=labels,logits=logits),model['validation_predictions'][scenario])
        summary=dict(identity=identity,prefix=k,node=node,n=len(ids),metrics=metrics,feature_distributions=norms,
                     final_feature_l2=distribution(final_norm),outputs=saved,historical_replay=drift,
                     feature_unit='participant' if node=='fusion_participant_feature' else 'eye',
                     test_access=False,upstream=[a.node_name for a in bank[:max(k-1,0)]])
        atomic(summary,path);results.append(summary);outputs.extend([record(path),*saved])
    rows=[]
    for v in results:
        row=dict(seed=model['seed'],policy=model['policy'],pattern=pattern,split=split,prefix=v['prefix'],node=v['node'],n=v['n'],
                 macro_f1=v['metrics']['macro_f1'],auroc=v['metrics']['macro_auroc_ovr'],nll=v['metrics']['negative_log_likelihood'],
                 final_feature_mean_l2=v['final_feature_l2']['mean'],final_feature_max_l2=v['final_feature_l2']['maximum'])
        for key,stats in v['feature_distributions'].items():
            for name,value in stats.items():row[key+'_'+name]=value
        rows.append(row)
    rp=out.parent/(out.name+'_stability.json');atomic(dict(rows=rows,test_access=False,identity=identity),rp);outputs.append(record(rp))
    return outputs


@torch.no_grad()
def benchmark(models, graph, loader, device, repeats=20, warmup=3):
    batch=next(iter(loader));o,c=batch['oct'].to(device),batch['cfp'].to(device)
    filler=NormalizedMeanFiller();rows=[];raw_samples={}
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    original=next(m for m in models if m['policy']=='joint')
    settings=[('filling',original),*[(m['policy'],m) for m in models]]
    for pattern in PATTERNS:
        for policy,model in settings:
            bank=[] if policy in ('filling','ssf','logit_affine') else load_selected_bank(Path(model['banks'][pattern]['root']))
            adapter=None
            if policy=='ssf':
                adapter=SSFAdapter(graph,pattern);adapter.load_state_dict(torch.load(verify(model['parameters'][pattern]),map_location=device,weights_only=True));adapter.eval();adapter.enabled=True
            def run():
                mo,mc=filler.fill(o,c,pattern)
                z=kernels.forward_control(graph,mo,mc,bank,policy=policy if bank else 'joint',pattern=pattern)
                if policy=='logit_affine':
                    from .method_logit import transform_logits
                    z=transform_logits(z.cpu().numpy(),model['parameters'][pattern])
                return z
            try:
                for _ in range(warmup):run()
                sync()
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                seconds=[]
                for _ in range(repeats):
                    sync();start=time.perf_counter();run();sync();seconds.append(time.perf_counter()-start)
                raw_samples[policy+'__'+pattern]=seconds
                learnable=sum(a.weight.numel()+a.bias.numel() for a in bank)
                if adapter is not None:learnable=sum(p.numel() for p in adapter.parameters())
                if policy=='logit_affine':learnable=2
                rows.append(dict(seed=model['seed'],policy=policy,pattern=pattern,participants_per_batch=len(o),
                    median_ms_per_participant=1000*float(np.median(seconds))/len(o),
                    p95_ms_per_batch=1000*float(np.quantile(seconds,.95)),warmup=warmup,repeats=repeats,
                    fitted_parameter_count=learnable,bank_tensor_bytes=sum(t.numel()*t.element_size() for a in bank for t in (a.mean,a.std,a.pca_mean,a.components,a.weight,a.bias)),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
                    device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                    physical_gpu=os.environ.get('LOOK_PHYSICAL_GPU'),physical_uuid=os.environ.get('LOOK_ASSIGNED_GPU_UUID'),
                    interpretation='shared workstation, sequential matched batch; allocator peaks are not total process VRAM'))
            finally:
                if adapter is not None:adapter.close()
    return dict(rows=rows,raw_seconds_per_batch=raw_samples,test_access=False,
                precision='FP32 model/correction; existing logit-affine kernel uses float64 CPU',
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                includes='filling, correction and inference',excludes='data loading, model loading, fitting')


def run_seed(manifest_path, output, seed, microbatch=8, device='cuda:0', limit=None):
    plan=read(manifest_path);validate_scope(plan);models=[m for m in plan['models'] if m['seed']==seed]
    if len(models)!=4:raise ValueError('Incomplete seed scope')
    for m in models:
        for r in job_artifacts(m):verify(r)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    identity=dict(manifest=record(manifest_path),seed=seed,microbatch=microbatch,limit=limit)
    done=out/'complete.json'
    if done.exists():
        old=read(done)
        if old['identity']!=identity:raise ValueError('Cross-execution resume')
        for r in old['outputs']:verify(r)
        return old
    seed_everything(seed);torch.set_num_threads(2);device=torch.device(device)
    if device.type=='cuda':
        if torch.cuda.device_count()!=1:raise ValueError('One assigned GPU per worker required')
        torch.cuda.set_per_process_memory_fraction(min(11*1024**3/torch.cuda.get_device_properties(0).total_memory,1.),0)
    original=next(m for m in models if m['policy']=='joint');graph=load_graph(original,device,microbatch)
    before=module_state_sha256(graph);outputs=[]
    try:
        loaders={s:loader_for(original,s,microbatch,limit) for s in ('train','validation')}
        if set(loaders['train'].dataset.participant_ids)&set(loaders['validation'].dataset.participant_ids):raise ValueError('Split overlap')
        cost=benchmark(models,graph,loaders['validation'],device,repeats=2 if limit else 20,warmup=1 if limit else 3)
        path=out/'benchmark.json';atomic(cost,path);outputs.append(record(path))
        for model in models:
            if model['policy'] not in ('joint','self_input_missing_only'):continue
            for pattern in PATTERNS:
                bank=load_selected_bank(Path(model['banks'][pattern]['root']))
                for split,loader in loaders.items():
                    path=out/f'micro_{microbatch}'/model['policy']/pattern/split
                    ident=dict(**identity,policy=model['policy'],pattern=pattern,split=split,model_sha256=digest(model))
                    outputs.extend(prefix_replay(model,graph,bank,pattern,split,loader,path,ident))
        if module_state_sha256(graph)!=before:raise ValueError('Diagnostics mutated the frozen backbone')
        value=dict(status='complete',identity=identity,test_access=False,outputs=outputs,backbone_unchanged=True,
                   acceptance_only=limit is not None,configuration_search=False,correction_refits=0)
        atomic(value,done);return value
    finally:
        del graph;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
