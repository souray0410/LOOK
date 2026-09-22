"""Frozen-host three-route LOOK fitting, replay, matched reporting and safe resume."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import resource
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from look.studies.spatial_protocol import VERSION,ROUTES,protocol,validate
from look.studies.project_case import read,evidence_files,verify_case as verify_host,CheckedLoader,Paused
from look.studies.mechanism_case import load_original
from look.runtime.state import stable_hash,file_sha256,atomic_write_json
from look.runtime.host_checkpoint import capture_rng,restore_rng,cpu_tree
from look.training.mechanism_training import state_equal
from look.data.observed_pair import collate_observed
from look.methods.operator import prepare_complete_pca_bank,greedy_fit_look,load_selected_bank,forward_with_look
from look.methods.joint import correction_sites,read_site
from look.evaluation.evaluator import evaluate_missing,save_prediction_bundle
from look.evaluation.observed_suite import check_matched,assemble_mixed
from look.analysis.spatial_report import report


def verify_case(root,spec):
    root=Path(root);r=read(root/'accepted.json')
    if (r.get('schema')!=VERSION or r.get('identity')!=stable_hash(spec) or r.get('test_access') is not False
        or r.get('state')!='accepted' or r.get('routes')!=list(ROUTES)
        or not all(r.get(k) is True for k in ('host_frozen','reload_exact','rng_restored','matched_report'))):
        raise ValueError('Spatial case incomplete or changed')
    required={'spec.json','report/paired_statistics.json','report/metrics.csv','costs.json','development/suite.json'}
    if not required.issubset(r['files']):raise ValueError('Spatial report evidence missing')
    for name,sha in r['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or file_sha256(p)!=sha:raise ValueError('Spatial evidence changed')
    return r


def dependencies(spec):
    validate(spec);source=spec['source'];run=Path(source['run_dir']);base=read(source['spec_path'])
    for p,sha in ((run/'accepted.json',source['accepted_sha256']),(run/'host/best.pt',source['best_sha256']),
                  (Path(source['spec_path']),source['spec_sha256'])):
        if file_sha256(p)!=sha:raise ValueError('Spatial host provenance changed')
    verify_host(run,base)
    if (base['disease'],base['model']['name'],base['position'],base['seed'])!=tuple(spec['host'][k] for k in ('disease','architecture','position','seed')):
        raise ValueError('Spatial host role mismatch')
    for row in spec['source_pins']:
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('Spatial source changed')
    if spec.get('pilot'):
        p=spec['pilot'];pilot_spec=read(Path(p['run_dir'])/'spec.json')
        if file_sha256(Path(p['run_dir'])/'accepted.json')!=p['sha256']:raise ValueError('Pilot receipt changed')
        verify_case(p['run_dir'],pilot_spec)
        if pilot_spec['host']!=dict(spec['host'],seed=3416) or pilot_spec['protocol']!=spec['protocol']:
            raise ValueError('Unmatched pilot; cannot unlock replication')
    return base,run


class ProbeSubset(Subset):
    split='train';augment=False


def run(spec,out,device,paused,profile=False):
    import psutil
    base,host=dependencies(spec);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    total=torch.cuda.get_device_properties(device).total_memory
    budget=total
    if budget<=0:raise ValueError('Insufficient GPU reserve')
    from mhd_models.scheduling.gpu_budget import configure_allocator
    budget=configure_allocator(device,cap=budget)
    torch.cuda.reset_peak_memory_stats(device)
    rng=capture_rng();start=time.time();p=spec['protocol'];costs={};records=[]
    _,models,graph,data=load_original(base,host,device)
    del models
    for parameter in graph.parameters():parameter.requires_grad_(False)
    frozen=cpu_tree(graph.state_dict());nodes=[(n.id,n.name) for n in graph.nodes]
    ram_limit=int(os.environ.get('LOOK_WORKER_MEMORY_BYTES',100*1024**3));process=psutil.Process()
    def check():
        if process.memory_info().rss>ram_limit*.85:raise MemoryError('LOOK worker RAM reserve breached')
        return paused()
    def loader(dataset):
        return CheckedLoader(DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0,
            collate_fn=collate_observed,generator=torch.Generator().manual_seed(base['seed'])),check)
    train_data=data['fit'];dev_data=data['development']
    if profile:
        # Independent train-only probe; no formal fitting result is reused.
        train_data=ProbeSubset(train_data,list(range(min(len(train_data),1024))))
        dev_data=ProbeSubset(data['fit'],list(range(min(len(data['fit']),32))))
    train_loader=loader(train_data);dev_loader=loader(dev_data);sites=correction_sites(graph)
    first=next(iter(train_loader))
    with torch.no_grad():
        forward_with_look(graph,first['oct'].to(device),first['cfp'].to(device),counts=first['counts'])
    # Conservative dense-SVD workspace estimate before any full-resolution fitting.
    dimensions=[int(np.prod(read_site(graph,n).shape[1:])) for n in sites]
    from look.methods.spatial import reduce_spatial
    common_ranks={n:min(p['max_rank'],len(train_data)-1,*[
        reduce_spatial(read_site(graph,n),f,m).flatten(1).shape[1]
        for m in ROUTES for f in p['factors'][m]]) for n in sites}
    atomic_write_json(dict(policy='same_feasible_rank_at_each_site_all_routes',ranks=common_ranks),out/'common_ranks.json')
    estimate=20*8*max(2*p['max_rank'],512)*max(dimensions)+4*p['max_rank']*sum(dimensions)+process.memory_info().rss
    atomic_write_json(dict(estimated_host_bytes=estimate,worker_limit=ram_limit,feature_dimensions=dimensions),out/'resource_estimate.json')
    if estimate>ram_limit*.85:raise MemoryError('Direct-PCA conservative workspace exceeds RAM; request larger measured resource class, never shrink features')
    baseline={pattern:evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern) for pattern in ('complete','oct_missing','cfp_missing')}
    def publish(method,scenario,result):
        path=out/'development'/f'{method}__{scenario}.npz';save_prediction_bundle(result,path)
        records.append(dict(method=method,scenario=scenario,metrics=result['metrics'],path=str(path),sha256=file_sha256(path)))
    for pattern,result in baseline.items():publish('host',pattern,result)
    for route in ROUTES:
        t=time.time();atomic_write_json(dict(state='running',stage=route,profile=profile,time=t,test_access=False),out/'status.json')
        factors=p['factors'][route]
        bank={}
        for site in sites:
            bank.update(prepare_complete_pca_bank(graph,train_loader,[site],factors,common_ranks[site],device,
                out/route/'pca',dict(identity=stable_hash(spec),profile=profile,route=route,common_ranks=common_ranks),out/'quarantine',
                spatial_method=route,strict_rank=True))
        predictions={'complete':baseline['complete']}
        for pattern in p['patterns']:
            target=out/route/pattern
            if (target/'factor_selection.json').exists():artifacts=load_selected_bank(target)
            else:artifacts,_=greedy_fit_look(graph,train_loader,dev_loader,pattern,sites,factors,p['latent_dims'],p['max_rank'],device,target,bank)
            predictions[pattern]=evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern,artifact_banks={pattern:artifacts})
            reloaded=load_selected_bank(target)
            replay=evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern,artifact_banks={pattern:reloaded})
            check_matched(predictions[pattern],replay)
            if not np.array_equal(predictions[pattern]['logits'],replay['logits']):raise ValueError('Spatial artifact reload diverged')
            check_matched(baseline[pattern],predictions[pattern]);publish(route,pattern,predictions[pattern])
        publish(route,'complete',predictions['complete'])
        for ratio in base['look']['ratios']:
            mixed=assemble_mixed(predictions['complete'],{k:predictions[k] for k in p['patterns']},ratio,base['look']['mask_seed'])
            publish(route,f'mixed_{ratio:.1f}',mixed)
        state_equal(graph,frozen)
        costs[route]=dict(seconds=time.time()-t,peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device),
            cumulative_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
            artifact_bytes=sum(x.stat().st_size for x in (out/route).rglob('*') if x.is_file()))
        del bank
    if nodes!=[(n.id,n.name) for n in graph.nodes]:raise ValueError('MHD Node IDs changed')
    state_equal(graph,frozen);restore_rng(rng)
    # capture_rng contains NumPy arrays; compare through explicit recursive helper.
    if not rng_equal(rng,capture_rng()):raise ValueError('Random state changed after restoration')
    atomic_write_json(costs,out/'costs.json');atomic_write_json(dict(records=records,test_access=False),out/'development/suite.json')
    if not profile:report(records,out/'report')
    receipt=dict(schema=VERSION,identity=stable_hash(spec),state='accepted',test_access=False,routes=list(ROUTES),
        host_frozen=True,reload_exact=True,rng_restored=True,matched_report=not profile,profile=profile,seconds=time.time()-start,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),files=evidence_files(out))
    atomic_write_json(receipt,out/('profile_accepted.json' if profile else 'accepted.json'))
    return receipt


def rng_equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(rng_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(rng_equal(x,y) for x,y in zip(a,b))
    return a==b


def execute(spec,output,device='cuda:0',profile=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);stop=False
    def pause(*_):
        nonlocal stop
        stop=True
    for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,pause)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not profile and (out/'accepted.json').exists():return verify_case(out,spec)
        if (out/'spec.json').exists() and read(out/'spec.json')!=spec:raise ValueError('Spatial run identity changed')
        atomic_write_json(spec,out/'spec.json')
        try:r=run(spec,out,torch.device(device),lambda:stop or (out/'pause.json').exists(),profile)
        except Paused:
            r=dict(state='paused');atomic_write_json(r,out/'status.json');return r
        except Exception as exc:
            import traceback
            atomic_write_json(dict(state='needs_review',error=repr(exc),traceback=traceback.format_exc(),test_access=False),out/'status.json');raise
        atomic_write_json(dict(state='completed',test_access=False),out/'status.json');return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);p.add_argument('--profile',action='store_true')
    a=p.parse_args();result=execute(read(a.spec),a.output,profile=a.profile)
    if result['state']=='paused':raise SystemExit(75)

if __name__=='__main__':main()
