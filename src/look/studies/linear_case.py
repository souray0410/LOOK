"""Matched affine-vector pilot, MHD replay and development-only reporting."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from look.runtime.state import stable_hash,file_sha256,atomic_write_json
from look.runtime.host_checkpoint import capture_rng,restore_rng,cpu_tree,atomic_save
from look.studies.project_case import read,evidence_files,verify_case as verify_source
from look.studies.mechanism_case import load_original
from look.studies.linear_protocol import VERSION,protocol,validate
from look.methods.linear_operator import ARMS,fit_bank,LinearFitPaused,LinearVectorArtifact
from look.methods.operator import load_selected_bank
from look.data.observed_pair import collate_observed
from look.evaluation.evaluator import evaluate_missing,save_prediction_bundle
from look.evaluation.observed_suite import check_matched,assemble_mixed
from look.training.mechanism_training import state_equal
from look.studies.spatial_case import rng_equal
from look.analysis.linear_report import report


class ProbeSubset(Subset):
    split='train';augment=False


def verify_case(root,spec):
    root=Path(root);r=read(root/'accepted.json')
    if (r.get('schema')!=VERSION or r.get('identity')!=stable_hash(spec) or r.get('test_access') is not False
        or r.get('state')!='accepted' or r.get('arms')!=list(ARMS) or r.get('profile') is not False
        or not all(r.get(k) is True for k in ('host_frozen','reload_exact','rng_restored','matched_report','pca_reference_replayed'))):
        raise ValueError('Linear comparison not accepted')
    required={'spec.json','development/suite.json','report/paired_statistics.json','report/metrics.csv','costs.json'}
    if not required.issubset(r['files']):raise ValueError('Missing matched comparison evidence')
    for name,sha in r['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha256(path)!=sha:raise ValueError('Linear evidence changed')
    return r


def dependencies(spec):
    validate(spec);s=spec['source'];root=Path(s['run_dir']);base=read(s['spec_path'])
    for p,sha in ((root/'accepted.json',s['accepted_sha256']),(root/'host/best.pt',s['best_sha256']),
                  (Path(s['spec_path']),s['spec_sha256'])):
        if file_sha256(p)!=sha:raise ValueError('Linear source dependency changed')
    verify_source(root,base)
    if (base['disease'],base['model']['name'],base['position'],base['seed'])!=tuple(spec['host'][k] for k in ('disease','architecture','position','seed')):
        raise ValueError('Host role mismatch')
    if not spec.get('source_pins'):raise ValueError('Missing immutable source pins')
    for row in spec['source_pins']:
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('Linear source changed')
    if spec.get('pilot'):
        p=spec['pilot'];ps=read(Path(p['run_dir'])/'spec.json')
        if file_sha256(Path(p['run_dir'])/'accepted.json')!=p['sha256']:raise ValueError('Pilot changed')
        verify_case(p['run_dir'],ps)
        if ps['host']!=dict(spec['host'],seed=3416) or ps['protocol']!=spec['protocol']:
            raise ValueError('Unmatched pilot')
    return base,root


def run(spec,out,device,pause,profile=False):
    import psutil
    base,source=dependencies(spec);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if device.type!='cuda' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Full study requires an admitted GPU Slurm step')
    if not profile:
        receipt_path=os.environ.get('LOOK_LINEAR_PROFILE_RECEIPT')
        if not receipt_path:raise ValueError('Full resource preflight acceptance required')
        r=read(receipt_path)
        if r.get('profile') is not True or r.get('state')!='accepted' or r.get('identity')!=stable_hash(spec):
            raise ValueError('Resource preflight identity mismatch')
        parent=Path(receipt_path).parent.resolve()
        for name,sha in r['files'].items():
            path=(parent/name).resolve()
            if not path.is_relative_to(parent) or file_sha256(path)!=sha:raise ValueError('Resource evidence changed')
    limit=int(os.environ.get('LOOK_WORKER_MEMORY_BYTES','0'))
    if limit<=0:raise ValueError('Explicit admitted worker RAM required')
    props=torch.cuda.get_device_properties(device);total=props.total_memory
    gpu_budget=min(.875*total,total-10*1024**3)
    if gpu_budget<=2*1024**3:raise MemoryError('Insufficient GPU reserve')
    torch.cuda.set_per_process_memory_fraction((gpu_budget-2*1024**3)/1.2/total,device)
    torch.cuda.reset_peak_memory_stats(device)
    rng=capture_rng();records=[];costs={};start=time.time()
    _,parents,graph,data=load_original(base,source,device);del parents
    graph.eval()
    for parameter in graph.parameters():parameter.requires_grad_(False)
    frozen=cpu_tree(graph.state_dict());node_ids=[(n.id,n.name) for n in graph.nodes]
    def loader(dataset):
        return DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0,collate_fn=collate_observed,
                          generator=torch.Generator().manual_seed(base['seed']))
    fit=data['fit'];dev=data['development']
    if profile:
        fit=ProbeSubset(fit,range(min(1024,len(fit))))
        dev=ProbeSubset(data['fit'],range(min(32,len(data['fit']))))
    train_loader=loader(fit);dev_loader=loader(dev);patterns=spec['protocol']['patterns']
    templates={p:load_selected_bank(source/'corrections/look'/p) for p in patterns}
    def check():
        if psutil.Process().memory_info().rss>.85*limit:raise MemoryError('Worker RAM reserve breached')
        return pause()
    def publish(method,scenario,result):
        path=out/'development'/f'{method}__{scenario}.npz';save_prediction_bundle(result,path)
        records.append(dict(method=method,scenario=scenario,path=str(path),sha256=file_sha256(path),metrics=result['metrics']))
    try:
        baseline={p:evaluate_missing(graph,dev_loader,device,fixed_pattern=p) for p in ['complete',*patterns]}
        reference={p:evaluate_missing(graph,dev_loader,device,fixed_pattern=p,artifact_banks={p:templates[p]}) for p in patterns}
        for p,r in baseline.items():publish('host',p,r)
        for p,r in reference.items():publish('reference_look',p,r)
        for arm in ARMS:
            begin=time.time();predictions={'complete':baseline['complete']};diagnostics={}
            for p in patterns:
                if check():raise LinearFitPaused()
                available=int(.85*limit-psutil.Process().memory_info().rss)
                bank,diagnostics[p]=fit_bank(graph,train_loader,templates[p],arm,device,out/arm/p,
                    identity=stable_hash(dict(spec=spec,profile=profile,pattern=p)),workspace_bytes=available,should_pause=check)
                path=out/arm/p/'bank.pt';atomic_save(path,[a.record() for a in bank])
                loaded=[LinearVectorArtifact.from_record(a) for a in torch.load(path,map_location='cpu',weights_only=False)]
                r=evaluate_missing(graph,dev_loader,device,fixed_pattern=p,artifact_banks={p:bank})
                replay=evaluate_missing(graph,dev_loader,device,fixed_pattern=p,artifact_banks={p:loaded})
                check_matched(r,replay);check_matched(r,reference[p])
                if not np.array_equal(r['logits'],replay['logits']):raise ValueError('Reload changed predictions')
                if arm=='shared_pca_ridge' and not profile:
                    if not np.allclose(r['logits'],reference[p]['logits'],rtol=1e-4,atol=1e-4) or not np.array_equal(r['logits'].argmax(1),reference[p]['logits'].argmax(1)):
                        raise ValueError('Recomputed PCA reference changed classifications or exceeded tolerance')
                predictions[p]=r;publish(arm,p,r)
            publish(arm,'complete',predictions['complete'])
            for ratio in base['look']['ratios']:
                publish(arm,f'mixed_{ratio:.1f}',assemble_mixed(predictions['complete'],{p:predictions[p] for p in patterns},ratio,base['look']['mask_seed']))
            atomic_write_json(diagnostics,out/arm/'diagnostics.json')
            costs[arm]=dict(seconds=time.time()-begin,
                            retained_fit_bytes=sum(p.stat().st_size for p in (out/arm).rglob('*') if p.is_file()),
                            inference_bank_bytes=sum(p.stat().st_size for p in (out/arm).glob('*/bank.pt')),
                            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),rank_is_not_equal_parameter_budget=True)
            state_equal(graph,frozen)
        if node_ids!=[(n.id,n.name) for n in graph.nodes]:raise ValueError('Node IDs changed')
        atomic_write_json(costs,out/'costs.json');atomic_write_json(dict(records=records,test_access=False),out/'development/suite.json')
        if not profile:report(records,out/'report')
    finally:
        state_equal(graph,frozen);restore_rng(rng)
    if not rng_equal(rng,capture_rng()):raise ValueError('RNG restore failed')
    receipt=dict(schema=VERSION,identity=stable_hash(spec),state='accepted',test_access=False,arms=list(ARMS),profile=profile,
        host_frozen=True,reload_exact=True,rng_restored=True,matched_report=not profile,pca_reference_replayed=not profile,
        seconds=time.time()-start,files=evidence_files(out))
    atomic_write_json(receipt,out/('profile_accepted.json' if profile else 'accepted.json'))
    return receipt


def execute(spec,output,device='cuda:0',profile=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);stop=False
    def pause(*args):
        nonlocal stop
        stop=True
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,pause)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not profile and (out/'accepted.json').exists():return verify_case(out,spec)
        if (out/'spec.json').exists() and read(out/'spec.json')!=spec:raise ValueError('Run identity changed')
        atomic_write_json(spec,out/'spec.json')
        try:
            result=run(spec,out,torch.device(device),lambda:stop or (out/'pause.json').exists(),profile)
            atomic_write_json(dict(state='completed',test_access=False,profile=profile),out/'status.json')
            return result
        except LinearFitPaused:
            atomic_write_json(dict(state='paused',test_access=False),out/'status.json');return dict(state='paused')
        except Exception as exc:
            import traceback
            atomic_write_json(dict(state='needs_review',error=repr(exc),traceback=traceback.format_exc(),test_access=False),out/'status.json')
            raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--profile',action='store_true');args=parser.parse_args()
    result=execute(read(args.spec),args.output,profile=args.profile)
    if result['state']=='paused':raise SystemExit(75)

if __name__=='__main__':main()
