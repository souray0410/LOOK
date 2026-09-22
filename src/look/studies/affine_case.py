"""Additive frozen-host family; immutable references, resumable fits and replay."""
import argparse
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import resource
import signal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.runtime.host_checkpoint import atomic_save, capture_rng, restore_rng, cpu_tree
from look.studies.project_case import read, evidence_files
from look.studies.affine_protocol import VERSION, validate
from look.studies import terminal_case, linear_case
from look.studies.mechanism_case import load_original
from look.studies.spatial_case import rng_equal
from look.methods.affine_family import ARMS, NEW_ARMS, REFERENCE_ARMS, FamilyArtifact, fit_map
from look.methods.linear_operator import fit_bank, fingerprint, LinearFitPaused
from look.methods.linear_vector import ResidualMoments, estimated_workspace_bytes
from look.methods.operator import LOOKArtifact, load_selected_bank
from look.data.observed_pair import collate_observed
from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle
from look.evaluation.observed_suite import check_matched, assemble_mixed
from look.training.mechanism_training import state_equal
from look.analysis.affine_report import report


def verify_case(root,spec):
    validate(spec)
    root=Path(root);r=read(root/'accepted.json')
    if (r.get('identity')!=stable_hash(spec) or r.get('schema')!=VERSION or r.get('test_access') is not False
        or r.get('state')!='accepted' or r.get('profile') is not False
        or not all(r.get(k) is True for k in ('host_frozen','reload_exact','rng_restored','full_mhd_replay'))):
        raise ValueError('Affine case not accepted')
    if not {'spec.json','costs.json','development/suite.json','report/paired_statistics.json'}.issubset(r['files']):
        raise ValueError('Incomplete affine evidence')
    for name,sha in r['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or file_sha256(p)!=sha:raise ValueError('Affine evidence changed')
    return r


def dependencies(spec):
    validate(spec);s=spec['source'];root=Path(s['run_dir']);prior=read(root/'spec.json')
    if file_sha256(root/'spec.json')!=s['spec_sha256'] or file_sha256(root/'accepted.json')!=s['accepted_sha256']:
        raise ValueError('Reference identity changed')
    if prior['host']!=spec['host']:raise ValueError('Reference host changed')
    module=terminal_case if spec['scope']=='terminal' else linear_case
    module.verify_case(root,prior)
    base,host_root=module.dependencies(prior)
    if not spec.get('source_pins'):raise ValueError('Missing immutable source')
    for row in spec['source_pins']:
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('Scientific source changed')
    if spec.get('pilot'):
        p=spec['pilot'];ps=read(Path(p['run_dir'])/'spec.json');verify_case(p['run_dir'],ps)
        if (ps['host']!=dict(spec['host'],seed=3416) or ps['protocol']!=spec['protocol']
            or file_sha256(Path(p['run_dir'])/'accepted.json')!=p['sha256']):raise ValueError('Pilot mismatch')
    return base,host_root,root


def terminal_fit(template, cache_root, pattern, output, identity, check, workspace):
    """One verified statistics stream shared across new methods, with shard cursor."""
    out=Path(output);out.mkdir(parents=True,exist_ok=True);a=template
    if estimated_workspace_bytes(a.std.numel(),a.latent_dim)>workspace:
        raise MemoryError('Dense affine workspace infeasible; no silent reduction')
    path=out/'moments.pt';s=ResidualMoments.empty(a.std.numel());cursor=0
    if path.exists():
        r=torch.load(path,map_location='cpu',weights_only=False)
        if r['identity']!=identity or fingerprint(r['statistics'])!=r['sha256']:raise ValueError('Statistics resume changed')
        s=ResidualMoments(**r['statistics']);cursor=r['cursor']
    def save(n):
        v=asdict(s);atomic_save(path,dict(identity=identity,cursor=n,statistics=v,sha256=fingerprint(v)))
    last=time.monotonic();n=0
    for n,row in enumerate(terminal_case.cached_rows(cache_root),1):
        if n<=cursor:continue
        if check():save(n-1);raise LinearFitPaused()
        x=row['features'][pattern];f=row['features']['complete'];s.update((x-a.mean)/a.std,(f-x)/a.std)
        if time.monotonic()-last>120:save(n);last=time.monotonic()
    if n<cursor:raise ValueError('Cache shortened')
    save(n)
    return s


def run(spec,out,device,pause,profile=False):
    import psutil
    base,host_root,reference=dependencies(spec);out=Path(out)
    if device.type!='cuda' or not os.environ.get('SLURM_JOB_ID'):raise ValueError('Admitted Slurm GPU required')
    limit=int(os.environ.get('LOOK_WORKER_MEMORY_BYTES','0'))
    if limit<=0:raise ValueError('Worker RAM must be admitted')
    if not profile:
        rp=Path(os.environ['LOOK_AFFINE_PROFILE_RECEIPT']);r=read(rp)
        if r.get('identity')!=stable_hash(spec) or r.get('state')!='accepted' or r.get('profile') is not True:
            raise ValueError('Affine profile mismatch')
        for name,sha in r['files'].items():
            p=(rp.parent/name).resolve()
            if not p.is_relative_to(rp.parent.resolve()) or file_sha256(p)!=sha:raise ValueError('Profile changed')
    torch.set_num_threads(min(4,int(os.environ.get('SLURM_CPUS_PER_TASK','4'))))
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    props=torch.cuda.get_device_properties(device);total=props.total_memory
    budget=min(total,int(os.environ.get('LOOK_AFFINE_GPU_BUDGET_BYTES',str(total))))
    if budget<=0:raise MemoryError('Insufficient admitted GPU budget')
    from mhd_models.scheduling.gpu_budget import configure_allocator
    budget=configure_allocator(device,cap=budget);torch.cuda.reset_peak_memory_stats(device)
    rng=capture_rng();_,parents,graph,data=load_original(base,host_root,device);del parents
    graph.eval()
    for p in graph.parameters():p.requires_grad_(False)
    frozen=cpu_tree(graph.state_dict());nodes=[(n.id,n.name) for n in graph.nodes]
    def loader(ds):return DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,collate_fn=collate_observed,generator=torch.Generator().manual_seed(base['seed']))
    fit=data['fit'];dev=data['development']
    if profile:
        fit=linear_case.ProbeSubset(fit,range(min(1024,len(fit))));dev=linear_case.ProbeSubset(data['fit'],range(min(32,len(data['fit']))))
    train_loader=loader(fit);dev_loader=loader(dev);costs={};start=time.time();records=[]
    def status(stage,**extra):atomic_write_json(dict(state='running',stage=stage,updated_at=time.time(),test_access=False,**extra),out/'status.json')
    def check():
        if psutil.Process().memory_info().rss>.85*limit:raise MemoryError('RAM reserve breached')
        return pause()
    def publish(method,pattern,result):
        p=out/'development'/f'{method}__{pattern}.npz';save_prediction_bundle(result,p)
        records.append(dict(method=method,scenario=pattern,path=str(p),sha256=file_sha256(p),metrics=result['metrics']))
    try:
        patterns=spec['protocol']['patterns'];baseline={};predictions={}
        if not profile:
            records=[r for r in read(reference/'development/suite.json')['records'] if r['method'] in (*REFERENCE_ARMS,'host')]
            for r in records:
                if file_sha256(r['path'])!=r['sha256']:raise ValueError('Reference predictions changed')
        for p in ['complete',*patterns]:
            baseline[p]=evaluate_missing(graph,dev_loader,device,fixed_pattern=p)
            if not profile:
                r=next(r for r in records if r['method']=='host' and r['scenario']==p)
                with np.load(r['path'],allow_pickle=False) as f:
                    for k in ('labels','participant_ids','logits'):
                        if not np.array_equal(baseline[p][k],f[k]):raise ValueError('Frozen host replay changed')
        if spec['scope']=='terminal' and profile:
            terminal_case.cache(graph,train_loader,device,out/'probe_cache',stable_hash(spec),check)
        templates={}
        for pattern in patterns:
            if spec['scope']=='terminal':
                selection=read(reference/'selection'/f'{pattern}.json')
                a=LOOKArtifact.load(reference/'candidates'/pattern/f"q{selection['rank']}.pt")
                templates[pattern]=[a]
                cache=out/'probe_cache' if profile else reference/'cache/train'
                status('statistics',pattern=pattern)
                stats=terminal_fit(a,cache,pattern,out/'statistics'/pattern,stable_hash(dict(spec=spec,profile=profile,pattern=pattern)),check,int(.85*limit-psutil.Process().memory_info().rss))
            else:templates[pattern]=load_selected_bank(host_root/'corrections/look'/pattern)
            for arm in NEW_ARMS:
                if check():raise LinearFitPaused()
                status('fit_and_replay',arm=arm,pattern=pattern);begin=time.time();folder=out/'mappings'/arm/pattern;folder.mkdir(parents=True,exist_ok=True)
                identity=stable_hash(dict(spec=spec,profile=profile,arm=arm,pattern=pattern))
                bank_path=folder/'bank.pt'
                if bank_path.exists():
                    saved=torch.load(bank_path,map_location='cpu',weights_only=False)
                    if saved['identity']!=identity or fingerprint(saved['bank'])!=saved['sha256']:raise ValueError('Bank changed')
                    bank=[FamilyArtifact.from_record(a) for a in saved['bank']]
                else:
                    if spec['scope']=='terminal':bank=[FamilyArtifact(a,fit_map(stats,a.latent_dim,a.ridge_lambda,arm=arm,basis=a.components))]
                    else:bank,_=fit_bank(graph,train_loader,templates[pattern],arm,device,folder,identity=identity,
                        workspace_bytes=int(.85*limit-psutil.Process().memory_info().rss),should_pause=check,family=True)
                    saved=[a.record() for a in bank];atomic_save(bank_path,dict(identity=identity,bank=saved,sha256=fingerprint(saved)))
                loaded=[FamilyArtifact.from_record(a) for a in torch.load(bank_path,map_location='cpu',weights_only=False)['bank']]
                if spec['scope']=='terminal' and not profile:
                    classifier=graph.get_edge_by_name('fusion_classifier_edge').edge_operations[0].function
                    r=terminal_case.result_from_cache(reference/'cache/development',pattern,classifier,device,bank[0])
                else:
                    r=evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern,artifact_banks={pattern:bank})
                replay=evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern,artifact_banks={pattern:loaded})
                check_matched(r,replay)
                if not np.array_equal(r['logits'],replay['logits']):raise ValueError('Reload changed full MHD predictions')
                predictions[arm,pattern]=r;publish(arm,pattern,r)
                costs[arm+'/'+pattern]=dict(seconds=time.time()-begin,bank_bytes=bank_path.stat().st_size,
                    diagnostics=[a.mapping.diagnostics for a in bank],empty_selected_bank=not bool(bank))
        for arm in NEW_ARMS:
            publish(arm,'complete',baseline['complete'])
            for ratio in base['look']['ratios']:
                publish(arm,f'mixed_{ratio:.1f}',assemble_mixed(baseline['complete'],{p:predictions[arm,p] for p in patterns},ratio,base['look']['mask_seed']))
        state_equal(graph,frozen)
        if nodes!=[(n.id,n.name) for n in graph.nodes]:raise ValueError('Node identity changed')
        atomic_write_json(dict(records=records,test_access=False,scope=spec['scope']),out/'development/suite.json')
        status('paired_statistics')
        if not profile:report([(spec['host'],spec['scope'],records)],out/'report')
        costs.update(seconds=time.time()-start,gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            allocation=os.environ['SLURM_JOB_ID'],device=props.name,test_access=False)
        if costs['gpu_peak_reserved_bytes']>budget or costs['host_peak_rss_bytes']>.85*limit:
            raise MemoryError('Resource acceptance failed')
        atomic_write_json(costs,out/'costs.json')
    finally:
        state_equal(graph,frozen);restore_rng(rng)
    if not rng_equal(rng,capture_rng()):raise ValueError('RNG restore failed')
    receipt=dict(schema=VERSION,identity=stable_hash(spec),state='accepted',profile=profile,test_access=False,
        host_frozen=True,reload_exact=True,rng_restored=True,full_mhd_replay=True,files=evidence_files(out))
    atomic_write_json(receipt,out/('profile_accepted.json' if profile else 'accepted.json'));return receipt


def execute(spec,out,profile=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);stop=False
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
            result=run(spec,out,torch.device('cuda:0'),lambda:stop or (out/'pause.json').exists(),profile)
            atomic_write_json(dict(state='completed',profile=profile,test_access=False,updated_at=time.time()),out/'status.json');return result
        except LinearFitPaused:
            atomic_write_json(dict(state='paused',test_access=False,updated_at=time.time()),out/'status.json');return dict(state='paused')
        except Exception as e:
            import traceback
            atomic_write_json(dict(state='needs_review',error=repr(e),traceback=traceback.format_exc(),test_access=False),out/'status.json');raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);p.add_argument('--profile',action='store_true');a=p.parse_args()
    if execute(read(a.spec),a.output,a.profile)['state']=='paused':raise SystemExit(75)

if __name__=='__main__':main()
