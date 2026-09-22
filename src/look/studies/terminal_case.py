"""Early accepted-host single-final stage; no dependency on other PCA sites.

This stage is a separately counted subset/extension, never a completed full LOOK
case. Cached terminal features replace repeated identical frozen forwards only.
"""
import argparse
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import random
import resource
import signal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from look.runtime.state import atomic_write_json, stable_hash, file_sha256
from look.runtime.host_checkpoint import atomic_save, capture_rng, restore_rng, cpu_tree
from look.studies.project_case import read, evidence_files, validate_spec
from look.studies.mechanism_case import load_original
from look.training.mechanism_training import state_equal
from look.methods.operator import (forward_with_look, apply_artifact, fit_complete_pca,
    fit_look_node, FullFeaturePCA, LOOKArtifact)
from look.methods.linear_vector import ResidualMoments, solve
from look.methods.linear_operator import ARMS, LinearVectorArtifact, fingerprint
from look.methods.joint import read_site
from look.methods.imputation import NormalizedMeanFiller
from look.data.observed_pair import collate_observed
from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle
from look.evaluation.observed_suite import check_matched, assemble_mixed
from look.evaluation.stability import logit_metrics, probabilities_from_logits
from look.analysis.terminal_report import report, COMPARISONS

VERSION='look_terminal_stage_v1'
SITE='fusion_participant_feature'
PATTERNS=('complete','oct_missing','cfp_missing')


def protocol():
    return dict(schema=VERSION,site=SITE,scope='single_final_conditional_mechanism_not_full_progressive_LOOK',
        arms=list(ARMS),patterns=list(PATTERNS),selection='original_PCA_candidate_grid_and_train_GCV',
        rank_penalty='all_three_arms_share_PCA_best_candidate_rank_and_lambda',
        disabled_candidate='report_original_on_off_choice_separately; compare_candidate_even_if_disabled',
        spatial_reduction='not_applicable_to_terminal_vector',test_access=False,
        comparisons=COMPARISONS,bootstrap_iterations=10000,seed_gate='technical_acceptance_only_not_positive_performance')


def dependencies(spec):
    if spec.get('schema')!=VERSION or spec.get('protocol')!=protocol() or spec.get('test_access') is not False:
        raise ValueError('Unregistered terminal stage')
    s=spec['source'];root=Path(s['run_dir']);base=read(s['spec_path']);validate_spec(base)
    for path,digest in ((s['spec_path'],s['spec_sha256']),
        (root/'host/accepted.json',s['host_accepted_sha256']),(root/'host/best.pt',s['best_sha256'])):
        if file_sha256(path)!=digest:raise ValueError('Source changed')
    receipt=read(root/'host/accepted.json')
    if receipt.get('state')!='accepted' or receipt.get('identity')!=stable_hash(base):
        raise ValueError('Host not accepted')
    for name,digest in receipt['files'].items():
        p=(root/'host'/name).resolve()
        if not p.is_relative_to((root/'host').resolve()) or file_sha256(p)!=digest:raise ValueError('Host receipt mismatch')
    if not spec.get('source_pins'):raise ValueError('Missing source provenance')
    for pin in spec['source_pins']:
        if file_sha256(pin['path'])!=pin['sha256']:raise ValueError('Stage source changed')
    if spec.get('pilot'):
        p=spec['pilot'];ps=read(Path(p['run_dir'])/'spec.json');verify_case(p['run_dir'],ps)
        if file_sha256(Path(p['run_dir'])/'accepted.json')!=p['sha256'] or ps['host']!=dict(spec['host'],seed=3416):
            raise ValueError('Unmatched pilot')
    return base,root


def verify_case(root,spec):
    root=Path(root);r=read(root/'accepted.json')
    if (r.get('schema')!=VERSION or r.get('identity')!=stable_hash(spec) or r.get('state')!='accepted'
        or r.get('test_access') is not False or not r.get('full_development_mhd_replay')
        or not r.get('host_frozen') or not r.get('reload_exact')):raise ValueError('Terminal stage not accepted')
    if not {'spec.json','development/suite.json','report/paired_statistics.json','costs.json','pca.pt'}.issubset(r['files']):raise ValueError('Stage evidence incomplete')
    for name,sha in r['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or file_sha256(p)!=sha:raise ValueError('Stage evidence changed')
    return r


class Paused(Exception):pass


class TrainProbe(Subset):
    split='train';augment=False


def cached_rows(root):
    m=read(Path(root)/'manifest.json')
    if not m.get('complete'):raise ValueError('Incomplete cache')
    for row in m['shards']:
        p=Path(root)/row['name']
        if file_sha256(p)!=row['sha256']:raise ValueError('Cache changed')
        yield torch.load(p,map_location='cpu',weights_only=False)


def cache(graph,loader,device,out,identity,pause):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);path=out/'manifest.json'
    manifest=read(path) if path.exists() else dict(identity=identity,shards=[],complete=False)
    if manifest['identity']!=identity:raise ValueError('Cache identity changed')
    if manifest['complete']:
        for _ in cached_rows(out):pass
        return
    filler=NormalizedMeanFiller();classifier=graph.get_edge_by_name('fusion_classifier_edge').edge_operations[0].function
    count=0
    with torch.no_grad():
        for i,batch in enumerate(loader):
            if pause():raise Paused()
            count+=len(batch['participant_id'])
            if i<len(manifest['shards']):
                r=manifest['shards'][i];p=out/r['name']
                if file_sha256(p)!=r['sha256']:raise ValueError('Partial cache changed')
                prior=torch.load(p,map_location='cpu',weights_only=False)
                if prior['participant_ids']!=list(batch['participant_id']):raise ValueError('Data order changed')
                continue
            row=dict(participant_ids=list(batch['participant_id']),labels=batch['label'].clone(),features={},logits={})
            for pattern in PATTERNS:
                oct_,cfp=filler.fill(batch['oct'].to(device),batch['cfp'].to(device),pattern)
                logits=forward_with_look(graph,oct_,cfp,counts=batch['counts'])
                feature=read_site(graph,SITE)
                if feature.ndim!=2:raise ValueError('Only participant vectors can use terminal caching')
                head_logits=classifier(feature)
                if not torch.equal(logits,head_logits):raise ValueError('Cached head differs from full MHD')
                row['features'][pattern]=feature.detach().cpu().clone()
                row['logits'][pattern]=logits.detach().cpu().clone()
            p=out/f'{i:06d}.pt';atomic_save(p,row)
            manifest['shards'].append(dict(name=p.name,sha256=file_sha256(p)))
            manifest.update(participants=count,updated_at=time.time());atomic_write_json(manifest,path)
    if len(manifest['shards'])!=len(loader):raise ValueError('Cache size changed')
    manifest.update(complete=True,participants=count);atomic_write_json(manifest,path)


def result_from_cache(root,pattern,classifier,device,artifact=None):
    labels=[];ids=[];logits=[]
    with torch.no_grad():
        for row in cached_rows(root):
            x=row['features'][pattern].to(device)
            if artifact is not None:x=apply_artifact(x,artifact)
            v=classifier(x).detach().cpu().numpy()
            logits.append(v);labels.append(row['labels'].numpy());ids+=row['participant_ids']
    labels=np.concatenate(labels);logits=np.concatenate(logits).astype(np.float64)
    return dict(labels=labels,logits=logits,participant_ids=np.asarray(ids),patterns=np.asarray([pattern]*len(ids)),
        probabilities=probabilities_from_logits(logits),scores=logits[:,1]-logits[:,0],metrics=logit_metrics(labels,logits))


def run(spec,out,device,pause,profile=False):
    import psutil
    base,source=dependencies(spec)
    if device.type!='cuda' or not os.environ.get('SLURM_JOB_ID'):raise ValueError('Slurm GPU required')
    limit=int(os.environ.get('LOOK_WORKER_MEMORY_BYTES','0'))
    if limit<=0:raise ValueError('RAM admission required')
    if not profile:
        receipt=read(os.environ['LOOK_TERMINAL_PROFILE_RECEIPT'])
        if receipt.get('identity')!=stable_hash(spec) or receipt.get('state')!='accepted' or receipt.get('profile') is not True:raise ValueError('Profile missing')
        profile_root=Path(os.environ['LOOK_TERMINAL_PROFILE_RECEIPT']).parent.resolve()
        for name,sha in receipt['files'].items():
            p=(profile_root/name).resolve()
            if not p.is_relative_to(profile_root) or file_sha256(p)!=sha:raise ValueError('Profile evidence changed')
    torch.set_num_threads(min(2,int(os.environ.get('SLURM_CPUS_PER_TASK','2'))));torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=capture_rng();torch.manual_seed(base['seed']);np.random.seed(base['seed']);random.seed(base['seed'])
    props=torch.cuda.get_device_properties(device);total=props.total_memory;budget=min(total,int(os.environ.get('LOOK_TERMINAL_GPU_BUDGET_BYTES',str(total))))
    from mhd_models.scheduling.gpu_budget import configure_allocator
    budget=configure_allocator(device,cap=budget);torch.cuda.reset_peak_memory_stats(device)
    start=time.time();out=Path(out);records=[];costs={};mappings={}
    def status(stage,**extra):
        atomic_write_json(dict(state='running',stage=stage,updated_at=time.time(),test_access=False,**extra),out/'status.json')
    def check():
        if psutil.Process().memory_info().rss>.85*limit:raise MemoryError('RAM reserve breached')
        return pause()
    status('host_strict_loading');_,parents,graph,data=load_original(base,source,device);del parents
    graph.eval()
    for p in graph.parameters():p.requires_grad_(False)
    frozen=cpu_tree(graph.state_dict());node_ids=sorted((n.id,n.name) for n in graph.nodes)
    def loader(dataset):return DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0,
        collate_fn=collate_observed,generator=torch.Generator().manual_seed(base['seed']))
    fit=data['fit'];dev=data['development']
    if profile:fit=TrainProbe(fit,range(min(1024,len(fit))));dev=TrainProbe(data['fit'],range(32))
    train_loader=loader(fit);dev_loader=loader(dev)
    classifier=graph.get_edge_by_name('fusion_classifier_edge').edge_operations[0].function
    def publish(method,pattern,r):
        path=out/'development'/f'{method}__{pattern}.npz';save_prediction_bundle(r,path)
        records.append(dict(method=method,scenario=pattern,path=str(path),sha256=file_sha256(path),metrics=r['metrics']))
    try:
        for role,ld in [('train',train_loader),('development',dev_loader)]:
            status('cache_'+role)
            cache(graph,ld,device,out/'cache'/role,stable_hash(dict(spec=spec,role=role,profile=profile)),check)
        train_cache=out/'cache/train';dev_cache=out/'cache/development'
        # Same numerical IPCA stream and batch partition as the original implementation.
        def features():
            for row in cached_rows(train_cache):
                x=row['features']['complete'];yield x,tuple(x.shape[1:]),tuple(x.shape[1:])
        status('terminal_pca');pca_path=out/'pca.pt'
        if pca_path.exists():pca=FullFeaturePCA.load(pca_path)
        else:
            pca=fit_complete_pca(graph,train_loader,SITE,1,base['look']['max_rank'],device,
                stable_hash(spec)+'/terminal',feature_factory=features);pca.save(pca_path)
        baseline={p:result_from_cache(dev_cache,p,classifier,device) for p in PATTERNS}
        for p,r in baseline.items():publish('host',p,r)
        for pattern in PATTERNS[1:]:
            status('single_final_selection',pattern=pattern)
            def pairs():
                for row in cached_rows(train_cache):
                    f=row['features']['complete'];m=row['features'][pattern]
                    yield f,m,tuple(f.shape[1:]),tuple(m.shape[1:])
            candidates=fit_look_node(graph,train_loader,SITE,pattern,1,base['look']['latent_dims'],
                base['look']['max_rank'],device,pca,feature_pairs=pairs)
            scored=[]
            for q,a in sorted(candidates.items()):
                if check():raise Paused()
                a.save(out/'candidates'/pattern/f'q{q}.pt')
                result=result_from_cache(dev_cache,pattern,classifier,device,a)
                scored.append((q,result['metrics']['macro_f1']))
            q=max(scored,key=lambda x:x[1])[0];a=candidates[q]
            enabled=dict(scored)[q]>baseline[pattern]['metrics']['macro_f1']
            atomic_write_json(dict(candidates=scored,rank=q,lambda_=a.ridge_lambda,enabled=enabled),out/'selection'/f'{pattern}.json')
            single=result_from_cache(dev_cache,pattern,classifier,device,a if enabled else None)
            publish('single_final',pattern,single)
            stats=ResidualMoments.empty(a.std.numel())
            status('residual_statistics',pattern=pattern)
            for row in cached_rows(train_cache):
                if check():raise Paused()
                x=row['features'][pattern];f=row['features']['complete']
                stats.update((x-a.mean)/a.std,(f-x)/a.std)
            for arm in ARMS:
                status('fit_'+arm,pattern=pattern)
                path=out/'mappings'/pattern/f'{arm}.pt'
                if path.exists():artifact=LinearVectorArtifact.from_record(torch.load(path,weights_only=False))
                else:
                    kw={'basis':a.components} if arm==ARMS[0] else {'intercept_basis':a.components} if arm==ARMS[1] else {}
                    artifact=LinearVectorArtifact(a,solve(stats,q,a.ridge_lambda,**kw));atomic_save(path,artifact.record())
                reloaded=LinearVectorArtifact.from_record(torch.load(path,weights_only=False))
                predicted=result_from_cache(dev_cache,pattern,classifier,device,artifact)
                reloaded_result=result_from_cache(dev_cache,pattern,classifier,device,reloaded)
                if not np.array_equal(predicted['logits'],reloaded_result['logits']):raise ValueError('Reload changed output')
                if arm==ARMS[0]:
                    original=result_from_cache(dev_cache,pattern,classifier,device,a)
                    if not np.allclose(predicted['logits'],original['logits'],rtol=1e-4,atol=1e-4) or not np.array_equal(predicted['logits'].argmax(1),original['logits'].argmax(1)):
                        raise ValueError('PCA vector map differs from original single-final candidate')
                # Every dev participant: cached predictions must replay through the full MHD graph.
                status('mhd_replay_'+arm,pattern=pattern)
                replay=evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern,artifact_banks={pattern:[reloaded]})
                check_matched(predicted,replay)
                if not np.array_equal(predicted['logits'],replay['logits']):raise ValueError('Cache differs from full MHD replay')
                mappings[(arm,pattern)]=predicted;publish(arm,pattern,replay)
                costs[f'{arm}/{pattern}']=artifact.mapping.diagnostics
            if check():raise Paused()
        for arm in ARMS:
            publish(arm,'complete',baseline['complete'])
            for ratio in base['look']['ratios']:
                publish(arm,f'mixed_{ratio:.1f}',assemble_mixed(baseline['complete'],{p:mappings[(arm,p)] for p in PATTERNS[1:]},ratio,base['look']['mask_seed']))
        state_equal(graph,frozen)
        if node_ids!=sorted((n.id,n.name) for n in graph.nodes):raise ValueError('Node identity changed')
        atomic_write_json(dict(records=records,test_access=False,scope=protocol()['scope']),out/'development/suite.json')
        status('paired_statistics')
        if not profile:report(records,out/'report')
        costs.update(seconds=time.time()-start,gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,allocation=os.environ['SLURM_JOB_ID'],
            device=props.name,full_network_training=False,test_access=False)
        if costs['gpu_peak_reserved_bytes']>budget:raise MemoryError('GPU peak admission failed')
        if costs['host_peak_rss_bytes']>.85*limit:raise MemoryError('RAM peak admission failed')
        atomic_write_json(costs,out/'costs.json')
    finally:
        state_equal(graph,frozen);restore_rng(rng)
    receipt=dict(schema=VERSION,identity=stable_hash(spec),state='accepted',profile=profile,
        full_development_mhd_replay=not profile,host_frozen=True,reload_exact=True,test_access=False,files=evidence_files(out))
    atomic_write_json(receipt,out/('profile_accepted.json' if profile else 'accepted.json'))
    return receipt


def execute(spec,out,profile=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);stop=False
    def request(*args):
        nonlocal stop
        stop=True
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,request)
    with (out/'run.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not profile and (out/'accepted.json').exists():return verify_case(out,spec)
        if (out/'spec.json').exists() and read(out/'spec.json')!=spec:raise ValueError('Run identity changed')
        atomic_write_json(spec,out/'spec.json')
        try:
            r=run(spec,out,torch.device('cuda:0'),lambda:stop or (out/'pause.json').exists(),profile)
            atomic_write_json(dict(state='completed',test_access=False,profile=profile,updated_at=time.time()),out/'status.json');return r
        except Paused:
            atomic_write_json(dict(state='paused',test_access=False,updated_at=time.time()),out/'status.json');return {'state':'paused'}
        except Exception as e:
            import traceback
            atomic_write_json(dict(state='needs_review',error=repr(e),traceback=traceback.format_exc(),updated_at=time.time()),out/'status.json');raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);p.add_argument('--profile',action='store_true')
    a=p.parse_args();r=execute(read(a.spec),a.output,a.profile)
    if r['state']=='paused':raise SystemExit(75)

if __name__=='__main__':main()
