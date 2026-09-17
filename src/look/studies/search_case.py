"""One matched search trajectory with frozen host and original complete-train PCA."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from look.runtime.state import stable_hash, file_sha256, atomic_write_json
from look.runtime.host_checkpoint import capture_rng, restore_rng, cpu_tree, atomic_save
from look.studies.project_case import read, evidence_files, validate_spec, CheckedLoader, Paused
from look.studies.mechanism_case import load_original
from look.studies.search_protocol import VERSION, PATTERNS, validate, execution_factors, execution_latent_dims
from look.methods.operator import FullFeaturePCA, load_selected_bank, forward_with_look
from look.methods.joint import correction_sites, read_site
from look.methods.search_policy import fit_search
from look.methods.independent_greedy import SelectionPaused
from look.data.observed_pair import collate_observed
from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle
from look.evaluation.observed_suite import check_matched
from look.training.mechanism_training import state_equal
from look.studies.spatial_case import rng_equal


def check_files(root, files):
    root=Path(root).resolve()
    for name,sha in files.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or file_sha256(p)!=sha:raise ValueError('Search evidence changed')


def verify_case(root,spec):
    validate(spec);root=Path(root);r=read(root/'accepted.json')
    if (r.get('schema')!=VERSION or r.get('identity')!=stable_hash(spec) or r.get('state')!='accepted'
        or r.get('test_access') is not False or r.get('profile') is not False
        or not all(r.get(k) is True for k in ('host_frozen','reload_exact','rng_restored','upstream_refit_verified','pca_unchanged'))):
        raise ValueError('Search has not passed technical acceptance')
    required={'spec.json','development/suite.json','costs.json'}
    required.update('corrections/'+p+'/factor_selection.json' for p in PATTERNS)
    if not required.issubset(r['files']):raise ValueError('Missing search evidence')
    check_files(root,r['files']);return r


def dependencies(spec):
    validate(spec);s=spec['source'];root=Path(s['run_dir']);base=read(s['spec_path']);validate_spec(base)
    for p,sha in ((s['spec_path'],s['spec_sha256']),(root/'host/accepted.json',s['host_accepted_sha256']),
                  (root/'host/best.pt',s['best_sha256']),(spec['pca']['path'],spec['pca']['sha256'])):
        if file_sha256(p)!=sha:raise ValueError('Search dependency changed')
    receipt=read(root/'host/accepted.json')
    if receipt.get('state')!='accepted' or receipt.get('identity')!=stable_hash(base):raise ValueError('Host not accepted')
    check_files(root/'host',receipt['files'])
    h=spec['host']
    if (base['disease'],base['model']['name'],base['position'],base['seed'])!=tuple(h[k] for k in ('disease','architecture','position','seed')):
        raise ValueError('Different host role')
    m=read(spec['pca']['path']);identity=m['identity']
    if identity.get('case')!=stable_hash(base) or identity.get('host_best_sha256')!=s['best_sha256']:
        raise ValueError('PCA belongs to another host')
    from look.methods.joint import PROTOCOL
    if identity.get('max_rank')!=base['look']['max_rank'] or identity.get('protocol')!=PROTOCOL:
        raise ValueError('PCA fitting protocol mismatch')
    # Verify the actual producer, not a field absent in the original locked bank.
    for key,relative in [('joint_code_sha256','methods/joint.py'),('pca_code_sha256','methods/operator.py')]:
        pins=[p['sha256'] for p in base['source_pins'] if p['path'].endswith('/look/'+relative)]
        if pins!=[identity.get(key)]:raise ValueError('PCA producer does not match locked host source')
    if not spec.get('source_pins'):raise ValueError('Missing immutable source')
    for pin in spec['source_pins']:
        if file_sha256(pin['path'])!=pin['sha256']:raise ValueError('Search source changed')
    if spec.get('pilot'):
        pilot=spec['pilot']
        if len(pilot)!=2:raise ValueError('Both first-seed search policies required')
        modes=set()
        for item in pilot:
            p=Path(item['path'])
            if file_sha256(p/'accepted.json')!=item['sha256']:raise ValueError('Pilot changed')
            ps=read(p/'spec.json');verify_case(p,ps)
            if ps['host']!=dict(h,seed=3416):raise ValueError('Different pilot host')
            modes.add(ps['mode'])
        if modes!={'greedy','best_forward'}:raise ValueError('Incomplete matched pilot')
    return base,root,m


def load_basis(spec,manifest):
    allowed=set(spec['eligible_sites']);root=Path(spec['pca']['path']).parent.resolve();bank={}
    for e in manifest['entries']:
        if e['node'] not in allowed:continue
        if 'spatial_factors' in spec and e['factor'] not in [1]+spec['spatial_factors']:continue
        p=Path(e['path']).resolve();key=(e['node'],e['factor'])
        if key in bank or not p.is_relative_to(root) or file_sha256(p)!=e['sha256']:raise ValueError('Changed PCA entry')
        b=FullFeaturePCA.load(p)
        if (b.node_name,b.factor)!=key or b.source_id!=e['source_id'] or b.spatial_method!='interpolate':raise ValueError('PCA entry identity')
        if not all(torch.isfinite(t).all() for t in (b.mean,b.std,b.pca_mean,b.components,b.explained_variance_ratio)):
            raise ValueError('Nonfinite PCA')
        bank[key]=b
    if {k[0] for k in bank}!=allowed:raise ValueError('Incomplete search PCA entries')
    return bank


class TrainProbe(Subset):
    split='train';augment=False


def run(spec,out,device,pause,profile=False):
    import psutil
    base,source,manifest=dependencies(spec);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if device.type!='cuda' or not os.environ.get('SLURM_JOB_ID'):raise ValueError('Admitted Slurm GPU required')
    limit=int(os.environ.get('LOOK_WORKER_MEMORY_BYTES','0'))
    if limit<=0:raise ValueError('Explicit worker RAM admission required')
    if not profile:
        p=Path(os.environ['LOOK_SEARCH_PROFILE_RECEIPT']);r=read(p)
        if r.get('identity')!=stable_hash(spec) or r.get('state')!='accepted' or r.get('profile') is not True:
            raise ValueError('Search resource receipt mismatch')
        check_files(p.parent,r['files'])
    props=torch.cuda.get_device_properties(device);budget=min(.875*props.total_memory,props.total_memory-10*1024**3)
    free,total=torch.cuda.mem_get_info(device)
    if free<10*1024**3:raise MemoryError('Whole-device reserve absent')
    torch.cuda.set_per_process_memory_fraction((budget-2*1024**3)/1.2/total,device);torch.cuda.reset_peak_memory_stats(device)
    rng=capture_rng();started=time.time();graph=None;frozen=None
    session=out/'sessions'/f'{time.time_ns()}.json'
    atomic_write_json(dict(started_at=started,state='running',profile=profile),session)
    torch.set_num_threads(spec.get('worker_threads',2));torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        _,parents,graph,data=load_original(base,source,device);del parents
        graph.eval()
        for p in graph.parameters():p.requires_grad_(False)
        frozen=cpu_tree(graph.state_dict());nodes=[(n.id,n.name) for n in sorted(graph.nodes,key=lambda n:n.id)]
        if correction_sites(graph)!=spec['candidate_sites']:raise ValueError('Actual MHD sites differ')
        bank=load_basis(spec,manifest);cfg=dict(base['look'],factors=execution_factors(spec,base['look']),
            latent_dims=execution_latent_dims(spec,base['look']))
        def check():
            if psutil.Process().memory_info().rss>.85*limit:raise MemoryError('Host RAM reserve breached')
            if torch.cuda.mem_get_info(device)[0]<10*1024**3:raise MemoryError('Whole-device GPU reserve breached')
            return pause()
        def loader(ds):
            return CheckedLoader(DataLoader(ds,batch_size=base['training']['microbatch'],shuffle=False,num_workers=0,
                collate_fn=collate_observed,generator=torch.Generator().manual_seed(base['seed'])),check)
        fit=data['fit'];dev=data['development']
        if profile:fit=TrainProbe(fit,range(min(512,len(fit))));dev=TrainProbe(data['fit'],range(min(32,len(data['fit']))))
        train_loader,dev_loader=loader(fit),loader(dev)
        first=next(iter(train_loader))
        with torch.no_grad():forward_with_look(graph,first['oct'].to(device),first['cfp'].to(device),counts=first['counts'])
        for site in spec['eligible_sites']:
            factors=[1] if read_site(graph,site).ndim==2 else cfg['factors']
            if any((site,f) not in bank for f in factors):raise ValueError('Missing required PCA factor')
        records=[]
        for pattern in PATTERNS:
            target=out/'corrections'/pattern
            if check():raise Paused()
            atomic_write_json(dict(state='running',stage='search_fit',pattern=pattern,mode=spec['mode'],updated_at=time.time()),out/'status.json')
            if (target/'factor_selection.json').exists():artifacts=load_selected_bank(target)
            else:
                artifacts,_=fit_search(graph,train_loader,dev_loader,pattern,spec['eligible_sites'],
                    cfg['factors'],cfg['latent_dims'],cfg['max_rank'],device,target,bank,
                    identity=stable_hash(spec),mode=spec['mode'],should_pause=check,
                    cache_identity=dict(host=spec['source']['best_sha256'],pca=spec['pca']['sha256'],
                        parent_spec=spec['source']['spec_sha256'],batch=base['training']['microbatch'],
                        train_count=len(fit),profile=profile),
                    reference_root=out.parent.parent/'shared_latents')
            if any(a.node_name not in spec['eligible_sites'] for a in artifacts):raise ValueError('Disabled prefix was corrected')
            corrected=evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern,artifact_banks={pattern:artifacts})
            loaded=load_selected_bank(target)
            replay=evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern,artifact_banks={pattern:loaded})
            check_matched(corrected,replay)
            if not np.array_equal(corrected['logits'],replay['logits']):raise ValueError('Reload predictions differ')
            baseline=evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern);check_matched(corrected,baseline)
            for name,result in [('search',corrected),('host',baseline)]:
                p=out/'development'/f'{name}_{pattern}.npz';save_prediction_bundle(result,p)
                records.append(dict(method=name,scenario=pattern,path=str(p),sha256=file_sha256(p),metrics=result['metrics']))
            state_equal(graph,frozen)
        if nodes!=[(n.id,n.name) for n in sorted(graph.nodes,key=lambda n:n.id)]:raise ValueError('Node IDs changed')
        if file_sha256(spec['pca']['path'])!=spec['pca']['sha256']:raise ValueError('PCA changed during fit')
        atomic_write_json(dict(records=records,test_access=False),out/'development/suite.json')
        prior_sessions=[read(p) for p in (out/'sessions').glob('*.json') if p!=session]
        costs=dict(seconds=time.time()-started,
            known_total_seconds=sum(p.get('elapsed_seconds',0) for p in prior_sessions)+time.time()-started,
            incomplete_previous_sessions=sum('elapsed_seconds' not in p for p in prior_sessions),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            rss_bytes=psutil.Process().memory_info().rss,profile=profile,recovery='node_decision_not_batch_exact')
        if costs['gpu_peak_reserved_bytes']*1.2+2*1024**3>budget:raise MemoryError('Conservative GPU peak exceeds budget')
        atomic_write_json(costs,out/'costs.json')
    finally:
        try:
            if graph is not None and frozen is not None:state_equal(graph,frozen)
        finally:
            restore_rng(rng)
            atomic_write_json(dict(started_at=started,ended_at=time.time(),elapsed_seconds=time.time()-started,
                state='exited',profile=profile),session)
    if not rng_equal(rng,capture_rng()):raise ValueError('RNG restore failed')
    files=evidence_files(out);files.pop('profile_accepted.json',None)
    receipt=dict(schema=VERSION,identity=stable_hash(spec),state='accepted',profile=profile,test_access=False,
        host_frozen=True,reload_exact=True,rng_restored=True,upstream_refit_verified=True,pca_unchanged=True,files=files)
    atomic_write_json(receipt,out/('profile_accepted.json' if profile else 'accepted.json'));return receipt


def execute(spec,output,profile=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);stop=False
    def request(*args):
        nonlocal stop
        stop=True
    for s in (signal.SIGTERM,signal.SIGUSR1):signal.signal(s,request)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'accepted.json').exists() and not profile:return verify_case(out,spec)
        if (out/'spec.json').exists() and read(out/'spec.json')!=spec:raise ValueError('Resume identity changed')
        atomic_write_json(spec,out/'spec.json')
        try:
            r=run(spec,out,torch.device('cuda:0'),lambda:stop or (out/'pause.json').exists(),profile)
            atomic_write_json(dict(state='completed',profile=profile,test_access=False,updated_at=time.time()),out/'status.json');return r
        except (Paused,SelectionPaused):
            atomic_write_json(dict(state='paused',test_access=False,updated_at=time.time()),out/'status.json');return {'state':'paused'}
        except Exception as e:
            import traceback
            atomic_write_json(dict(state='needs_review',error=repr(e),traceback=traceback.format_exc(),updated_at=time.time()),out/'status.json');raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);p.add_argument('--profile',action='store_true');a=p.parse_args()
    r=execute(read(a.spec),a.output,a.profile)
    if r['state']=='paused':raise SystemExit(75)

if __name__=='__main__':main()
