"""Independent full-MHD family tree with sealed test and measured admission."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import resource
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from look.runtime.state import stable_hash, file_sha256, atomic_write_json
from look.runtime.host_checkpoint import capture_rng, restore_rng, cpu_tree
from look.studies.project_case import read, evidence_files, CheckedLoader, Paused
from look.studies.mechanism_case import load_original
from look.studies.family_search_protocol import VERSION, PATTERNS, validate
from look.studies.search_case import dependencies as reference_dependencies, load_basis, check_files
from look.methods.affine_family import FamilyArtifact
from look.methods.linear_operator import fingerprint
from look.methods.family_greedy import fit_family_trajectory
from look.methods.operator import forward_with_look
from look.methods.joint import correction_sites, read_site

from look.methods.independent_greedy import SelectionPaused
from look.data.observed_pair import collate_observed
from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle
from look.evaluation.observed_suite import check_matched
from look.training.mechanism_training import state_equal
from look.studies.spatial_case import rng_equal


def load_bank(root):
    saved=torch.load(Path(root)/'bank.pt',map_location='cpu',weights_only=False)
    if fingerprint(saved['bank'])!=saved['sha256']:raise ValueError('Changed family bank')
    return [FamilyArtifact.from_record(r) for r in saved['bank']]


def verify_case(root,spec):
    validate(spec);root=Path(root);r=read(root/'accepted.json')
    if (r.get('schema')!=VERSION or r.get('identity')!=stable_hash(spec) or r.get('state')!='accepted'
        or r.get('test_access') is not False or r.get('profile') is not False
        or not all(r.get(k) is True for k in ('host_frozen','reload_exact','rng_restored','upstream_refit_verified','pca_unchanged'))):
        raise ValueError('Family search has not passed technical acceptance')
    required={'spec.json','development/suite.json','costs.json'}
    for pattern in PATTERNS:
        required.update('corrections/'+pattern+'/'+n for n in ('bank.pt','replay.json','selection.json'))
    if not required.issubset(r['files']):raise ValueError('Missing family search evidence')
    check_files(root,r['files']);return r


def dependencies(spec):
    validate(spec)
    for pin in spec['source_pins']:
        if file_sha256(pin['path'])!=pin['sha256']:raise ValueError('Family source changed')
    return reference_dependencies(spec['reference_search'])


class TrainProbe(Subset):
    split='train';augment=False

class DevProbe(TrainProbe):
    split='train_probe'


def run(spec,out,device,pause,profile=False):
    import psutil
    base,source,manifest=dependencies(spec);ref=spec['reference_search'];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if device.type!='cuda' or not os.environ.get('SLURM_JOB_ID'):raise ValueError('Admitted Slurm GPU required')
    limit=int(os.environ.get('LOOK_WORKER_MEMORY_BYTES','0'))
    if limit<=0:raise ValueError('Explicit worker RAM admission required')
    profile_receipt=None
    if not profile:
        p=Path(os.environ['LOOK_FAMILY_SEARCH_PROFILE_RECEIPT']);r=read(p)
        if r.get('identity')!=stable_hash(spec) or r.get('state')!='accepted' or r.get('profile') is not True:
            raise ValueError('Search resource receipt mismatch')
        check_files(p.parent,r['files']);profile_receipt=r
    props=torch.cuda.get_device_properties(device);budget=min(.875*props.total_memory,props.total_memory-10*1024**3)
    hardware=dict(name=props.name,total_memory=props.total_memory,major=props.major,minor=props.minor,
        worker_threads=ref.get('worker_threads',2),worker_memory_bytes=limit)
    requested_budget=int(os.environ.get('LOOK_FAMILY_SEARCH_GPU_BUDGET_BYTES',str(int(budget))))
    if requested_budget<=2*1024**3:raise MemoryError('GPU fitting budget must exceed reserve overhead')
    budget=min(budget,requested_budget)
    hardware['process_gpu_budget_bytes']=int(budget)
    if profile_receipt is not None and profile_receipt.get('hardware')!=hardware:
        raise ValueError('Family resource receipt process GPU budget mismatch')
    free,total=torch.cuda.mem_get_info(device)
    if free<10*1024**3:raise MemoryError('Whole-device reserve absent')
    torch.cuda.set_per_process_memory_fraction((budget-2*1024**3)/1.2/total,device);torch.cuda.reset_peak_memory_stats(device)
    rng=capture_rng();started=time.time();graph=None;frozen=None
    session=out/'sessions'/f'{time.time_ns()}.json'
    atomic_write_json(dict(started_at=started,state='running',profile=profile),session)
    torch.set_num_threads(ref.get('worker_threads',2));torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        _,parents,graph,data=load_original(base,source,device);del parents
        graph.eval()
        for p in graph.parameters():p.requires_grad_(False)
        frozen=cpu_tree(graph.state_dict());nodes=[(n.id,n.name) for n in sorted(graph.nodes,key=lambda n:n.id)]
        if correction_sites(graph)!=ref['candidate_sites']:raise ValueError('Actual MHD sites differ')
        bank=load_basis(ref,manifest)
        if spec['workspace_bytes']>.7*limit:raise MemoryError('Fitting workspace exceeds admitted worker reserve')
        def check():
            if psutil.Process().memory_info().rss>.85*limit:raise MemoryError('Host RAM reserve breached')
            if torch.cuda.mem_get_info(device)[0]<10*1024**3:raise MemoryError('Whole-device GPU reserve breached')
            return pause()
        def loader(ds):
            return CheckedLoader(DataLoader(ds,batch_size=base['training']['microbatch'],shuffle=False,num_workers=0,
                collate_fn=collate_observed,generator=torch.Generator().manual_seed(base['seed'])),check)
        fit=data['fit'];dev=data['development']
        if profile:fit=TrainProbe(fit,range(min(512,len(fit))));dev=DevProbe(data['fit'],range(min(32,len(data['fit']))))
        train_loader,dev_loader=loader(fit),loader(dev)
        first=next(iter(train_loader))
        with torch.no_grad():forward_with_look(graph,first['oct'].to(device),first['cfp'].to(device),counts=first['counts'])
        for site in ref['eligible_sites']:
            factors=[1] if read_site(graph,site).ndim==2 else [16]
            if any((site,f) not in bank for f in factors):raise ValueError('Missing required PCA factor')
        records=[]
        for pattern in PATTERNS:
            target=out/'corrections'/pattern
            if check():raise Paused()
            atomic_write_json(dict(state='running',stage='search_fit',pattern=pattern,mode=ref['mode'],updated_at=time.time()),out/'status.json')
            # The adapter replays and verifies an already completed trajectory;
            # never trust an orphan bank.pt as a completion marker.
            artifacts,_=fit_family_trajectory(graph,train_loader,dev_loader,
                arm=spec['arm'],pattern=pattern,sites=ref['eligible_sites'],factor=16,
                candidates=spec['candidates'],pca_bank=bank,identity=stable_hash(spec),
                output=target,device=device,workspace_bytes=spec['workspace_bytes'],
                mode=ref['mode'],should_pause=check,penalty_policy=spec['penalty_policy'])
            if any(a.node_name not in ref['eligible_sites'] for a in artifacts):raise ValueError('Disabled prefix was corrected')
            corrected=evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern,artifact_banks={pattern:artifacts})
            loaded=load_bank(target)
            replay=evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern,artifact_banks={pattern:loaded})
            check_matched(corrected,replay)
            if not np.array_equal(corrected['logits'],replay['logits']):raise ValueError('Reload predictions differ')
            baseline=evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern);check_matched(corrected,baseline)
            for name,result in [('search',corrected),('host',baseline)]:
                p=out/'development'/f'{name}_{pattern}.npz';save_prediction_bundle(result,p)
                records.append(dict(method=name,scenario=pattern,path=str(p),sha256=file_sha256(p),metrics=result['metrics']))
            state_equal(graph,frozen)
        if nodes!=[(n.id,n.name) for n in sorted(graph.nodes,key=lambda n:n.id)]:raise ValueError('Node IDs changed')
        if file_sha256(ref['pca']['path'])!=ref['pca']['sha256']:raise ValueError('PCA changed during fit')
        atomic_write_json(dict(records=records,test_access=False),out/'development/suite.json')
        prior_sessions=[read(p) for p in (out/'sessions').glob('*.json') if p!=session]
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if peak_rss>.85*limit:raise MemoryError('Peak host RAM exceeds admitted worker reserve')
        costs=dict(seconds=time.time()-started,peak_rss_bytes=int(peak_rss),
            known_total_seconds=sum(p.get('elapsed_seconds',0) for p in prior_sessions)+time.time()-started,
            incomplete_previous_sessions=sum('elapsed_seconds' not in p for p in prior_sessions),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            rss_bytes=psutil.Process().memory_info().rss,profile=profile,recovery='family_statistics_and_prefix_decisions')
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
    files=evidence_files(out);files.pop('profile_accepted.json',None);files.pop('accepted.json',None)
    receipt=dict(schema=VERSION,identity=stable_hash(spec),state='accepted',profile=profile,test_access=False,
        hardware=hardware,profile_coverage=('512_train_fit_32_train_eval_frozen_host_and_serialization_replay;not_full_cohort_peak_or_interruption_resume' if profile else 'full_train_fit_full_dev_evaluation_serialization_replay'),host_frozen=True,reload_exact=True,rng_restored=True,upstream_refit_verified=True,pca_unchanged=True,files=files)
    atomic_write_json(receipt,out/('profile_accepted.json' if profile else 'accepted.json'));return receipt


def execute(spec,output,profile=False):
    validate(spec)
    out=Path(output);out.mkdir(parents=True,exist_ok=True);stop=False
    if profile and (out/'accepted.json').exists():raise ValueError('Profile requires separate output from accepted formal case')
    if not profile and (out/'profile_accepted.json').exists():raise ValueError('Formal run cannot reuse train-only profile directory')
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
