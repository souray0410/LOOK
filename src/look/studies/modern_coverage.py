"""Independent full-coverage successor of an accepted modern EmbraceNet host.

Reads the original frozen host and accepted PCA; never resumes its writer or
changes its spec. Results remain development-search evidence, not sealed test.
"""
import argparse, csv, fcntl, json, os, signal, time
from pathlib import Path
import numpy as np
import torch
from look.runtime.state import file_sha256, stable_hash, atomic_write_json
from look.runtime.host_checkpoint import cpu_tree
from look.studies.embracenet_delivery import (
    read, _load_selected, make_dataset, _loader, _simultaneous_contrasts,
    SelectionPaused, EvaluationPaused, LoaderPaused,
)
from look.studies.modern_embracenet import validate, SITES
from look.methods.operator import FullFeaturePCA
from look.methods.embracenet_family import fit_embracenet_search_suite
from look.evaluation.embracenet import evaluate_single_missing
from look.evaluation.evaluator import save_prediction_bundle
from look.training.mechanism_training import state_equal

ARMS=('shared_pca_ridge','pca_free_mean','rrr_shared_intercept','residual_rrr')
PATTERNS=('oct_missing','cfp_missing')

def accepted_pca(parent, spec):
    receipt=read(parent/'pca/accepted.json')
    if (receipt.get('state')!='accepted' or receipt.get('identity')!=stable_hash(spec)
        or receipt.get('host_best_sha256')!=file_sha256(parent/'host/best.pt')):
        raise ValueError('PCA identity does not match the accepted selected host')
    bank={}
    for entry in receipt['entries']:
        if file_sha256(entry['path'])!=entry['sha256']: raise ValueError('PCA artifact changed')
        basis=FullFeaturePCA.load(entry['path'])
        key=(entry['node'],entry['factor'])
        if (key in bank or basis.node_name!=entry['node'] or basis.factor!=entry['factor']
            or basis.source_id!=entry['source_id'] or basis.sample_count!=spec['cohort']['train']):
            raise ValueError('PCA metadata or training coverage changed')
        bank[key]=basis
    if len(bank)!=len(SITES) or {k[0] for k in bank}!=set(SITES): raise ValueError('Incomplete PCA coverage')
    return bank

def verify_coverage(records, sites):
    expected={(a,p,s,(n,)) for a in ARMS for p in PATTERNS for s in ('single_site',) for n in sites}
    expected|={(a,p,s,tuple(sites)) for a in ARMS for p in PATTERNS for s in ('best_forward','positive_forward_tree')}
    actual=[(r['arm'],r['pattern'],r['search'],tuple(r['sites'])) for r in records]
    if len(actual)!=len(expected) or set(actual)!=expected: raise ValueError('Missing or repeated comparison')
    for row in records:
        if file_sha256(Path(row['output'])/'selection.json')!=row['selection_sha256']:
            raise ValueError('Selection evidence changed')
        if file_sha256(row['final']['prediction'])!=row['final']['sha256']:
            raise ValueError('Prediction evidence changed')

def prediction(path, ids=None, labels=None):
    with np.load(path,allow_pickle=False) as z:
        out={k:z[k].copy() for k in ('participant_ids','labels','logits')}
    if ids is not None and (not np.array_equal(ids,out['participant_ids']) or not np.array_equal(labels,out['labels'])):
        raise ValueError('Development participants/order/labels differ')
    return out

def report(root, records, baselines, spec, identity):
    verify_coverage(records,SITES)
    first=prediction(baselines[PATTERNS[0]]['path'])
    if len(first['labels'])!=spec['cohort']['development']: raise ValueError('Incomplete development cohort')
    logits={};definitions=[]
    for pattern, row in baselines.items():
        if file_sha256(row['path'])!=row['sha256']: raise ValueError('Baseline changed')
        logits['A_'+pattern]=prediction(row['path'],first['participant_ids'],first['labels'])['logits']
    # Keep all registered searches in one simultaneous contrast family.
    # This does not correct the separate within-development selection bias.
    for row in records:
        values=prediction(row['final']['prediction'],first['participant_ids'],first['labels'])
        key='comparison_'+str(len(definitions));logits[key]=values['logits']
        definitions.append(dict(method_key=key,reference_key='A_'+row['pattern'],
                                family=row['arm'],search=row['search'],sites=row['sites']))
    intervals=_simultaneous_contrasts(first['labels'],logits,definitions,
                                    spec['bootstrap']['iterations'],spec['bootstrap']['seed'])
    for value in intervals.values():
        value['family']='all_prespecified_coverage_same_A_comparisons'
        value['limitation']='same development data searched; intervals do not remove selection bias'
    rows=[]
    for row in records:
        selection=read(Path(row['output'])/'selection.json')
        cost=Path(row['output'])/'feature_costs.json'
        path=selection.get('selected_path')
        if path is None:
            path=[d['winner']['node'] for d in selection['decisions'] if d['enabled']]
        rows.append({**row,'selected_path':path,
                     'cost':read(cost) if cost.exists() else {'unavailable':True}})
    result=dict(identity=identity,comparisons=rows,baselines=baselines,intervals=intervals,
                test_access=False,scientific_acceptance=False,
                limitations=['single seed 3416','development selection and evaluation overlap',
                             'seeds3417/3418 remain required regardless of result'])
    atomic_write_json(result,root/'results.json')
    with (root/'results.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['family','missing','search','sites','macro_f1','macro_auroc_ovr'])
        for r in rows:
            m=r['final']['metrics'];writer.writerow([r['arm'],r['pattern'],r['search'],';'.join(r['sites']),m['macro_f1'],m['macro_auroc_ovr']])
    (root/'method.md').write_text('# Frozen modern EmbraceNet with LOOK\n\n```mermaid\nflowchart LR\nCFP[CFP ConvNeXt-B] --> A[Same frozen EmbraceNet A]\nOCT[OCT B-scan ConvNeXt-B] --> A\nA --> B[Uncorrected missing-modality predictions]\nA --> C[Train-only PCA and four correction families]\nC --> D[Independent single-site / best-forward / forward-tree]\nD --> E[Full development predictions and matched costs]\nB --> E\n```\n\n88 search trajectories, each contrasted with the same A, with simultaneous intervals across the registered coverage. Development-search estimates are not test confirmation. All registered repeats remain required.\n')
    files={n:file_sha256(root/n) for n in ('results.json','results.csv','method.md','suite/coverage.json')}
    atomic_write_json(dict(state='self_checked_pending_independent_review',identity=identity,
                           files=files,scientific_acceptance=False,test_access=False),root/'accepted.json')

def run(contract_path):
    contract=read(contract_path);parent=Path(contract['parent']);root=Path(contract['output'])
    root.mkdir(parents=True,exist_ok=True)
    if file_sha256(parent/'spec.json')!=contract['parent_spec_sha256']: raise ValueError('Parent spec changed')
    for row in contract['source_pins']:
        if file_sha256(row['path'])!=row['sha256']: raise ValueError('Successor source changed')
    spec=read(parent/'spec.json');validate(spec)
    identity=dict(contract_sha256=file_sha256(contract_path),host_sha256=file_sha256(parent/'host/best.pt'),
                  pca_receipt_sha256=file_sha256(parent/'pca/accepted.json'))
    paused=[False]
    def request_pause(*_): paused[0]=True
    for sig in (signal.SIGUSR1,signal.SIGTERM): signal.signal(sig,request_pause)
    end=float(os.environ['MHD_EXECUTION_LEASE_END'])
    def check(): return paused[0] or time.time()>=end-900
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'accepted.json').exists():
            receipt=read(root/'accepted.json')
            if receipt['identity']!=identity: raise ValueError('Completed successor identity changed')
            for name,h in receipt['files'].items():
                if file_sha256(root/name)!=h: raise ValueError('Completed successor report changed')
            return 0
        try:
            graph,_=_load_selected(spec,parent,torch.device('cuda:0'));frozen=cpu_tree(graph.state_dict())
            bank=accepted_pca(parent,spec)
            train=_loader(make_dataset(spec,'train'),spec,check);dev=_loader(make_dataset(spec,'development'),spec,check)
            atomic_write_json(dict(state='full_coverage_fitting',identity=identity,time=time.time()),root/'status.json')
            rows=fit_embracenet_search_suite(graph,train,dev,sites=SITES,
                candidates_by_arm={a:[{'rank':spec['look']['rank'],'ridge_lambda':None}] for a in ARMS},
                output=root/'suite',identity=identity,factor=spec['look']['factor'],pca_bank=bank,
                device=torch.device('cuda:0'),workspace_bytes=spec['workspace_bytes'],should_pause=check,
                penalty_policy=spec['look']['penalty_policy'])
            baselines={}
            for pattern in PATTERNS:
                value=evaluate_single_missing(graph,dev,torch.device('cuda:0'),pattern)
                path=root/(pattern+'_baseline.npz');save_prediction_bundle(value,path)
                baselines[pattern]=dict(path=str(path),sha256=file_sha256(path),metrics=value['metrics'])
            state_equal(graph,frozen)
            atomic_write_json(dict(identity=identity,baselines=baselines,frozen_host_exact=True,
                coverage_sha256=file_sha256(root/'suite/coverage.json'),test_access=False),root/'fitting_accepted.json')
            # The expensive bootstrap/report is CPU work after GPU release.
            atomic_write_json(dict(state='completed_fitting',identity=identity,time=time.time()),root/'status.json')
            return 0
        except (SelectionPaused,EvaluationPaused,LoaderPaused):
            atomic_write_json(dict(state='paused',identity=identity,time=time.time()),root/'status.json')
            return 75

def report_only(contract_path):
    contract=read(contract_path);parent=Path(contract['parent']);root=Path(contract['output'])
    if file_sha256(parent/'spec.json')!=contract['parent_spec_sha256']: raise ValueError('Parent changed')
    for row in contract['source_pins']:
        if file_sha256(row['path'])!=row['sha256']: raise ValueError('Source changed')
    accepted=read(root/'fitting_accepted.json')
    identity=dict(contract_sha256=file_sha256(contract_path),host_sha256=file_sha256(parent/'host/best.pt'),
                  pca_receipt_sha256=file_sha256(parent/'pca/accepted.json'))
    if accepted['identity']!=identity or not accepted['frozen_host_exact']:
        raise ValueError('Fitting acceptance identity changed')
    if file_sha256(root/'suite/coverage.json')!=accepted['coverage_sha256']: raise ValueError('Coverage changed')
    report(root,read(root/'suite/coverage.json')['records'],accepted['baselines'],read(parent/'spec.json'),identity)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--contract',required=True);parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only: report_only(args.contract)
    else: raise SystemExit(run(args.contract))
