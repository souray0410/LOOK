"""Train-only monotone affine output control; no feature or backbone correction."""
from __future__ import annotations
import gc
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from .stable_metrics import logit_metrics, probabilities_from_logits
from .state import atomic_write_json, stable_hash, utc_now
from .start_study import read_json, file_record, verify_file, verify_source, SuffixRunner, PATTERNS
from .method_kernels import evaluate_control
from .filling import NormalizedMeanFiller
from .evaluate import save_prediction_bundle
from .missingness import missingness_plan
from .reproducibility import seed_everything, implementation_sha256


def fit_affine(labels, logits, spec):
    """Convex binary CE + fixed identity-centered L2, fitted only on train.

    Scaling the optimization coordinate improves conditioning without changing
    the penalty or fitted function. Positive slope preserves directional ranks.
    """
    y=np.asarray(labels,dtype=np.int64);z=np.asarray(logits,dtype=np.float64)
    if z.shape!=(len(y),2) or not np.isfinite(z).all() or set(np.unique(y))!={0,1}:
        raise ValueError('Finite paired binary train logits with both classes required')
    s=z[:,1]-z[:,0];scale=max(float(np.std(s)),1.)
    x=s/scale;lam=float(spec['identity_l2']);amin=float(spec['minimum_slope'])
    if lam<=0 or amin<=0:raise ValueError('Positive prespecified penalty and slope required')
    def objective(v):
        u,c=v;a=u/scale;t=x*u+c
        residual=expit(t)-y
        loss=np.mean(np.logaddexp(0,t)-y*t)+.5*lam*((a-1)**2+c*c)
        grad=np.array([np.mean(residual*x)+lam*(a-1)/scale,np.mean(residual)+lam*c])
        return float(loss),grad
    solved=minimize(objective,np.array([scale,0.]),jac=True,method='L-BFGS-B',
        bounds=[(amin*scale,None),(None,None)],
        options={'maxiter':int(spec['max_iterations']),'ftol':1e-13,'gtol':1e-9,'maxls':50})
    if not solved.success or not np.isfinite(solved.x).all():
        raise RuntimeError(f'Affine optimizer failed: {solved.message}')
    return dict(a=float(solved.x[0]/scale),c=float(solved.x[1]),
        train_objective=float(solved.fun),iterations=int(solved.nit),
        train_n=len(y),supervision='train_labels_binary_cross_entropy',spec=spec)


def transform_logits(logits,parameters):
    z=np.asarray(logits,dtype=np.float64)
    a,c=float(parameters['a']),float(parameters['c'])
    if not np.isfinite([a,c]).all() or a<=0:raise ValueError('Finite positive slope required')
    if a==1. and c==0.:return z.copy()
    s=a*(z[:,1]-z[:,0])+c
    out=np.column_stack((-s/2,s/2))
    if not np.isfinite(out).all():raise ValueError('Non-finite affine output')
    return out


def with_logits(bundle,logits):
    z=np.asarray(logits,dtype=np.float64)
    return dict(bundle,logits=z,scores=z[:,1]-z[:,0],
        probabilities=probabilities_from_logits(z),metrics=logit_metrics(bundle['labels'],z))


def freeze_direction(train,validation,spec):
    fitted=fit_affine(train['labels'],train['logits'],spec)
    off=logit_metrics(validation['labels'],validation['logits'])
    candidate=logit_metrics(validation['labels'],transform_logits(validation['logits'],fitted))
    # Fitting never receives validation labels. Validation chooses fitted vs OFF only.
    enabled=candidate['macro_f1']>off['macro_f1']
    selected=fitted if enabled else dict(a=1.,c=0.)
    for key in ('macro_auroc_ovr','macro_auprc_ovr'):
        if not np.isclose(candidate[key],off[key],rtol=0,atol=1e-12):
            raise RuntimeError('Positive affine correction changed directional ranking')
    return dict(enabled=enabled,selected=selected,fitted=fitted,off_metrics=off,
        candidate_metrics=candidate,macro_f1=candidate['macro_f1'] if enabled else off['macro_f1'],
        selection='strict_validation_macro_f1_improvement_else_identity; no threshold search')


def frozen_random(bundles,parameters,ratio,seed):
    """Reuse frozen direction scores; never fit or select at a random ratio."""
    full=bundles['complete'];ids=np.asarray(full['participant_ids']).astype(str)
    plan,metadata=missingness_plan(ids,ratio,seed)
    patterns=np.array([plan[str(pid)] for pid in ids])
    z=np.asarray(full['logits'],dtype=np.float64).copy()
    for pattern in PATTERNS:
        b=bundles[pattern]
        if not np.array_equal(ids,np.asarray(b['participant_ids']).astype(str)) or not np.array_equal(full['labels'],b['labels']):
            raise ValueError('Directional participant order/labels changed')
        selected=patterns==pattern
        z[selected]=transform_logits(b['logits'][selected],parameters[pattern])
    return with_logits(dict(full,patterns=patterns,missingness=metadata),z)


def load_bundle(path):
    with np.load(path,allow_pickle=False) as data:result={k:data[k] for k in data.files}
    return with_logits(result,result['logits'])


def run_logit_case(source,case,spec,output,device,devices):
    from .method_study import make_resource_case
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity=dict(case=case,spec=spec,source_sha256=stable_hash(source),
        implementation_sha256=implementation_sha256(Path(__file__).resolve().parents[2]))
    identity_path=output/'identity.json';path=output/'validation_result.json'
    if identity_path.exists() and read_json(identity_path)!=identity:
        raise RuntimeError('Output-control identity changed')
    atomic_write_json(identity,identity_path);verify_source(source)
    if path.exists():
        result=read_json(path)
        if result['identity']!=identity or result['status']!='complete':raise RuntimeError('Invalid completed output control')
        for rec in result['provenance']:verify_file(rec)
        return result
    started=time.perf_counter();seed_everything(case['seed'])
    runner=SuffixRunner(source,make_resource_case(case['seed']),output/'resources',output/'cache',device,devices)
    loaders,datasets=runner._build_loaders()
    graph,_=runner._load_frozen_graph(runner._train_or_resume())
    filler=NormalizedMeanFiller();provenance=[file_record(identity_path)]
    bundles={};decisions={};fit_times={}
    try:
        for pattern in ['complete',*PATTERNS]:
            # Cached inference is immutable and bound to this case/source identity.
            cache=output/'inference'/f'validation_{pattern}.npz';record=cache.with_suffix('.json')
            if record.exists():verify_file(read_json(record));bundles[pattern]=load_bundle(cache)
            else:
                bundles[pattern]=evaluate_control(graph,loaders['validation'],device,filler=filler,fixed_pattern=pattern)
                save_prediction_bundle(bundles[pattern],cache);atomic_write_json(file_record(cache),record)
            provenance.extend([file_record(cache),file_record(record)])
            if pattern=='complete':continue
            cache=output/'inference'/f'train_{pattern}.npz';record=cache.with_suffix('.json')
            if record.exists():verify_file(read_json(record));train=load_bundle(cache)
            else:
                train=evaluate_control(graph,loaders['look_train'],device,filler=filler,fixed_pattern=pattern)
                save_prediction_bundle(train,cache);atomic_write_json(file_record(cache),record)
            provenance.extend([file_record(cache),file_record(record)])
            start=time.perf_counter()
            decisions[pattern]=freeze_direction(train,bundles[pattern],spec['logit_affine'])
            fit_times[pattern]=time.perf_counter()-start
        parameters={p:decisions[p]['selected'] for p in PATTERNS}
        atomic_write_json(decisions,output/'frozen_directions.json')
        provenance.append(file_record(output/'frozen_directions.json'))
        predictions={'complete':bundles['complete']}
        for p in PATTERNS:predictions[p]=with_logits(bundles[p],transform_logits(bundles[p]['logits'],parameters[p]))
        for ratio in spec['random_ratios']:
            predictions[f'random_{ratio:.1f}']=frozen_random(bundles,parameters,ratio,spec['missingness_seed'])
        validation={}
        for name,b in predictions.items():
            pred=output/'predictions'/f'validation__{name}.npz';save_prediction_bundle(b,pred)
            provenance.append(file_record(pred));validation[name]=b['metrics']
        costs=dict(fit_seconds=fit_times,case_elapsed_seconds_this_invocation=time.perf_counter()-started,
            added_parameters_per_direction=2,shared_pca_reused=False,
            inference='Frozen filling backbone plus one scalar multiply and add; latency not measured here')
        atomic_write_json(costs,output/'costs.json');provenance.append(file_record(output/'costs.json'))
        result=dict(status='complete',phase='validation',test_access=False,identity=identity,case=case,
            completed_at_utc=utc_now(),source=source['result'],checkpoint=source['checkpoint'],validation=validation,
            decisions=decisions,diagnostics={},costs=costs,provenance=provenance,
            interpretation='Features unchanged from filling; auxiliary supervised boundary/scale control. Directional AUROC/AUPRC unchanged; mixed-direction ranking need not be.')
        atomic_write_json(result,path);return result
    finally:
        del graph,runner,loaders,datasets;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
