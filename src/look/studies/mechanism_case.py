"""Execute one accepted-parent supplement; historical experiments stay immutable."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import time
from contextvars import ContextVar
import numpy as np
import torch
from torch.utils.data import DataLoader
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.runtime.host_checkpoint import cpu_tree
from look.studies.mechanism_protocol import validate_task, TRAINING, VERSION
from look.models.native_materialization import load_selected
from look.models.native_host import build_native_host
from look.data.observed_pair import from_parent_specs, collate_observed
from look.data.mechanism_samples import ParticipantSubset
from look.methods.operator import (load_selected_bank, prepare_complete_pca_bank,
    greedy_fit_look, LOOKArtifact, forward_with_look)
from look.methods.mechanism_operator import fit_variant, FitPaused
from look.methods.joint import correction_sites
from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle
from look.evaluation.observed_suite import evaluate_suite, fit_logit_controls, PATTERNS
from look.training.mechanism_training import train, state_equal
from look.evaluation.mechanism_diagnostics import feature_diagnostics,latency,resources


def read(path): return json.loads(Path(path).read_text())

PAUSE = ContextVar('look_supplement_pause',default=lambda:False)


class PauseLoader:
    def __init__(self, wrapped): self.wrapped=wrapped;self.dataset=wrapped.dataset;self.drop_last=wrapped.drop_last
    def __len__(self):return len(self.wrapped)
    def __iter__(self):
        for batch in self.wrapped:
            if PAUSE.get()():raise FitPaused()
            yield batch


def verify_dependencies(spec):
    validate_task(spec);s=spec['source'];root=Path(s['run_dir'])
    for path,digest in ((root/'accepted.json',s['accepted_sha256']),
                        (root/'host/best.pt',s['best_sha256']),
                        (Path(s['spec_path']),s['spec_sha256'])):
        if file_sha256(path)!=digest:raise ValueError('Source model/evidence changed')
    for item in spec['source_pins']:
        if file_sha256(item['path'])!=item['sha256']:raise ValueError('Supplement source changed')
    return read(s['spec_path'])


def context(spec,device):
    return load_original(verify_dependencies(spec),Path(spec['source']['run_dir']),device)


def load_original(base,run,device):
    from mhd_framework.models import create_model
    from expanded.native import Inputs
    models=[];parents=[]
    for role in ('first','second'):
        path=base['parents'][role]['path']
        models.append(load_selected(path,create_model,device='cpu',allow_inference_equivalence=True))
        parents.append(read(Path(path)/'spec.json'))
    graph=build_native_host(*models,base['position'],device=device)
    state=torch.load(Path(run)/'host/best.pt',map_location='cpu',weights_only=False)
    ids=[(n.id,n.name) for n in sorted(graph.nodes,key=lambda n:n.id)]
    if state['node_ids']!=ids:raise ValueError('Source MHD Node IDs changed')
    graph.load_state_dict(state['model'],strict=True);graph.eval()
    data={role:from_parent_specs(*parents,'train' if role=='fit' else role,Inputs,
        augment=role=='train',seed=base['seed']) for role in ('train','fit','development')}
    return base,models,graph,data


def loader(dataset,seed):
    return PauseLoader(DataLoader(dataset,batch_size=16,collate_fn=collate_observed,shuffle=False,num_workers=0,
                      generator=torch.Generator().manual_seed(seed)))


def frozen_suite(graph,data,base,out,device,identity,pause):
    """The original six comparisons for each newly trained frozen host."""
    for p in graph.parameters():p.requires_grad_(False)
    graph.eval();frozen=cpu_tree(graph.state_dict());cfg=base['look'];seed=base['seed']
    train_loader=loader(data['fit'],seed);dev_loader=loader(data['development'],seed)
    bank=prepare_complete_pca_bank(graph,train_loader,correction_sites(graph),cfg['factors'],cfg['max_rank'],device,
        out/'pca',dict(supplement=identity),out/'quarantine')
    banks={m:{} for m in ('look','single_final','all_on')}
    for m in banks:
        for pattern in PATTERNS[1:]:
            if pause():raise FitPaused()
            target=out/'corrections'/m/pattern
            if (target/'factor_selection.json').exists(): banks[m][pattern]=load_selected_bank(target)
            else:
                sites=['fusion_participant_feature'] if m=='single_final' else correction_sites(graph)
                banks[m][pattern],_=greedy_fit_look(graph,train_loader,dev_loader,pattern,sites,cfg['factors'],
                    cfg['latent_dims'],cfg['max_rank'],device,target,bank,force_enable=m=='all_on')
    full=evaluate_missing(graph,train_loader,device,fixed_pattern='complete')
    controls={p:fit_logit_controls(full,evaluate_missing(graph,train_loader,device,fixed_pattern=p)) for p in PATTERNS[1:]}
    atomic_write_json(controls,out/'controls.json')
    records=evaluate_suite(graph,dev_loader,device,banks,controls,out/'development',cfg['ratios'],cfg['mask_seed'])
    state_equal(graph,frozen)
    return records


def execute_task(spec,out,device,pause,preflight_updates=None):
    PAUSE.set(pause)
    base,models,graph,data=context(spec,device);task=spec['task'];seed=task['host']['seed'];identity=stable_hash(spec)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    from look.studies.mechanism_data_audit import audit_parent_specs
    atomic_write_json(audit_parent_specs([read(Path(base['parents'][role]['path'])/'spec.json') for role in ('first','second')]),out/'data_audit.json')
    if task['kind'].endswith('_training'):
        student=task['kind']=='student_training'
        model=models[0 if task.get('track')=='cfp' else 1].to(device) if student else graph
        teacher=graph if task['arm']=='distill' else None
        if student and teacher is None:graph.cpu()
        training=train(model,data['train'],data['development'],TRAINING,seed,out/'training',identity,device,
            task['arm'],task.get('track'),teacher,pause,preflight_updates)
        if training['state']!='accepted':return training
        if student:
            scenario='oct_missing' if task['track']=='cfp' else 'cfp_missing'
            with np.load(out/'training/student.npz',allow_pickle=False) as r:
                from look.evaluation.stability import logit_metrics
                records=[dict(method=task['arm'],scenario=scenario,path=str(out/'training/student.npz'),
                    sha256=file_sha256(out/'training/student.npz'),metrics=logit_metrics(r['labels'],r['logits']))]
        else:
            records=frozen_suite(graph,data,base,out,device,identity,pause)
            from look.evaluation.observed_native import evaluate_available
            records+=evaluate_available(models,data['development'],device,out/'development',16)
        model.to(device).eval()
        batch=next(iter(loader(data['fit'],seed)))
        if student:
            timed=lambda:model(batch[task['track']].to(device),batch['counts'])
        else:
            timed=lambda:forward_with_look(graph,batch['oct'].to(device),batch['cfp'].to(device),counts=batch['counts'])
        timing=latency(timed,device)
        timing['participant_sha256']=stable_hash(batch['participant_id']);timing['batch']=len(batch['counts'])
        timing['parameters']=sum(p.numel() for p in model.parameters())
        atomic_write_json(timing,out/'latency.json')
        atomic_write_json(records,out/'records.json')
        return dict(state='accepted',training_complete=True)
    for p in graph.parameters():p.requires_grad_(False)
    frozen=cpu_tree(graph.state_dict())
    pattern=task['pattern'];source=Path(spec['source']['run_dir'])/'corrections/look'/pattern
    templates=load_selected_bank(source);fit=data['fit']
    diagnostics=[]
    if task['kind']=='sample_curve':
        fit=ParticipantSubset(fit,task['fraction'],7341618+task['repeat'])
        labels=[int(fit.dataset.first.rows[i]['label']) for i in fit.indices] if fit.dataset.first.rows and 'label' in fit.dataset.first.rows[0] else None
        fit.receipt['cases']=sum(labels) if labels is not None else None
        fit.receipt['cases_status']='recorded' if labels is not None else 'manifest_label_field_requires_audit'
        atomic_write_json(fit.receipt,out/'subset.json')
        if task['fraction']==1.:
            # These views are exactly the original full-data selection. Reference
            # its accepted fit instead of rerunning an identical scientific task.
            result=evaluate_missing(graph,loader(data['development'],seed),device,fixed_pattern=pattern,artifact_banks={pattern:templates})
            path=out/'development.npz';save_prediction_bundle(result,path)
            atomic_write_json([dict(method=task['arm'],scenario=pattern,path=str(path),sha256=file_sha256(path),metrics=result['metrics'])],out/'records.json')
            atomic_write_json(dict(source=str(source),source_acceptance=spec['source']['accepted_sha256'],
                reason='identical_full_train_fit_and_selection',fit_seconds=0.),out/'alias.json')
            state_equal(graph,frozen)
            return dict(state='accepted',training_complete=False,host_frozen=True,fit_alias=True)
        cfg=base['look'];sites=correction_sites(graph)
        bank=prepare_complete_pca_bank(graph,loader(fit,seed),sites,cfg['factors'],cfg['max_rank'],device,
            out/'pca',dict(task=identity,subset=fit.receipt['participant_sha256']),out/'quarantine')
        if task['arm']=='reselect':
            artifacts,_=greedy_fit_look(graph,loader(fit,seed),loader(data['development'],seed),pattern,sites,
                cfg['factors'],cfg['latent_dims'],cfg['max_rank'],device,out/'reselect',bank)
        else:
            replaced=[]
            from dataclasses import replace
            for a in templates:
                pca=bank[a.node_name,a.factor]
                if len(pca.components)<a.latent_dim:
                    return dict(state='infeasible',reason='fixed_dimension_exceeds_subset_rank',node=a.node_name)
                replaced.append(replace(a,mean=pca.mean,std=pca.std,pca_mean=pca.pca_mean,
                    components=pca.components[:a.latent_dim],pca_source_id=pca.source_id,
                    pca_explained_variance=float(pca.explained_variance_ratio[:a.latent_dim].sum()),
                    pca_fit_seconds=pca.fit_seconds,pca_peak_rss_bytes=pca.peak_rss_bytes))
            artifacts,diagnostics=fit_variant(graph,loader(fit,seed),replaced,'sequential',device,seed,out/'fit',pause)
    else:
        artifacts,diagnostics=fit_variant(graph,loader(fit,seed),templates,task['arm'],device,
            7341618+task.get('repeat',0) if task['arm']=='shuffle' else seed,out/'fit',pause)
    if pause():raise FitPaused()
    from look.methods.mechanism_operator import MechanismArtifact
    from look.runtime.host_checkpoint import atomic_save
    atomic_save(out/'artifacts.pt',[(a if isinstance(a,MechanismArtifact) else MechanismArtifact(a)).record() for a in artifacts])
    result=evaluate_missing(graph,loader(data['development'],seed),device,fixed_pattern=pattern,artifact_banks={pattern:artifacts})
    path=out/'development.npz';save_prediction_bundle(result,path)
    records=[dict(method=task['arm'],scenario=pattern,path=str(path),sha256=file_sha256(path),metrics=result['metrics'])]
    if task['kind']=='correction':
        detailed=feature_diagnostics(graph,loader(data['fit'],seed),artifacts,pattern,device)
        atomic_write_json(detailed,out/'feature_diagnostics.json')
    from look.methods.imputation import NormalizedMeanFiller
    batch=next(iter(loader(data['fit'],seed)))
    oct_x,cfp=NormalizedMeanFiller().fill(batch['oct'].to(device),batch['cfp'].to(device),pattern)
    timing=latency(lambda:forward_with_look(graph,oct_x,cfp,artifacts,counts=batch['counts']),device)
    timing['participant_sha256']=stable_hash(batch['participant_id']);timing['batch']=len(batch['counts'])
    timing['host_parameters']=sum(p.numel() for p in graph.parameters())
    timing['correction_parameters']=sum(
        sum(v.numel() for v in a.mlp_state.values()) if getattr(a,'mlp_state',None) is not None else a.weight.numel()+a.bias.numel()
        for a in artifacts)
    timing['matched_host']=latency(lambda:forward_with_look(graph,oct_x,cfp,counts=batch['counts']),device)
    timing['matched_standard_look']=latency(lambda:forward_with_look(graph,oct_x,cfp,templates,counts=batch['counts']),device)
    atomic_write_json(timing,out/'latency.json')
    original_path=Path(spec['source']['run_dir'])/'development'/('host__'+pattern+'.npz')
    with np.load(original_path,allow_pickle=False) as original:
        if not np.array_equal(original['participant_ids'],result['participant_ids']) or not np.array_equal(original['labels'],result['labels']):
            raise ValueError('Changed participant order in error transition diagnostic')
        before=original['logits'].argmax(1)==result['labels'];after=result['logits'].argmax(1)==result['labels']
        atomic_write_json(dict(wrong_to_correct=int((~before&after).sum()),correct_to_wrong=int((before&~after).sum()),
            participants=len(before),scope='paired_dev_prediction_diagnostic_not_causal'),out/'error_transitions.json')
    atomic_write_json(records,out/'records.json');atomic_write_json(diagnostics,out/'diagnostics.json')
    state_equal(graph,frozen)
    return dict(state='accepted',training_complete=False,host_frozen=True)


def verify_case(root,spec):
    r=read(Path(root)/'accepted.json')
    if r.get('schema')!=VERSION or r.get('identity')!=stable_hash(spec) or r.get('test_access') is not False or r.get('state') not in ('accepted','infeasible'):
        raise ValueError('Invalid supplement receipt')
    for name,digest in r['files'].items():
        p=(Path(root)/name).resolve()
        if not p.is_relative_to(Path(root).resolve()) or file_sha256(p)!=digest:raise ValueError('Supplement evidence changed')
    if 'spec.json' not in r['files'] or (r['state']=='accepted' and 'records.json' not in r['files']):
        raise ValueError('Incomplete supplement evidence')
    if spec['task']['kind'].endswith('_training') and r['state']=='accepted':
        training=read(Path(root)/'training/accepted.json')
        if training.get('plateau') is not True or training['identity']!=stable_hash(spec):raise ValueError('Missing plateau evidence')
    return r


def execute(spec,out,device=None):
    device=device or torch.device('cuda:0');out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'accepted.json').exists():return verify_case(out,spec)
        validate_task(spec)
        if (out/'spec.json').exists() and read(out/'spec.json')!=spec:raise ValueError('Run identity changed')
        atomic_write_json(spec,out/'spec.json')
        stopped=[False]
        def stop(*args): stopped[0]=True
        for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,stop)
        def pause():return stopped[0] or (out/'pause.json').exists()
        seed=spec['task']['host']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if device.type=='cuda':
            total=torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction((min(.875*total,total-10*1024**3)-2*1024**3)/total)
        started=time.time();atomic_write_json(dict(state='running',pid=os.getpid(),time=started),out/'status.json')
        try:result=execute_task(spec,out,device,pause)
        except FitPaused:result=dict(state='paused')
        except Exception as exc:
            atomic_write_json(dict(state='failed',error=repr(exc),time=time.time()),out/'status.json');raise
        attempt_dir=out/'attempts';attempt_dir.mkdir(exist_ok=True)
        atomic_write_json(dict(start=started,end=time.time(),seconds=time.time()-started,state=result['state']),
            attempt_dir/(str(time.time_ns())+'.json'))
        if result['state'] in ('accepted','infeasible'):
            atomic_write_json(resources(device),out/'resources.json')
            files={str(p.relative_to(out)):file_sha256(p) for p in out.rglob('*') if p.is_file() and
                p.name not in ('run.lock','pause.json','status.json','accepted.json') and '.partial' not in p.name}
            # Nested training acceptance is required separately, not omitted as a
            # mutable top-level status record.
            for p in out.rglob('accepted.json'):files[str(p.relative_to(out))]=file_sha256(p)
            atomic_write_json(dict(schema=VERSION,identity=stable_hash(spec),test_access=False,
                seconds=sum(read(p)['seconds'] for p in attempt_dir.glob('*.json')),files=files,**result),out/'accepted.json')
        atomic_write_json(dict(state='completed' if result['state']=='accepted' else result['state'],time=time.time()),out/'status.json')
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();execute(read(args.spec),args.output)
