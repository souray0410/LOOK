"""Real execution of one selected-parent LOOK host, correction and evaluation.

Each stage is content-bound and resumable. This command is GPU work, never run
on a login node by the CPU discovery controller.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.runtime.host_checkpoint import atomic_save
from look.models.native_materialization import verify_selected, load_selected
from look.models.native_host import build_native_host
from look.data.observed_pair import from_parent_specs, collate_observed
from look.evaluation.native_replay import replay_selected
from look.evaluation.evaluator import evaluate_missing
from look.evaluation.observed_suite import PATTERNS, fit_logit_controls, evaluate_suite
from look.methods.operator import prepare_complete_pca_bank, greedy_fit_look, load_selected_bank
from look.methods.joint import correction_sites
from look.training.observed_host import train_host, validate_config


class Paused(Exception):pass


class CheckedLoader:
    def __init__(self, loader, should_pause):
        self.loader=loader;self.dataset=loader.dataset;self.drop_last=loader.drop_last
        self.should_pause=should_pause
    def __iter__(self):
        for batch in self.loader:
            if self.should_pause():raise Paused()
            yield batch
    def __len__(self):return len(self.loader)


def read(path):return json.loads(Path(path).read_text())


def evidence_files(root):
    root=Path(root)
    # Nested acceptance receipts (in particular the host's plateau evidence)
    # belong to the evidence. Only our own not-yet-written receipt is excluded.
    return {str(p.relative_to(root)):file_sha256(p) for p in root.rglob('*')
        if p.is_file() and p!=root/'accepted.json' and
        p.name not in ('status.json','run.lock','pause.json') and
        '.partial' not in p.name and 'quarantine' not in p.relative_to(root).parts}


def verify_case(root,spec):
    root=Path(root);receipt=read(root/'accepted.json')
    if receipt.get('schema')!='look_project_case_v1' or receipt.get('identity')!=stable_hash(spec) or receipt.get('test_access') is not False or receipt.get('state')!='accepted':
        raise ValueError('Project acceptance identity mismatch')
    if not receipt.get('host_plateau') or receipt.get('host_frozen_for_correction') is not True:
        raise ValueError('Missing host completion/freeze evidence')
    for name,digest in receipt['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha256(path)!=digest:raise ValueError('Project evidence changed')
    required={'host/accepted.json','host/best.pt','development/suite.json','controls.json','spec.json'}
    if not required.issubset(receipt['files']):raise ValueError('Missing project evidence')
    return receipt


def validate_spec(spec):
    if spec.get('schema')!='look_project_case_v1' or spec.get('test_access') is not False:
        raise ValueError('Unregistered/sealed project task')
    if spec['seed'] not in (3416,3417,3418) or spec['position'] not in ('middle','deep','features'):
        raise ValueError('Unregistered host/seed')
    if spec['methods']!=['look','single_final','all_on','bias','affine','available_parent']:
        raise ValueError('Incomplete matched method scope')
    validate_config(spec['training'])
    for role in ('first','second'):
        item=spec['parents'][role]
        if file_sha256(Path(item['path'])/'selected_artifact.json')!=item['manifest_sha256']:
            raise ValueError('Parent manifest changed')
        _,parent,_=verify_selected(item['path'])
        if parent['training']['seed']!=spec['seed']:raise ValueError('Parent seed mismatch')
    for item in spec['source_pins']:
        if file_sha256(Path(item['path']))!=item['sha256']:raise ValueError('Source pin changed')


def _execute(spec,out,device,should_pause):
    from mhd_framework.models import create_model
    from mhd_models.workflows.native import Inputs,collate
    validate_spec(spec);identity=stable_hash(spec)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(spec['seed']);np.random.seed(spec['seed']);random.seed(spec['seed'])
    def status(stage):
        atomic_write_json(dict(state='running',stage=stage,identity=identity,pid=os.getpid(),updated_at=time.time(),test_access=False),out/'status.json')
    if (out/'spec.json').exists() and read(out/'spec.json')!=spec:raise ValueError('Run identity changed')
    atomic_write_json(spec,out/'spec.json')
    models=[];parent_specs=[]
    for role in ('first','second'):
        status('parent_replay_'+role);root=Path(spec['parents'][role]['path'])
        model=load_selected(root,create_model,device="cpu");models.append(model)
        parent_specs.append(read(root/'spec.json'))
        replay_path=out/'parents'/role/'replay.json'
        if replay_path.exists():
            r=read(replay_path)
            if (r['best_sha256']!=file_sha256(root/'best.pt') or r['status']!='accepted' or
                r.get('source_receipt_sha256')!=file_sha256(root/'source_accepted.json') or
                r.get('execution_provenance')!=model.execution_provenance):raise ValueError('Parent replay receipt changed')
        else:
            model.to(device)
            try:replay_selected(model,root,Inputs,collate,device,spec['training']['microbatch'],replay_path)
            finally:model.cpu()
    status('host_materialization')
    graph=build_native_host(*models,spec['position'],device=device)
    # Each selected parent remains available on CPU for native-only evaluation.
    for model in models:model.cpu()
    train=from_parent_specs(*parent_specs,'train',Inputs,augment=True,seed=spec['seed'])
    fit=from_parent_specs(*parent_specs,'train',Inputs,augment=False,seed=spec['seed'])
    dev=from_parent_specs(*parent_specs,'development',Inputs,augment=False,seed=spec['seed'])
    if (out/'host'/'accepted.json').exists():
        host=read(out/'host'/'accepted.json')
        if host['identity']!=identity:raise ValueError('Host identity changed')
        for name,digest in host['files'].items():
            if file_sha256(out/'host'/name)!=digest:raise ValueError('Host evidence changed')
        selected=torch.load(out/'host'/'best.pt',map_location='cpu',weights_only=False)
        graph.load_state_dict(selected['model'],strict=True)
    else:
        status('host_training')
        host=train_host(graph,train,dev,spec['training'],spec['seed'],out/'host',identity,device,should_pause)
        if host.get('state')!='accepted':return host
    graph.eval()
    for p in graph.parameters():p.requires_grad_(False)
    frozen={k:v.detach().cpu().clone() for k,v in graph.state_dict().items()}
    def loader(dataset):
        return CheckedLoader(DataLoader(dataset,batch_size=spec['training']['microbatch'],collate_fn=collate_observed,
            num_workers=spec['training']['num_workers'],shuffle=False,generator=torch.Generator().manual_seed(spec['seed'])),should_pause)
    train_loader,dev_loader=loader(fit),loader(dev)
    status('complete_training_pca')
    sites=correction_sites(graph);cfg=spec['look']
    bank=prepare_complete_pca_bank(graph,train_loader,sites,cfg['factors'],cfg['max_rank'],device,
        out/'pca',dict(case=identity,host_best_sha256=host['files']['best.pt']),out/'quarantine')
    banks={method:{} for method in ('look','single_final','all_on')}
    for pattern in PATTERNS[1:]:
        for method in banks:
            status('correction_'+pattern+'_'+method)
            destination=out/'corrections'/method/pattern
            if (destination/'factor_selection.json').exists():
                artifacts=load_selected_bank(destination)
            else:
                artifacts,_=greedy_fit_look(graph,train_loader,dev_loader,pattern,
                    ['fusion_participant_feature'] if method=='single_final' else sites,
                    cfg['factors'],cfg['latent_dims'],cfg['max_rank'],device,destination,bank,
                    force_enable=method=='all_on')
            banks[method][pattern]=artifacts
    status('simple_train_only_controls')
    if (out/'controls.json').exists():controls=read(out/'controls.json')
    else:
        complete=evaluate_missing(graph,train_loader,device,fixed_pattern='complete')
        controls={pattern:fit_logit_controls(complete,evaluate_missing(graph,train_loader,device,fixed_pattern=pattern)) for pattern in PATTERNS[1:]}
        atomic_write_json(controls,out/'controls.json')
    status('matched_development_evaluation')
    records=evaluate_suite(graph,dev_loader,device,banks,controls,out/'development',cfg['ratios'],cfg['mask_seed'])
    from look.evaluation.observed_native import evaluate_available
    records+=evaluate_available(models,dev,device,out/'development',spec['training']['microbatch'])
    atomic_write_json(dict(schema='look_matched_suite_v1',split='development',test_access=False,records=records),out/'development'/'suite.json')
    for k,v in graph.state_dict().items():
        if not torch.equal(v.cpu(),frozen[k]):raise ValueError('Frozen host or BN changed during correction')
    status('statistics_and_report')
    from look.analysis.observed_report import write_report
    write_report(records,out/'report',spec['bootstrap_iterations'],spec['seed'])
    # Last/checkpoint, decisions and all prediction evidence remain available;
    # completion never means that an optimizer checkpoint merely exists.
    files=evidence_files(out)
    receipt=dict(schema='look_project_case_v1',identity=identity,state='accepted',test_access=False,
        host_plateau=True,host_frozen_for_correction=True,records=len(records),files=files)
    atomic_write_json(receipt,out/'accepted.json');return verify_case(out,spec)


def execute(spec,output,device='cuda:0'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    stop=False
    def request_stop(*_):
        nonlocal stop
        stop=True
    for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,request_stop)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'accepted.json').exists():return verify_case(out,spec)
        try:
            result=_execute(spec,out,torch.device(device),lambda:stop or (out/'pause.json').exists())
        except Paused:
            result=dict(state='paused',stage='correction_or_evaluation',test_access=False)
        except Exception as error:
            atomic_write_json(dict(state='needs_review',error=repr(error),test_access=False,updated_at=time.time()),out/'status.json')
            raise
        atomic_write_json(dict(state='completed' if result.get('state')=='accepted' else result['state'],updated_at=time.time(),test_access=False),out/'status.json')
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();execute(read(a.spec),a.output,a.device)

if __name__=='__main__':main()
