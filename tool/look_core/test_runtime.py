"""Inference-only test adapter. No training or hyperparameter search calls."""
from pathlib import Path
import gc
import os
import time
import numpy as np
import torch
from .dual_queue import read, record, verify, atomic, digest
from .test_evaluation import PATTERNS, job_artifacts
from .data import UKBBilateralVisitDataset, make_loader
from .config import ExperimentSelection
from .graph import build_resnet50_mhd_graph
from .filling import NormalizedMeanFiller, RawZeroFiller, PairedCGANFiller
from .gan import load_generator
from .look import load_selected_bank
from .method_kernels import evaluate_control, forward_control
from .method_logit import transform_logits, with_logits, frozen_random, load_bundle
from .method_ssf import SSFAdapter
from .evaluate import save_prediction_bundle, evaluate_missing
from .reproducibility import seed_everything
from .distributed import module_state_sha256


def validate_gate(frozen_path, gate_path):
    f = read(frozen_path); gate = read(gate_path)
    if (gate.get('status') != 'complete' or gate.get('test_access') is not False
            or gate['frozen_manifest'] != record(frozen_path)
            or set(gate['replays']) != {j['id'] for j in f['jobs']}):
        raise ValueError('Every frozen job must pass validation replay before test')
    by_id={j['id']:j for j in f['jobs']}
    for name,rec in gate['replays'].items():
        r = read(verify(rec))
        expected=dict(frozen=record(frozen_path),job_sha256=digest(by_id[name]),phase='validation_replay')
        if (r.get('phase') != 'validation_replay' or r.get('test_access') is not False or r.get('status') != 'complete'
                or r.get('identity') != expected or r.get('kernel_parity') is not True):
            raise ValueError('Invalid replay receipt')
    return f


def load_graph(job, device, microbatch):
    b = job['base']; c = b['config']; s = b['selection']
    checkpoint = torch.load(verify(b['checkpoint']), map_location=device, weights_only=False)
    expected = dict(architecture_id=ExperimentSelection(**s).architecture_id, backbone_training='complete_modalities_only',
                    training_strategy='end_to_end_finetuning', training_stage='complete_modalities',
                    labels_sha256=b['labels']['sha256'],primary_metric='macro_f1')
    if any(checkpoint.get(k) != v for k,v in expected.items()): raise ValueError('Frozen backbone identity mismatch')
    graph = build_resnet50_mhd_graph(s['fusion_position'],c['num_classes'],microbatch,c['image_size'],device,
        pretrained=False,classifier_dropout=c['classifier_dropout'],label_smoothing=c['label_smoothing'])
    graph.load_state_dict(checkpoint['graph_state_dict']); del checkpoint
    graph.eval()
    for p in graph.parameters(): p.requires_grad_(False)
    return graph


def make_test_loader(job, phase, microbatch, replay_participants=8):
    if phase not in ('validation_replay','test'): raise ValueError('Unsupported phase')
    c = job['base']['config']
    # Crucially: no natural-label path is passed or opened, and no train loader exists.
    dataset = UKBBilateralVisitDataset(labels_csv=verify(job['base']['labels']), data_root=c['image_root'],
        image_size=c['image_size'],split='test' if phase=='test' else 'validation',augment=False,
        limit=None if phase=='test' else replay_participants,
        preprocess_cache_root=c['preprocess_cache_root'])
    if len(set(dataset.participant_ids)) != len(dataset): raise ValueError('Repeated participant in evaluation')
    if phase=='test' and len(dataset)!=290: raise ValueError('Test must contain all 290 participants')
    return make_loader(dataset,microbatch,0,False,job['seed'],c['sampling_strategy'])


def load_filler(job, device):
    filling=job['filling']
    if filling=='normalized_mean': return NormalizedMeanFiller()
    if filling=='raw_zero': return RawZeroFiller()
    if filling!='paired_cgan': raise ValueError('Unknown filling')
    g=job['base']['generators']
    return PairedCGANFiller(load_generator(verify(g['cfp_to_oct']),device),
                           load_generator(verify(g['oct_to_cfp']),device),device)


def compare_replay(bundle, ref):
    expected=load_bundle(verify(ref))
    ids=expected['participant_ids'].astype(str)
    mapping={pid:i for i,pid in enumerate(ids)}
    if len(mapping)!=len(ids): raise ValueError('Repeated reference participant')
    index=[mapping[str(pid)] for pid in bundle['participant_ids']]
    actual=bundle['logits']; target=expected['logits'][index]
    if not np.array_equal(bundle['labels'],expected['labels'][index]): raise ValueError('Replay label/participant mismatch')
    error=float(np.max(np.abs(actual-target)))
    # CPU/GPU and convolution batch algorithms need not be bitwise identical.
    # The separate same-device kernel parity gate is strict; historical floating
    # drift is recorded rather than hidden by an empirically widened tolerance.
    if not np.array_equal(np.argmax(actual,1),np.argmax(target,1)): raise ValueError('Frozen replay classifications differ')
    return dict(max_absolute_logit_error=error,n=len(index),argmax_equal=True,
                strict_historical_logits_close=bool(np.allclose(actual,target,atol=1e-3,rtol=1e-4)),
                historical_comparison='execution-device/batch floating drift; same-device kernel parity checked separately')


def infer(job, loader, device, *, check_reference=False):
    """Three fixed directions; ratios reuse fixed participant logits exactly."""
    graph=load_graph(job,device,loader.batch_size); before=module_state_sha256(graph)
    banks={p:load_selected_bank(Path(b['root'])) for p,b in job['banks'].items()}
    filler=load_filler(job,device); adapters={}; baseline={}; corrected={}
    try:
        if job['policy']=='ssf':
            for p in PATTERNS:
                a=SSFAdapter(graph,p)
                a.load_state_dict(torch.load(verify(job['parameters'][p]),map_location=device,weights_only=True))
                a.eval()
                for parameter in a.parameters():parameter.requires_grad_(False)
                adapters[p]=a
        def predict(o,c,pattern):
            for p,a in adapters.items(): a.enabled=p==pattern
            return forward_control(graph,o,c)
        patterns=['complete'] if job['family']=='single_modality' else ['complete',*PATTERNS]
        for pattern in patterns:
            for a in adapters.values():a.enabled=False
            baseline[pattern]=evaluate_control(graph,loader,device,filler=filler,fixed_pattern=pattern)
            if pattern=='complete':corrected[pattern]=baseline[pattern]
            elif job['policy']=='logit_affine':
                corrected[pattern]=with_logits(baseline[pattern],transform_logits(baseline[pattern]['logits'],job['parameters'][pattern]))
            else:
                corrected[pattern]=evaluate_control(graph,loader,device,banks,filler=filler,fixed_pattern=pattern,
                    policy='joint' if adapters else job['policy'],predict=predict if adapters else None)
        if check_reference and job['policy']=='joint':
            # Independent historical LOOK forward path, same device/data/batch.
            for pattern in patterns:
                reference=evaluate_missing(graph,loader,device,fixed_pattern=pattern,filler=filler,artifact_banks=banks)
                if not np.allclose(corrected[pattern]['logits'],reference['logits'],atol=1e-6,rtol=1e-6):
                    raise ValueError('Same-device original LOOK kernel parity failed')
        # Method policies call their unchanged published validation inference
        # functions directly. Their module hashes are pinned by the request.
        if before!=module_state_sha256(graph): raise ValueError('Inference mutated frozen backbone state')
        return baseline,corrected
    finally:
        for a in adapters.values():a.close()
        del graph,banks,filler,adapters;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()


def run_job(job, output, phase, frozen_record, *, gate=None, device='cuda:0', microbatch=8, replay_participants=8):
    if phase not in ('validation_replay','test'): raise ValueError('Unknown evaluation phase')
    frozen_path=verify(frozen_record)
    if phase=='test':
        if gate is None: raise ValueError('Test requires the validation replay gate')
        manifest=validate_gate(frozen_path,gate)
        if job not in manifest['jobs']: raise ValueError('Job outside frozen roster')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity=dict(frozen=frozen_record,job_sha256=digest(job),phase=phase)
    path=output/'complete.json'
    if path.exists():
        old=read(path)
        if old['identity']!=identity or old['status']!='complete':raise ValueError('Cross-case/phase resume')
        for r in old['predictions'].values():verify(r)
        return old
    for r in job_artifacts(job):verify(r)
    seed_everything(job['seed']);device=torch.device(device)
    torch.set_num_threads(2)
    if device.type=='cuda':
        if torch.cuda.device_count()!=1:raise ValueError('Exactly one assigned GPU must be visible')
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(11*1024**3/total,1.),0)
    loader=make_test_loader(job,phase,microbatch,replay_participants)
    begun=time.time()
    baseline,corrected=infer(job,loader,device,check_reference=phase=='validation_replay')
    bundles={'complete':baseline['complete']}
    if job['family']!='single_modality':
        for p in PATTERNS:bundles['fill_'+p]=baseline[p];bundles['corrected_'+p]=corrected[p]
        if phase=='test':
            identity_parameters={p:dict(a=1.,c=0.) for p in PATTERNS}
            for ratio in (.2,.4,.6,.8,1.):
                bundles[f'fill_random_{ratio:.1f}']=frozen_random(baseline,identity_parameters,ratio,3407)
                bundles[f'corrected_random_{ratio:.1f}']=frozen_random(corrected,identity_parameters,ratio,3407)
    replay={}
    if phase=='validation_replay':
        is_method=job['family']=='method'
        for name,bundle in bundles.items():
            refname=(name.removeprefix('corrected_') if is_method else name.replace('corrected_','look_after_fill_'))
            if is_method and name.startswith('fill_'):continue
            if refname not in job['validation_predictions']:raise ValueError(f'Missing frozen validation reference: {refname}')
            replay[name]=compare_replay(bundle,job['validation_predictions'][refname])
    predictions={}
    for name,bundle in bundles.items():
        p=output/'predictions'/f'{phase}__{name}.npz';save_prediction_bundle(bundle,p);predictions[name]=record(p)
    value=dict(status='complete',identity=identity,phase=phase,test_access=phase=='test',cohort='test' if phase=='test' else 'validation_subset',
        job_id=job['id'],family=job['family'],seed=job['seed'],fusion=job['fusion'],filling=job['filling'],
        n=len(loader.dataset),microbatch=microbatch,precision='FP32',kernel_parity=True,parameters_refit=False,configuration_search=False,
        metrics={k:v['metrics'] for k,v in bundles.items()},predictions=predictions,replay=replay,
        missingness={k:v['missingness'] for k,v in bundles.items() if 'missingness' in v},
        physical_gpu=os.environ.get('LOOK_PHYSICAL_GPU'),physical_uuid=os.environ.get('LOOK_ASSIGNED_GPU_UUID'),
        started_at_unix=begun,completed_at_unix=time.time())
    atomic(value,path);return value
