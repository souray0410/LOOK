"""Resume the original matched LOOK study using a converted V5 frozen host.

The historical data adapter is used only by this explicit migration entrypoint.
All outputs have one V5 identity and keep the original source run as provenance.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from look.data.observed_pair import collate_observed
from look.evaluation.evaluator import evaluate_missing
from look.evaluation.observed_suite import PATTERNS, evaluate_suite, fit_logit_controls
from look.methods.joint import correction_sites
from look.methods.operator import greedy_fit_look, load_selected_bank
from look.models.native_host import build_native_host
from look.runtime.host_checkpoint import read_selected
from look.runtime.provenance import write_json_atomic
from look.runtime.state import file_sha256, stable_hash
from look.studies.project_case import CheckedLoader, Paused, evidence_files
from look.studies.v5_project_feature_replay import datasets
from look.studies.v5_project_full_replay import FRAMEWORK, MODELS, _parent, read, validate_claim
from look.studies.v5_project_sharded_first_decision import load_pca

METHODS = ['look', 'single_final', 'all_on', 'bias', 'affine', 'available_parent']


def fit_banks(graph, train, dev, sites, cfg, bank, device, output):
    """Execute the original method/pattern/factor scope without selecting new cases."""
    banks = {method: {} for method in ('look', 'single_final', 'all_on')}
    for pattern in PATTERNS[1:]:
        for method in banks:
            destination = output / 'corrections' / method / pattern
            if (destination / 'factor_selection.json').exists():
                artifacts = load_selected_bank(destination)
            else:
                artifacts, _ = greedy_fit_look(
                    graph, train, dev, pattern,
                    ['fusion_participant_feature'] if method == 'single_final' else sites,
                    cfg['factors'], cfg['latent_dims'], cfg['max_rank'], device,
                    destination, bank, resume=True, force_enable=method == 'all_on')
            banks[method][pattern] = artifacts
    return banks


def verify_complete(output, identity):
    receipt = read(output / 'accepted.json')
    if (receipt.get('schema') != 'look_formal_v5_matched_suite_receipt_v1'
            or receipt.get('identity_sha256') != stable_hash(identity)
            or receipt.get('state') != 'accepted' or receipt.get('test_access') is not False):
        raise ValueError('Matched suite acceptance identity changed')
    for name, digest in receipt['files'].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or file_sha256(path) != digest:
            raise ValueError('Matched suite evidence changed')
    return receipt


def execute(*, source_run, checkpoint, cache, pca, output, inputs_factory,
            device, gpu_budget_bytes, should_pause=lambda: False):
    from mhd_framework import __api_version__
    if __api_version__ != 'V5':
        raise ValueError('Formal V5 runtime required')
    source_run, checkpoint, cache, pca, output = map(
        lambda p: Path(p).resolve(), (source_run, checkpoint, cache, pca, output))
    if output == source_run or output.is_relative_to(source_run):
        raise ValueError('Migration output must be separate from its historical source')
    spec = read(source_run / 'spec.json')
    if (spec.get('schema') != 'look_project_case_v1'
            or spec.get('test_access') is not False or spec.get('methods') != METHODS):
        raise ValueError('The original complete matched study scope is required')
    cache_receipt = read(cache / 'accepted.json')
    if cache_receipt.get('state') != 'accepted_complete':
        raise ValueError('Complete feature cache required')
    pca_identity, bank = load_pca(pca, file_sha256(cache / 'accepted.json'))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(spec['seed']); np.random.seed(spec['seed']); random.seed(spec['seed'])
    total = torch.cuda.get_device_properties(device).total_memory
    if not 0 < gpu_budget_bytes < total:
        raise ValueError('A measured GPU budget is required')
    torch.cuda.set_per_process_memory_fraction(gpu_budget_bytes / total, device)
    parents = [_parent(spec['parents'][key]['path'], spec['parents'][key]['manifest_sha256'])
               for key in ('first', 'second')]
    graph = build_native_host(*parents, spec['position'], device=device)
    accepted = read(source_run / 'host/accepted.json')
    nodes = [(n.id, n.name) for n in sorted(graph.nodes, key=lambda n: n.id)]
    state = read_selected(checkpoint, identity=accepted['identity'], node_ids=nodes)
    graph.load_state_dict(state['model'], strict=True); graph.eval()
    for parameter in graph.parameters():
        parameter.requires_grad_(False)
    frozen = {name: value.detach().cpu().clone() for name, value in graph.state_dict().items()}
    data, cohort = datasets(source_run, inputs_factory, seed=spec['seed'])
    sites = correction_sites(graph)
    if len(sites) != 9 or sites != pca_identity['sites']:
        raise ValueError('All nine ordered sites must match the PCA bank')
    identity = dict(schema='look_formal_v5_matched_suite_v1', framework_commit=FRAMEWORK,
        models_commit=MODELS, source_spec_sha256=file_sha256(source_run / 'spec.json'),
        checkpoint_sha256=file_sha256(checkpoint), cache_sha256=file_sha256(cache / 'accepted.json'),
        pca_sha256=file_sha256(pca / 'accepted.json'), cohort=cohort, sites=sites,
        methods=METHODS, look=spec['look'], test_access=False)
    cache_identity = read(cache / 'identity.json')
    if (cache_identity.get('source_spec_sha256') != identity['source_spec_sha256']
            or cache_identity.get('target_checkpoint_sha256') != identity['checkpoint_sha256']):
        raise ValueError('Feature cache belongs to a different source or checkpoint')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'identity.json').exists() and read(output / 'identity.json') != identity:
            raise ValueError('Resumed suite identity changed')
        write_json_atomic(identity, output / 'identity.json')
        if (output / 'accepted.json').exists():
            return verify_complete(output, identity)
        def status(stage, state='running', **fields):
            write_json_atomic(dict(state=state, stage=stage, updated_at=time.time(),
                pid=os.getpid(), test_access=False, **fields), output / 'status.json')
        def loader(role):
            return CheckedLoader(DataLoader(data[role], batch_size=spec['training']['microbatch'],
                shuffle=False, num_workers=spec['training']['num_workers'], collate_fn=collate_observed,
                generator=torch.Generator().manual_seed(spec['seed'])), should_pause)
        try:
            train, dev = loader('train'), loader('development'); cfg = spec['look']
            status('correction_fitting')
            banks = fit_banks(graph, train, dev, sites, cfg, bank, device, output)
            status('train_only_controls')
            if (output / 'controls.json').exists():
                controls = read(output / 'controls.json')
            else:
                complete = evaluate_missing(graph, train, device, fixed_pattern='complete')
                controls = {pattern: fit_logit_controls(complete,
                    evaluate_missing(graph, train, device, fixed_pattern=pattern)) for pattern in PATTERNS[1:]}
                write_json_atomic(controls, output / 'controls.json')
            status('matched_development')
            records = evaluate_suite(graph, dev, device, banks, controls,
                output / 'development', cfg['ratios'], cfg['mask_seed'])
            for name, value in graph.state_dict().items():
                if not torch.equal(value.cpu(), frozen[name]):
                    raise ValueError('Frozen host changed during correction')
            graph.cpu()
            from look.evaluation.observed_native import evaluate_available
            records += evaluate_available(parents, data['development'], device,
                output / 'development', spec['training']['microbatch'])
            write_json_atomic(dict(schema='look_matched_suite_v1', split='development',
                test_access=False, records=records), output / 'development/suite.json')
            status('report')
            from look.analysis.observed_report import write_report
            write_report(records, output / 'report', spec['bootstrap_iterations'], spec['seed'])
            receipt = dict(schema='look_formal_v5_matched_suite_receipt_v1', state='accepted',
                identity_sha256=stable_hash(identity), methods=METHODS, patterns=list(PATTERNS[1:]),
                sites=sites, records=len(records), host_frozen=True, test_access=False,
                files=evidence_files(output), completed_at=time.time(),
                v4_numerical_equivalence_accepted=False)
            write_json_atomic(receipt, output / 'accepted.json'); status('complete', 'completed')
            return verify_complete(output, identity)
        except Paused:
            status('resume_required', 'paused')
            return {'state': 'paused', 'test_access': False}
        except Exception as error:
            status('failed', 'needs_review', error=repr(error)); raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('source-run', 'checkpoint', 'cache', 'pca', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-budget-bytes', required=True, type=int)
    args = parser.parse_args(argv); validate_claim()
    stop = [False]
    def request_stop(*_):stop[0] = True
    for sig in (signal.SIGUSR1, signal.SIGTERM):signal.signal(sig, request_stop)
    deadline = float(os.environ.get('LOOK_LEASE_END_EPOCH', 'inf'))
    from expanded.native import Inputs
    result = execute(source_run=args.source_run, checkpoint=args.checkpoint, cache=args.cache,
        pca=args.pca, output=args.output, inputs_factory=Inputs, device=torch.device(args.device),
        gpu_budget_bytes=args.gpu_budget_bytes,
        should_pause=lambda: stop[0] or time.time() >= deadline or (Path(args.output) / 'pause.json').exists())
    if result['state'] == 'paused':raise SystemExit(75)


if __name__ == '__main__':main()
