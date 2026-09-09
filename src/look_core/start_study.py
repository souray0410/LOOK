"""Suffix-start LOOK experiments, with immutable references to a completed parent study.

The numerical LOOK/PCA implementation is unchanged. Every suffix is a fresh greedy
fit; selected starts are chosen per fixed missing direction, never per mask ratio.
"""
from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
import numpy as np

from .config import ExperimentConfig, ExperimentSelection
from .filling import NormalizedMeanFiller, RawZeroFiller, PairedCGANFiller
from .gan import load_generator
from .joint import correction_sites, PROTOCOL as JOINT_PROTOCOL
from .look import FullFeaturePCA, load_selected_bank, validate_global_factor_bank
from .pipeline import ExperimentRunner, PipelineOptions
from .reproducibility import backbone_implementation_sha256, implementation_sha256, seed_everything
from .state import PipelineState, atomic_write_json, file_sha256, stable_hash, utc_now
from .unified_study import correction_stages, validate_spec

PROTOCOL = 'suffix_start_validation_selection_v1'
STAGES = ['input', 'stem', 'layer1', 'layer2', 'layer3', 'layer4', 'feature']
FILLINGS = ['normalized_mean', 'raw_zero', 'paired_cgan']
SEEDS = [3407, 3408, 3409]
PATTERNS = ['oct_missing', 'cfp_missing']
RULE = 'validation_macro_f1_desc_then_canonical_start_ordinal_asc'


def read_json(path):
    return json.loads(Path(path).read_text())


def sites_for_fusion(fusion):
    i = STAGES.index(fusion)
    return [f'joint_{s}' for s in STAGES[:i + 1]] + [f'fusion_{s}' for s in STAGES[i:]] + ['fusion_participant_feature']


def make_cases(spec, selected):
    validate_spec(spec)
    if spec['seeds'] != SEEDS or len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError('Suffix protocol requires three selected positions and the three declared seeds')
    original = dict(correction_stages(spec, selected))
    cases = []
    for filling in FILLINGS:
        for fusion in selected:
            sites = sites_for_fusion(fusion)
            for seed in SEEDS:
                context = f'{filling}_{fusion}_{seed}'
                for i, start in enumerate(sites):
                    reuse = context if i == 0 else None
                    if filling == 'normalized_mean' and seed == 3407 and start == f'fusion_{fusion}':
                        reuse = f'fusion_only_{fusion}_3407'
                        if original[reuse].look_profiles[0]['correction_nodes'] != sites[i:]:
                            raise RuntimeError('Parent fusion_only is not the requested suffix')
                    cases.append(dict(case_id=f'{context}__start_{i + 1:02d}_{start}', context=context,
                        fusion_position=fusion, filling=filling, seed=seed, start_ordinal=i + 1,
                        allowed_start=start, candidate_sites=sites, eligible_sites=sites[i:], reuse_stage=reuse))
    if len(cases) != 243 or sum(c['reuse_stage'] is None for c in cases) != 213:
        raise RuntimeError('Unexpected suffix grid size')
    return cases


def choose_start(rows):
    if len(rows) != 9 or sorted(r['start_ordinal'] for r in rows) != list(range(1, 10)):
        raise ValueError('All nine unique starts must complete before selection')
    if any(not math.isfinite(float(r['macro_f1'])) for r in rows):
        raise ValueError('Non-finite selection score')
    return min(rows, key=lambda r: (-float(r['macro_f1']), r['start_ordinal']))


def file_record(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=file_sha256(path), bytes=path.stat().st_size)


def verify_file(record):
    path = Path(record['path'])
    if not path.is_file() or path.stat().st_size != record['bytes'] or file_sha256(path) != record['sha256']:
        raise RuntimeError(f'Pinned artifact changed or missing; replacement forbidden: {path}')
    return path


def verify_parent(parent, spec, spec_path):
    if parent.get('test_access') is not False or parent.get('protocol') != spec['protocol']:
        raise RuntimeError('Parent protocol or sealed-test contract differs')
    if parent.get('spec_sha256') != file_sha256(spec_path):
        raise RuntimeError('Parent spec hash differs')
    if parent.get('status') != 'complete' or len(parent.get('completed_stages', {})) != 52:
        raise RuntimeError('Parent must complete all 52 stages first')
    if any(r.get('status') != 'complete' for r in parent['completed_stages'].values()):
        raise RuntimeError('A parent stage is incomplete')


def source_result(parent, runs, stage):
    plan_id = parent['completed_stages'][stage]['plan_id']
    plan = read_json(runs / 'sweeps' / f'validation__{plan_id}' / 'study_plan.json')
    if plan['phase'] != 'validation' or len(plan['experiment_ids']) != 1:
        raise RuntimeError('Expected one validation experiment in parent stage')
    eid = plan['experiment_ids'][0]
    if Path(eid).name != eid:
        raise ValueError('Invalid experiment identifier')
    return runs / 'experiments' / eid / 'validation_result.json'


def pin_source(result_path, parent, runs, project_root):
    result = read_json(result_path)
    manifest_path = result_path.with_name('validation_manifest.json')
    manifest = read_json(manifest_path)
    config = manifest['config']
    selection = result['selection']
    key = f"{selection['fusion_position']}:{selection['seed']}"
    if (result['status'] != 'complete' or result['phase'] != 'validation' or result.get('test')
            or manifest['implementation_sha256'] != parent['implementation_sha256']
            or manifest['labels_sha256'] != parent['labels_sha256']
            or manifest['selection'] != selection or result['options']['smoke_limit'] is not None):
        raise RuntimeError('Parent result identity or validation provenance differs')
    if config['primary_metric'] != 'macro_f1' or config['missing_patterns'] != PATTERNS:
        raise RuntimeError('Parent correction criterion or directions differ')
    cp = result['checkpoint']
    accepted = parent['backbones'][key]
    if cp['sha256'] != accepted['sha256'] or cp['backbone_id'] != accepted['backbone_id']:
        raise RuntimeError('Parent result uses a different accepted backbone')
    checkpoint = file_record(cp['path'])
    if checkpoint['sha256'] != cp['sha256']:
        raise RuntimeError('Backbone hash mismatch')
    banks = {s.split('/')[0] for s in parent['pca_sources'][key]}
    if len(banks) != 1:
        raise RuntimeError('Expected one shared complete-feature PCA bank')
    bank_path = runs / 'pca' / banks.pop() / 'bank_manifest.json'
    bank = read_json(bank_path)
    identity = bank['identity']
    expected = dict(backbone_id=cp['backbone_id'], checkpoint_sha256=cp['sha256'],
        labels_sha256=manifest['labels_sha256'], split='train', augment=False, max_rank=config['max_pca_rank'],
        feature_implementation_sha256=backbone_implementation_sha256(project_root),
        joint_code_sha256=file_sha256(project_root/'src/look_core/joint.py'),
        pca_code_sha256=file_sha256(project_root/'src/look_core/look.py'),
        image_size=config['image_size'], feature_batch_size=config['micro_batch_size'],
        smoke_limit=None, protocol=JOINT_PROTOCOL)
    if any(identity.get(k) != v for k, v in expected.items()) or stable_hash(identity)[:16] != bank['bank_id']:
        raise RuntimeError('Shared PCA scientific identity mismatch')
    if {e['source_id'] for e in bank['entries']} != set(parent['pca_sources'][key]):
        raise RuntimeError('Shared PCA source inventory mismatch')
    pcas = []
    for entry in bank['entries']:
        record = file_record(entry['path'])
        if record['sha256'] != entry['sha256']:
            raise RuntimeError('Shared PCA hash mismatch')
        pcas.append(dict(record, node=entry['node'], factor=entry['factor'], source_id=entry['source_id']))
    generators = {}
    if selection['filling_strategy'] == 'paired_cgan':
        for direction in ('cfp_to_oct', 'oct_to_cfp'):
            d = result['filling']['directions'][direction]
            record = file_record(d['checkpoint'])
            if record['sha256'] != d['checkpoint_sha256']:
                raise RuntimeError('Generator hash mismatch')
            generators[direction] = record
    return dict(result=file_record(result_path), manifest=file_record(manifest_path), checkpoint=checkpoint,
        pca_manifest=file_record(bank_path), pcas=pcas, generators=generators,
        config=config, selection=selection, filling=result['filling'],
        labels=file_record(config['labels_csv']), natural_labels=file_record(config['natural_labels_csv']))


def verify_source(source):
    for key in ('result', 'manifest', 'checkpoint', 'pca_manifest', 'labels', 'natural_labels'):
        verify_file(source[key])
    for record in [*source['pcas'], *source['generators'].values()]:
        verify_file(record)


def lock_case_identity(path, case, source):
    identity = dict(protocol=PROTOCOL, case=case, source_sha256=stable_hash(source))
    if path.exists() and read_json(path) != identity:
        raise ValueError('Cross-start or cross-source resume is forbidden')
    if not path.exists():
        atomic_write_json(identity, path)
    return identity


class SuffixRunner(ExperimentRunner):
    """Use explicit, pinned parent resources; never enter any resource fitting path."""
    def __init__(self, source, case, runs, cache, device, devices=(0, 1)):
        self.source, self.start_case = source, case
        cfg = dict(source['config'], correction_nodes=case['eligible_sites'], output_root=str(runs), cache_root=str(cache))
        selection = ExperimentSelection(**dict(source['selection'], run_label=f"suffix-start-{case['start_ordinal']:02d}"))
        super().__init__(ExperimentConfig.from_dict(cfg), selection,
            PipelineOptions(train_if_missing=False, strict_backbone_reuse=True, gpu_devices=tuple(devices)), device)
        if self.data_hash != source['labels']['sha256'] or self.natural_data_hash != source['natural_labels']['sha256']:
            raise RuntimeError('Source dataset identity changed')

    def run(self):
        lock_case_identity(self.experiment_dir/'suffix_identity.json', self.start_case, self.source)
        return super().run()

    def _experiment_id(self):
        return super()._experiment_id() + '__source-' + stable_hash(self.source)[:12]

    def _train_or_resume(self):
        return verify_file(self.source['checkpoint'])

    def _prepare_shared_pca(self, graph, train_loader, checkpoint_path):
        if correction_sites(graph) != self.start_case['candidate_sites']:
            raise RuntimeError('Actual graph correction topology differs')
        bank = {}
        for record in self.source['pcas']:
            basis = FullFeaturePCA.load(verify_file(record))
            if (basis.source_id != record['source_id'] or basis.node_name != record['node']
                    or basis.factor != record['factor'] or basis.protocol != JOINT_PROTOCOL):
                raise RuntimeError('PCA payload differs from its pinned identity')
            bank[(basis.node_name, basis.factor)] = basis
        return bank

    def _prepare_filler(self, datasets):
        if self.selection.filling_strategy == 'normalized_mean':
            filler = NormalizedMeanFiller()
        elif self.selection.filling_strategy == 'raw_zero':
            filler = RawZeroFiller()
        else:
            generators = {k: load_generator(verify_file(v), self.device) for k, v in self.source['generators'].items()}
            filler = PairedCGANFiller(generators['cfp_to_oct'], generators['oct_to_cfp'], self.device)
        return filler, self.source['filling']

    def _load_frozen_graph(self, checkpoint_path):
        graph, checkpoint = super()._load_frozen_graph(checkpoint_path)
        if read_json(self.source['result']['path'])['checkpoint']['backbone_id'] != self._backbone_id():
            raise RuntimeError('Frozen backbone implementation or configuration differs')
        return graph, checkpoint


def audit_case(case, result_path):
    """Derived topology/decision evidence; never rewrite legacy experiment files."""
    result = read_json(result_path)
    if result['status'] != 'complete' or result['phase'] != 'validation' or result.get('test'):
        raise RuntimeError('Only complete sealed-test validation results are reportable')
    config = read_json(result_path.with_name('validation_manifest.json'))['config']
    actual = case['candidate_sites'] if config['correction_nodes'] == ['all_available'] else config['correction_nodes']
    if actual != case['eligible_sites']:
        raise RuntimeError('Reused result eligible sites differ')
    from .stable_metrics import logit_metrics
    predictions = []
    for scenario, metrics in result['validation'].items():
        path = result_path.parent/'predictions'/f'validation__{scenario}.npz'
        with np.load(path, allow_pickle=False) as bundle:
            actual_f1 = logit_metrics(bundle['labels'], bundle['logits'])['macro_f1']
        if actual_f1 != metrics['macro_f1']:
            raise RuntimeError(f'Saved predictions do not reproduce F1: {path}')
        predictions.append(file_record(path))
    record = dict(case, result=file_record(result_path), manifest=file_record(result_path.with_name('validation_manifest.json')),
        predictions=predictions, banks={}, validation=result['validation'])
    for pattern in PATTERNS:
        root = result_path.parent / 'look' / pattern
        fs_path = root / 'factor_selection.json'
        fs = read_json(fs_path)
        factor = fs['selected_factor']
        artifacts = load_selected_bank(root)
        validate_global_factor_bank(artifacts, actual, factor)
        decisions = read_json(root/'factors'/f'x{factor}'/'bank_complete.json')['decisions']
        if [d['identity']['node'] for d in decisions] != actual:
            raise RuntimeError('Decision trace does not cover the eligible suffix')
        decision_rows = []
        by_node = {d['identity']['node']: d for d in decisions}
        for ordinal, site in enumerate(case['candidate_sites'], 1):
            d = by_node.get(site)
            row = dict(node=site, ordinal=ordinal, state='excluded' if d is None else ('on' if d['enabled'] else 'off'))
            if d:
                if d['enabled'] != (d['best_candidate_score'] > d['baseline_score']):
                    raise RuntimeError('Decision violates strict-improvement rule')
                row.update(baseline_score=d['baseline_score'], selected_score=d['selected_score'],
                    delta=d['selected_score']-d['baseline_score'], chosen_dimension=d['best_dimension'] if d['enabled'] else None,
                    best_candidate_dimension=d['best_dimension'], candidates=[{k:c[k] for k in ('latent_dim','factor','primary_score','prediction_path','prediction_sha256')} for c in d['candidates']], identity=d['identity'])
            decision_rows.append(row)
        enabled = [a.node_name for a in artifacts]
        if enabled != [r['node'] for r in decision_rows if r['state'] == 'on']:
            raise RuntimeError('Selected matrices and decision trace disagree')
        record['banks'][pattern] = dict(macro_f1=float(result['validation'][f'look_after_fill_{pattern}']['macro_f1']),
            selected_factor=factor, spatial_ratio=f'1/{factor}', first_enabled=enabled[0] if enabled else None,
            enabled_count=len(enabled), decisions=decision_rows, bank_root=str(root),
            selected_manifest=file_record(root/'selected_manifest.json'), factor_selection=file_record(fs_path),
            artifacts=[file_record(root/a['path']) for a in read_json(root/'selected_manifest.json')['artifacts']],
            decision_files=[file_record(p) for p in sorted((root/'factors'/f'x{factor}'/'decisions').glob('*.json'))])
    return record


def evaluate_selected(context, records, source, output, device, devices):
    """Re-evaluate the two independently selected banks together on identical masks."""
    chosen = {}
    for pattern in PATTERNS:
        row = choose_start([dict(start_ordinal=r['start_ordinal'], macro_f1=r['banks'][pattern]['macro_f1'], record=r) for r in records])
        chosen[pattern] = row['record']
    frozen = dict(protocol=PROTOCOL, context=context, selection_rule=RULE, test_access=False,
        source_identity=stable_hash(source), original=records[0]['result'], directions={})
    banks = {}
    for pattern, record in chosen.items():
        b = record['banks'][pattern]
        banks[pattern] = load_selected_bank(Path(b['bank_root']))
        frozen['directions'][pattern] = dict(case_id=record['case_id'], start_ordinal=record['start_ordinal'],
            allowed_start=record['allowed_start'], first_enabled=b['first_enabled'], factor=b['selected_factor'],
            validation_macro_f1=b['macro_f1'], result=record['result'], selected_manifest=b['selected_manifest'], artifacts=b['artifacts'])
    destination = output/'selected_starts'/context
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination/'frozen_selection.json'
    if manifest_path.exists() and read_json(manifest_path) != frozen:
        raise RuntimeError('Frozen start selection changed')
    atomic_write_json(frozen, manifest_path)
    result_path = destination/'validation_result.json'
    if result_path.exists():
        prior = read_json(result_path)
        if prior['frozen_selection_sha256'] != file_sha256(manifest_path):
            raise RuntimeError('Selected-start evaluation identity changed')
        return file_record(result_path)
    runner = SuffixRunner(source, records[0], output/'runtime', output/'cache', device, devices)
    runner.prediction_dir = destination/'predictions'
    runner.factor_look_banks = {}
    seed_everything(source['selection']['seed'])
    loaders, datasets = runner._build_loaders()
    graph, _ = runner._load_frozen_graph(runner._train_or_resume())
    filler, _ = runner._prepare_filler(datasets)
    values = runner._evaluate_split(graph, loaders['validation'], 'validation', filler, banks)
    for pattern in PATTERNS:
        actual = values[f'look_after_fill_{pattern}']['metrics']['macro_f1']
        if actual != frozen['directions'][pattern]['validation_macro_f1']:
            raise RuntimeError('Reloaded selected bank does not reproduce its validation F1')
    atomic_write_json(dict(status='complete', phase='validation', test_access=False,
        frozen_selection_sha256=file_sha256(manifest_path), validation={k:v['metrics'] for k,v in values.items()},
        missingness={k:v['missingness'] for k,v in values.items() if 'missingness' in v},
        interpretation='Validation-selected starts: development evidence, not independent test evidence.'), result_path)
    del runner, graph, loaders, datasets, filler, banks
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return file_record(result_path)


def run_start_study(paths, parent_summary_path, parent_source, devices, *, execute=False, wait_parent=False, render=True):
    parent_summary_path, parent_source = Path(parent_summary_path).resolve(), Path(parent_source).resolve()
    spec_path = parent_source/'configs/unified_study.json'
    spec = read_json(spec_path)
    parent = read_json(parent_summary_path)
    selected = parent['selected_fusions']
    cases = make_cases(spec, selected)
    identity = dict(protocol=PROTOCOL, parent_study=str(parent_summary_path), parent_implementation=parent['implementation_sha256'],
        parent_spec_sha256=file_sha256(spec_path), implementation_sha256=implementation_sha256(paths.project_root),
        selection_rule=RULE, cases=cases, devices=list(devices))
    output = paths.runs_root/'start_study'/stable_hash(identity)[:12]
    summary = dict(identity=identity, output=str(output), status='planned', expected_cases=243, expected_new_cases=213,
        expected_reused_cases=30, parent_stages=52, combined_development_stages=265, completed_cases={}, selected_contexts={}, test_access=False)
    if not execute:
        return summary
    if output.is_relative_to(parent_summary_path.parents[2]):
        raise ValueError('Supplement output must be outside the parent runs root')
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output/'summary.json'
    with PipelineState(output/'state', 'suffix_study', identity, [spec_path]) as state:
        if summary_path.exists():
            old = read_json(summary_path)
            if old['identity'] != identity:
                raise RuntimeError('Supplement resume identity changed')
            summary = old
        def report(status, **fields):
            summary.update(status=status, updated_at_utc=utc_now(), **fields)
            atomic_write_json(summary, summary_path)
            print(f"{utc_now()} suffix_status={status} completed={len(summary['completed_cases'])}/243 {fields}", flush=True)
        try:
            if parent['status'] != 'complete':
                if not wait_parent:
                    raise RuntimeError('Parent incomplete; use --wait-parent to arm the continuation')
                report('waiting_parent', parent_completed=len(parent.get('completed_stages', {})))
                while parent['status'] != 'complete':
                    if parent['status'] in ('failed', 'cancelled'):
                        raise RuntimeError('Parent failed; supplement will not start')
                    time.sleep(30)
                    parent = read_json(parent_summary_path)
            verify_parent(parent, spec, spec_path)
            if parent['implementation_sha256'] != identity['parent_implementation'] or parent['selected_fusions'] != selected:
                raise RuntimeError('Parent identity changed while waiting')
            report('preflight')
            runs = parent_summary_path.parents[2]
            inventory_path = output/'source_inventory.json'
            if inventory_path.exists():
                inventory = read_json(inventory_path)
                verify_file(inventory['parent_summary'])
                for source in inventory['sources'].values():
                    verify_source(source)
            else:
                if implementation_sha256(parent_source) != parent['implementation_sha256']:
                    raise RuntimeError('Parent source changed since execution')
                # Only pipeline reporting may differ. Numerical kernels and data semantics must match.
                protected = [p for folder in ('src/look_core','third_party/MHD_Project/V4') for p in (parent_source/folder).glob('*.py')
                    if p.name != 'pipeline.py']
                for p in protected:
                    new = paths.project_root/p.relative_to(parent_source)
                    if file_sha256(p) != file_sha256(new):
                        raise RuntimeError(f'Scientific source changed: {p.name}')
                sources = {}
                for context in dict.fromkeys(c['context'] for c in cases):
                    sources[context] = pin_source(source_result(parent, runs, context), parent, runs, paths.project_root)
                inventory = dict(parent_summary=file_record(parent_summary_path), sources=sources,
                    protected_sources=[file_record(p) for p in protected])
                atomic_write_json(inventory, inventory_path)
            for p in inventory['protected_sources']:
                verify_file(p)
            source_cases = {}
            for case in cases:
                if case['reuse_stage']:
                    result_path = source_result(parent, runs, case['reuse_stage'])
                    original = inventory['sources'][case['context']]
                    reused = read_json(result_path)
                    reused_manifest = read_json(result_path.with_name('validation_manifest.json'))
                    expected_config = dict(original['config'], correction_nodes=case['eligible_sites'])
                    observed_config = dict(reused_manifest['config'])
                    if observed_config['correction_nodes'] == ['all_available']:
                        observed_config['correction_nodes'] = case['candidate_sites']
                    if (observed_config != expected_config
                            or reused['checkpoint']['sha256'] != original['checkpoint']['sha256']
                            or reused_manifest['implementation_sha256'] != parent['implementation_sha256']
                            or reused_manifest['labels_sha256'] != parent['labels_sha256']
                            or reused['selection']['fusion_position'] != case['fusion_position']
                            or reused['selection']['seed'] != case['seed']
                            or reused['selection']['filling_strategy'] != case['filling']):
                        raise RuntimeError('Reused case scientific configuration differs')
                    source_cases[case['case_id']] = audit_case(case, result_path)
            if len(source_cases) != 30:
                raise RuntimeError('All 30 source cases must pass preflight')
            atomic_write_json(source_cases, output/'legacy_topology_audit.json')
            for name, record in source_cases.items():
                summary['completed_cases'].setdefault(name, record)
            report('running', active_case=None)
            if render:
                from .start_report import write_report
                write_report(summary, output/'report')
            device = torch.device('cuda:0')
            for case in cases:
                name = case['case_id']
                source = inventory['sources'][case['context']]
                if name in summary['completed_cases']:
                    previous = summary['completed_cases'][name]
                    path = verify_file(previous['result'])
                    verify_file(previous['manifest'])
                    for evidence in previous['predictions']:
                        verify_file(evidence)
                    for bank in previous['banks'].values():
                        for evidence in [bank['selected_manifest'], bank['factor_selection'], *bank['artifacts'], *bank['decision_files']]:
                            verify_file(evidence)
                    record = audit_case(case, path)
                elif case['reuse_stage']:
                    record = source_cases[name]
                else:
                    report('running', active_case=name)
                    runner = SuffixRunner(source, case, output/'runtime', output/'cache', device, devices)
                    verify_source(source)
                    runner.run()
                    record = audit_case(case, runner.experiment_dir/'validation_result.json')
                    del runner
                    gc.collect(); torch.cuda.empty_cache()
                summary['completed_cases'][name] = record
                atomic_write_json(record, output/'case_records'/f'{name}.json')
                report('running', active_case=name)
                group = sorted([r for r in summary['completed_cases'].values() if r['context'] == case['context']], key=lambda r:r['start_ordinal'])
                if len(group) == 9:
                    summary['selected_contexts'][case['context']] = evaluate_selected(case['context'], group, source, output, device, devices)
                    report('running', active_case=name)
                    if render:
                        from .start_report import write_report
                        write_report(summary, output/'report')
                state.checkpoint(last_case=name, completed_cases=len(summary['completed_cases']))
            if len(summary['completed_cases']) != 243 or len(summary['selected_contexts']) != 27:
                raise RuntimeError('Incomplete suffix study')
            report('complete', active_case=None)
            if render:
                from .start_report import write_report
                write_report(summary, output/'report')
            state.complete([summary_path, inventory_path])
        except BaseException as error:
            report('failed', error=repr(error))
            raise
    return summary
