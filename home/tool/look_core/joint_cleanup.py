"""Manifest-scoped retirement of the superseded LOOK protocol, independent of Step 35."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
from .state import atomic_write_json, file_sha256, utc_now

TARGET_STUDY = 'e6d740a884be'


def _read(path):
    return json.loads(Path(path).read_text())


def _inside(path, root):
    return path == root or root in path.parents


def validate_target(path, allowed_roots):
    path = Path(path).absolute()
    if not any(_inside(path, root) and path != root for root in allowed_roots):
        raise ValueError(f'Outside cleanup roots: {path}')
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f'Symlink forbidden: {parent}')
    forbidden = {'backbones', 'baseline_selection', 'generators', 'dataset', 'preprocessed_pairs', 'primary_test', 'natural_test'}
    if forbidden.intersection(path.parts) or path.name.startswith('test__'):
        raise ValueError(f'Protected path: {path}')
    files = list(path.rglob('*')) if path.is_dir() else [path]
    for child in files:
        if child.is_symlink():
            raise ValueError(f'Symlink forbidden: {child}')
        if child.name.startswith(('test_', 'primary_test', 'natural_test')):
            raise ValueError(f'Test evidence is protected: {child}')
        if child.name == 'stage.lock':
            lock = _read(child)
            try:
                os.kill(int(lock['pid']), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f'Live lock: {child}')
    return sum(child.stat().st_size for child in files if child.is_file())


def assert_no_writers(targets):
    """Inspect same-user /proc FDs in addition to stage locks and the old runner."""
    for process in Path('/proc').glob('[0-9]*'):
        if process.name == str(os.getpid()):
            continue
        try:
            command = (process / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
            # Exclude shells merely displaying commands and this cleanup CLI.
            if 'python' in Path(command.split(' ')[0]).name and '34_run_reviewed_look_study.py' in command:
                raise RuntimeError(f'Old runner alive: {process.name}')
            for fd in (process / 'fd').iterdir():
                try:
                    destination = Path(os.readlink(fd).removesuffix(' (deleted)'))
                    if not any(_inside(destination, p) for p in targets):
                        continue
                    info = (process / 'fdinfo' / fd.name).read_text()
                    flags = int(next(line.split()[1] for line in info.splitlines() if line.startswith('flags:')), 8)
                    if flags & os.O_ACCMODE:
                        raise RuntimeError(f'Live writer {process.name}: {destination}')
                except (FileNotFoundError, PermissionError):
                    continue
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def build_cleanup(runs, cache, study=TARGET_STUDY, log=None):
    runs, cache = Path(runs).absolute(), Path(cache).absolute()
    if study != TARGET_STUDY:
        raise ValueError('This migration is authorized for e6d740a884be only')
    source = runs / 'reviewed_look' / study / 'summary.json'
    summary = _read(source)
    if summary.get('status') == 'running':
        raise RuntimeError('Stop the old study before cleanup')
    plans = {x['plan_id'] for x in summary.get('completed_stages', {}).values()}
    if summary.get('active_plan_id'):
        plans.add(summary['active_plan_id'])
    targets = {source.parent}
    eids = set()
    plan_records = []
    for pid in sorted(plans):
        if len(pid) != 12 or any(c not in '0123456789abcdef' for c in pid):
            raise ValueError('Invalid plan ID')
        sweep = runs / 'sweeps' / f'validation__{pid}'
        plan = _read(sweep / 'study_plan.json')
        if plan['phase'] != 'validation' or not all(p.get('enabled') for p in plan['grid']['look_profiles']):
            raise ValueError('Non-LOOK or test plan')
        plan_records.append(plan)
        targets.add(sweep)
        for eid in plan['experiment_ids']:
            if Path(eid).name != eid or eid in ('', '.', '..'):
                raise ValueError('Invalid experiment ID')
            eids.add(eid)
            targets.update((runs / 'experiments' / eid, cache / 'pipeline_state' / f'experiment__{eid}__validation'))
    for other in (runs / 'sweeps').glob('*/study_plan.json'):
        if other.parent in targets:
            continue
        if eids.intersection(_read(other).get('experiment_ids', [])):
            raise ValueError(f'Experiment referenced by another plan: {other}')
    state = cache / 'pipeline_state' / 'reviewed_look'
    if state.exists():
        if _read(state / 'state.json').get('progress', {}).get('summary') != str(source):
            raise ValueError('Study state belongs to another study')
        targets.add(state)
    if log is not None:
        log = Path(log).absolute()
        if log.parent != runs / 'logs' or not log.name.startswith('look-reviewed_'):
            raise ValueError('Invalid old log')
        # The stop audit records the exact command and summary before shutdown.
        stop = _read(runs / 'maintenance' / f'joint_protocol_stop_{study}.json')
        if study not in json.dumps(stop):
            raise ValueError('Missing matching stop audit')
        bank_ids = {v.split("/")[0] for values in summary.get("shared_pca_sources", {}).values() for v in values}
        log_text = log.read_text()
        if not bank_ids or not all(bank in log_text for bank in bank_ids):
            raise ValueError("Log does not identify all PCA banks of the stopped study")
        targets.add(log)
    banks = {s.split('/')[0] for values in summary.get('shared_pca_sources', {}).values() for s in values}
    for bank in banks:
        if len(bank) != 16 or any(c not in '0123456789abcdef' for c in bank):
            raise ValueError('Invalid PCA bank ID')
        path = runs / 'pca' / bank
        if not path.exists():
            continue
        manifest = _read(path / 'bank_manifest.json')
        if manifest['bank_id'] != bank or manifest.get('identity', {}).get('protocol'):
            raise ValueError('PCA does not belong to the incompatible old protocol')
        # Outside the retiring scope, every scientific JSON reference blocks deletion.
        for area in ('experiments', 'sweeps', 'reviewed_look', 'joint_look', 'baseline_selection'):
            for reference in (runs / area).rglob('*.json'):
                if any(_inside(reference, p) for p in targets):
                    continue
                if bank in reference.read_text():
                    raise ValueError(f'PCA bank is externally referenced: {reference}')
        targets.add(path)
    roots = (runs, cache / 'pipeline_state')
    records = [dict(path=str(p), bytes=validate_target(p, roots)) for p in sorted(targets)]
    assert_no_writers(targets)
    evidence = []
    for eid in sorted(eids):
        root = runs / 'experiments' / eid
        item = dict(experiment_id=eid, files={})
        for name in ('experiment_manifest.json', 'validation_result.json', 'experiment_result.json'):
            p = root / name
            if p.exists():
                value = _read(p)
                item['files'][name] = dict(sha256=file_sha256(p), content=value)
        # Preserve completed and partial search aggregate scores including negative results.
        item['search'] = {str(p.relative_to(root)): _read(p) for p in root.glob('look/**/search_history.json')}
        item['factor_selection'] = {str(p.relative_to(root)): _read(p) for p in root.glob('look/*/factor_selection.json')}
        evidence.append(item)
    return dict(status='planned', study=study,
        reason='replace fusion-only mandatory corrections, flattened-coordinate statistics and probability ranking with joint sequential optional corrections and logit ranking',
        source_summary=summary, plans=plan_records, experiment_ids=sorted(eids), paths=records,
        prior_results=evidence, test_access=False,
        preserves=['three backbone checkpoints', 'baseline selection evidence', 'datasets and preprocessing', 'generators', 'environment', 'source history', 'existing maintenance audits'])


def execute_cleanup(report, runs, cache):
    runs, cache = Path(runs).absolute(), Path(cache).absolute()
    if report['study'] != TARGET_STUDY:
        raise ValueError('Wrong study')
    path = runs / 'maintenance' / f'joint_protocol_cleanup_{TARGET_STUDY}.json'
    targets = [Path(r['path']) for r in report['paths']]
    for target in targets:
        validate_target(target, (runs, cache / 'pipeline_state'))
    assert_no_writers(targets)
    report.update(started_at_utc=utc_now(), status='deleting')
    atomic_write_json(report, path)
    for target in targets:
        validate_target(target, (runs, cache / 'pipeline_state'))
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
    registry_path = runs / 'experiment_registry.json'
    if registry_path.exists():
        registry = _read(registry_path)
        report['removed_registry_records'] = [r for r in registry['experiments'] if r['experiment_id'] in report['experiment_ids']]
        registry['experiments'] = [r for r in registry['experiments'] if r['experiment_id'] not in report['experiment_ids']]
        atomic_write_json(registry, registry_path)
    report.update(status='complete', completed_at_utc=utc_now(), absent_after=[str(p) for p in targets if not p.exists()])
    atomic_write_json(report, path)
    return path
