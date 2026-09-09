"""Immutable representative evidence, without marking the historical study complete."""
from pathlib import Path
from look.runtime.state import atomic_write_json, stable_hash, file_sha256, utc_now
from look.studies.starts import read_json, file_record, verify_file, audit_case, choose_start, PATTERNS

PROTOCOL = 'representative_start_evidence_v1'
CONTEXTS = [f'normalized_mean_layer3_{s}' for s in (3407, 3408, 3409)]


def verify_records(value, seen=None):
    seen = set() if seen is None else seen
    if isinstance(value, dict):
        if {'path', 'sha256', 'bytes'} <= value.keys():
            key = (value['path'], value['sha256'], value['bytes'])
            if key not in seen:
                verify_file(value); seen.add(key)
        for item in value.values(): verify_records(item, seen)
    elif isinstance(value, list):
        for item in value: verify_records(item, seen)


def validate_scope(manifest):
    if (manifest.get('protocol') != PROTOCOL or manifest.get('status') != 'evidence_complete'
            or manifest.get('test_access') is not False):
        raise ValueError('Not sealed representative evidence')
    if sorted(manifest.get('sources',{})) != sorted(CONTEXTS):
        raise ValueError('Three pinned source inventories required')
    rows = manifest['completed_cases']
    if len(rows) != 27 or sorted(manifest['selected_contexts']) != sorted(CONTEXTS):
        raise ValueError('Representative evidence requires 27 cases and three contexts')
    from look.studies.starts import sites_for_fusion
    sites = sites_for_fusion('layer3')
    expected = {f'{ctx}__start_{i:02d}_{site}' for ctx in CONTEXTS for i, site in enumerate(sites, 1)}
    if set(rows) != expected: raise ValueError('Unexpected representative case IDs')
    for key, row in rows.items():
        ordinal = row['start_ordinal']
        if (row['case_id'] != key or row['context'] not in CONTEXTS
                or row['context'] != f"normalized_mean_layer3_{row['seed']}"
                or row['fusion_position'] != 'layer3' or row['filling'] != 'normalized_mean'
                or row['candidate_sites'] != sites or row['eligible_sites'] != sites[ordinal-1:]
                or row['allowed_start'] != sites[ordinal-1]):
            raise ValueError('Representative topology or identity mismatch')


def load_evidence(path, *, deep=True):
    manifest = read_json(Path(path)); validate_scope(manifest)
    if deep:
        verify_records(manifest)
        for row in manifest['completed_cases'].values():
            audited = audit_case(row, Path(row['result']['path']))
            if audited != row: raise ValueError('Stored case differs from original audited evidence')
        for context, rec in manifest['selected_contexts'].items():
            result = read_json(Path(rec['path']))
            frozen_path = Path(rec['path']).with_name('frozen_selection.json')
            frozen = read_json(frozen_path)
            if result.get('test_access') is not False or result.get('status') != 'complete':
                raise ValueError('Invalid selected-start evaluation')
            if result['frozen_selection_sha256'] != file_sha256(frozen_path):
                raise ValueError('Selected-start freeze changed')
            rows = [r for r in manifest['completed_cases'].values() if r['context'] == context]
            for pattern in PATTERNS:
                best = choose_start([dict(start_ordinal=r['start_ordinal'], macro_f1=r['banks'][pattern]['macro_f1'], record=r) for r in rows])['record']
                selected = frozen['directions'][pattern]
                if selected['case_id'] != best['case_id'] or selected['result'] != best['result']:
                    raise ValueError('Selected start does not match fixed tie rule')
    return manifest


def freeze_evidence(suffix_path, destination):
    suffix_path, destination = Path(suffix_path), Path(destination)
    if destination.exists(): return load_evidence(destination)
    suffix = read_json(suffix_path)
    if suffix.get('test_access') is not False: raise ValueError('Test must remain sealed')
    records = {k:v for k,v in suffix['completed_cases'].items() if v['context'] in CONTEXTS}
    selected = {k:v for k,v in suffix['selected_contexts'].items() if k in CONTEXTS}
    inventory=read_json(suffix_path.with_name('source_inventory.json'))
    sources={k:inventory['sources'][k] for k in CONTEXTS}
    frozen_files = []
    for rec in selected.values():
        root = Path(rec['path']).parent
        frozen_files += [file_record(root/'frozen_selection.json')]
        frozen_files += [file_record(p) for p in sorted((root/'predictions').glob('*.npz'))]
    manifest = dict(protocol=PROTOCOL, status='evidence_complete', test_access=False,
        identity=dict(protocol=PROTOCOL, historical_identity=stable_hash(suffix['identity']), contexts=CONTEXTS),
        created_at_utc=utc_now(), expected_cases=27, completed_cases=records, selected_contexts=selected,
        sources=sources, selected_artifacts=frozen_files, historical_summary=file_record(suffix_path),
        historical_completed=len(suffix['completed_cases']), historical_planned=243,
        interpretation='Post-development budget revision. Other starts deferred, not failed or zero. Selection remains validation-only.')
    validate_scope(manifest); verify_records(manifest)
    atomic_write_json(manifest, destination)
    return load_evidence(destination)
