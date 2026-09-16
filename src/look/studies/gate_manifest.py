"""Explicit read-only import of accepted study evidence to one gate-review schema.

Historical records stay byte-identical. No server-specific runtime readers.
"""
import json
from pathlib import Path
from look.runtime.state import file_sha256, stable_hash

VERSION = 'look_gate_review_v1'
PROTOCOL = dict(schema=VERSION, data_role='development', test_access=False,
    views=['forced_on_mechanism', 'dev_selected_strategy'],
    rule='independent_method_and_missing_pattern_strict_macro_f1_improvement_else_off',
    scope='terminal_map_or_whole_progressive_bank_not_new_per_node_search',
    iterations=10000, bootstrap_seed=7341618,
    selection_bias='conditional_intervals_no_claim_of_removing_dev_selection_bias',
    amendment='added_after_initial_development_results_20260916_not_preregistered')


def import_run(run):
    root = Path(run).resolve()
    spec_path, receipt_path = root/'spec.json', root/'accepted.json'
    spec = json.loads(spec_path.read_text()); receipt = json.loads(receipt_path.read_text())
    from look.methods.linear_operator import ARMS as linear_arms
    from look.methods.affine_family import ARMS as family_arms
    from look.studies.affine_protocol import COMPARISONS
    from look.studies.linear_protocol import protocol as linear_protocol
    schemas = {
        'look_terminal_stage_v1': ('terminal', linear_arms),
        'look_linear_vector_v1': ('progressive', linear_arms),
        'look_affine_family_v1': (spec.get('scope'), family_arms),
    }
    # These are explicit source-format imports; the report consumes only VERSION.
    if spec.get('schema') not in schemas:
        raise ValueError('Unsupported source schema for explicit gate import: '+str(spec.get('schema')))
    scope, methods = schemas[spec['schema']]
    if (scope not in ('terminal', 'progressive') or spec.get('test_access') is not False
            or receipt.get('test_access') is not False or receipt.get('state') != 'accepted'
            or receipt.get('profile') is not False or receipt.get('identity') != stable_hash(spec)
            or receipt.get('schema') != spec['schema']
            or not all(receipt.get(k) is True for k in ('host_frozen', 'reload_exact'))
            or (spec['schema']=='look_terminal_stage_v1' and receipt.get('full_development_mhd_replay') is not True)
            or (spec['schema']!='look_terminal_stage_v1' and receipt.get('rng_restored') is not True)):
        raise ValueError('Source is not accepted sealed-development evidence')
    suite_path = root/'development/suite.json'
    for name, path in [('spec.json', spec_path), ('development/suite.json', suite_path)]:
        if receipt['files'].get(name) != file_sha256(path):
            raise ValueError('Source acceptance digest changed: '+name)
    suite = json.loads(suite_path.read_text())
    if suite.get('test_access') is not False:
        raise ValueError('Test evidence is prohibited')
    records = suite['records']
    required = {(m, p) for m in ('host', *methods) for p in ('oct_missing', 'cfp_missing')}
    if not required.issubset({(r['method'], r['scenario']) for r in records}):
        raise ValueError('Incomplete matched gate evidence')
    source_comparisons = COMPARISONS if spec['schema']=='look_affine_family_v1' else linear_protocol()['comparisons']
    comparisons = [list(p) for p in source_comparisons if set(p) <= set(methods) | {'host'}]
    for m in methods:
        if [m, 'host'] not in comparisons: comparisons.append([m, 'host'])
    body = dict(schema=VERSION, protocol=PROTOCOL, host=spec['host'], scope=scope,
        methods=list(methods), comparisons=comparisons, records=records,
        source=dict(run_dir=str(root), spec_sha256=file_sha256(spec_path),
                    accepted_sha256=file_sha256(receipt_path), suite_sha256=file_sha256(suite_path)))
    return dict(body, identity=stable_hash(body))


def validate_manifest(manifest):
    if (manifest.get('schema') != VERSION or manifest.get('protocol') != PROTOCOL
            or manifest.get('identity') != stable_hash({k:v for k,v in manifest.items() if k!='identity'})):
        raise ValueError('Unregistered gate review manifest')
    s = manifest['source']; root = Path(s['run_dir'])
    for name, key in [('spec.json','spec_sha256'), ('accepted.json','accepted_sha256'),
                      ('development/suite.json','suite_sha256')]:
        if file_sha256(root/name) != s[key]: raise ValueError('Gate source changed')
