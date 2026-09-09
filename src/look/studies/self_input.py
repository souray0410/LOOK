"""Three self-input cases after the immutable 15-case method study. Test sealed."""
from pathlib import Path
import time
import torch
from look.runtime.state import atomic_write_json, stable_hash, utc_now, PipelineState
from look.studies.starts import read_json, file_record, verify_file, verify_source
from look.runtime.provenance import implementation_sha256
from look.studies.methods import run_method_case
from look.methods.self_input import POLICY


def make_cases(spec):
    if (spec['protocol'] != 'self_input_missing_only_shared_pca_v1'
        or spec['methods'] != [POLICY] or spec['seeds'] != [3407,3408,3409]
        or spec['patterns'] != ['oct_missing','cfp_missing']
        or spec['fusion_position'] != 'layer3' or spec['filling'] != 'normalized_mean'
        or spec['random_ratios'] != [.2,.4,.6,.8,1.] or spec['missingness_seed'] != 3407
        or spec['expected_new_cases'] != 3 or spec['expected_direction_fits'] != 6
        or spec['test_access'] is not False):
        raise ValueError('Unexpected self-input study scope')
    return [dict(case_id=f'{POLICY}_layer3_normalized_mean_{seed}',method=POLICY,
        fusion_position='layer3',filling='normalized_mean',seed=seed) for seed in spec['seeds']]


def verify_predecessor(summary):
    if (summary.get('status') != 'complete' or summary.get('test_access') is not False
        or summary.get('spec',{}).get('protocol') != 'frozen_look_method_evidence_v2'
        or len(summary.get('completed_cases',{})) != 15):
        raise RuntimeError('The original 15 method cases must complete before self-input cases')
    for seed in [3407,3408,3409]:
        key=f'missing_only_layer3_normalized_mean_{seed}'
        rec=summary['completed_cases'][key];verify_file(rec)
        r=read_json(Path(rec['path']))
        if r['status']!='complete' or r['test_access'] is not False or r['case']['seed']!=seed or r['case']['method']!='missing_only':
            raise RuntimeError('Invalid paired missing-only reference')
        for artifact in r['provenance']:verify_file(artifact)


def run_self_input_study(paths, predecessor_path, devices, *, execute=False, wait=False):
    predecessor_path=Path(predecessor_path).resolve()
    spec_path=paths.project_root/'configs/self_input_evidence.json';spec=read_json(spec_path)
    cases=make_cases(spec);parent=read_json(predecessor_path)
    if parent.get('test_access') is not False:raise ValueError('Predecessor test must remain sealed')
    identity=dict(protocol=spec['protocol'],spec_sha256=file_record(spec_path)['sha256'],
        predecessor_path=str(predecessor_path),predecessor_identity=stable_hash(parent['identity']),
        implementation_sha256=implementation_sha256(paths.project_root),devices=list(devices))
    output=paths.runs_root/'self_input_study'/stable_hash(identity)[:12]
    protected = predecessor_path.parent if parent.get('start_scope')=='representative' else predecessor_path.parents[2]
    if output.resolve().is_relative_to(protected):
        raise ValueError('Self-input outputs require a separate release run root')
    summary=dict(identity=identity,spec=spec,cases=cases,status='planned',output=str(output),
        predecessor_summary=str(predecessor_path),completed_cases={},expected_new_cases=3,
        expected_direction_fits=6,combined_development_stages=None if parent.get('combined_development_stages') is None else parent['combined_development_stages']+3,test_access=False)
    if not execute:return summary
    output.mkdir(parents=True,exist_ok=True);path=output/'summary.json'
    with PipelineState(output/'state','self_input',identity,[spec_path]) as state:
        if path.exists():
            summary=read_json(path)
            if summary['identity']!=identity:raise RuntimeError('Self-input resume identity changed')
        def report(status,**fields):
            summary.update(dict(status=status,updated_at_utc=utc_now(),error=None)|fields)
            atomic_write_json(summary,path)
            print(f'{utc_now()} self_input_status={status} completed={len(summary["completed_cases"])}/3 {fields}',flush=True)
        try:
            report('waiting_predecessor')
            while parent['status']!='complete':
                if parent['status'] in ('failed','cancelled'):raise RuntimeError('Predecessor failed; no automatic restart')
                if not wait:raise RuntimeError('Predecessor incomplete; --wait-predecessor required')
                time.sleep(30);parent=read_json(predecessor_path)
                if stable_hash(parent['identity'])!=identity['predecessor_identity']:
                    raise RuntimeError('Predecessor identity changed while waiting')
            if stable_hash(parent['identity'])!=identity['predecessor_identity']:
                raise RuntimeError('Predecessor identity changed')
            verify_predecessor(parent)
            inventory_path=predecessor_path.parent/'source_inventory.json'
            freeze_path=predecessor_path.parent/'comparison_freeze.json'
            freeze=read_json(freeze_path);verify_file(freeze['source_inventory'])
            if freeze['source_inventory']['path']!=str(inventory_path):raise RuntimeError('Wrong source inventory')
            if freeze['method_results']!=parent['completed_cases']:raise RuntimeError('Predecessor freeze mismatch')
            inv=read_json(inventory_path)
            for rec in inv['upstream_records']+inv['source_files']:verify_file(rec)
            for source in inv['sources'].values():verify_source(source)
            provenance=dict(predecessor=file_record(predecessor_path),freeze=file_record(freeze_path),
                inventory=file_record(inventory_path),paired_references={str(s):parent['completed_cases'][f'missing_only_layer3_normalized_mean_{s}'] for s in spec['seeds']})
            pin_path=output/'predecessor_freeze.json'
            if pin_path.exists() and read_json(pin_path)!=provenance:raise RuntimeError('Pinned predecessor changed')
            atomic_write_json(provenance,pin_path)
            for case in cases:
                report('running',active_case=case['case_id'])
                source=inv['sources'][str(case['seed'])]
                result=run_method_case(source,case,spec,output/'cases'/case['case_id'],torch.device('cuda:0'),devices)
                summary['completed_cases'][case['case_id']]=file_record(output/'cases'/case['case_id']/'validation_result.json')
                report('running',active_case=case['case_id']);state.checkpoint(last_case=case['case_id'])
            atomic_write_json(dict(identity=identity,spec=spec,test_access=False,predecessor=provenance,
                results=summary['completed_cases'],interpretation='Freeze before later sealed-test evaluation; development comparisons only'),output/'comparison_freeze.json')
            report('complete',active_case=None)
            state.complete([path,pin_path,output/'comparison_freeze.json'])
        except BaseException as e:
            report('failed',error=repr(e));raise
    return summary
