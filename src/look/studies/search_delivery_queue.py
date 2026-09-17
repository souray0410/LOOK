"""Compile existing five-case sequence into independent shared-dispatch tasks.

No training, claims, allocation or owner replacement occurs here. Operators must
drain the old sequence controller and reconcile live claims before activation.
"""
import argparse
from pathlib import Path
from look.runtime.state import atomic_write_json,file_sha256,stable_hash
from look.runtime.delivery_policy import read
from look.studies.search_protocol import validate,representative_starts,TREE_MODE


def compile_sequence(sequence,output):
    sequence=Path(sequence);config=read(sequence);tasks=[];entries=[];specs=[];seen=set()
    for item in config['tasks']:
        if file_sha256(item['config'])!=item['sha256']:raise ValueError('Sequence configuration changed')
        launch=read(item['config']);task=launch['task'];spec=read(task['spec']);validate(spec)
        if file_sha256(task['spec'])!=task['spec_sha256']:raise ValueError('Search specification changed')
        run=str(Path(task['run_dir']).resolve())
        if run in seen:raise ValueError('Duplicate search execution')
        seen.add(run);specs.append(spec)
        tasks.append(dict(task,execution='look_search',search_mode=spec['mode']))
        entries.append(dict(spec=task['spec'],spec_sha256=task['spec_sha256'],run_dir=task['run_dir'],
            execution='look_search',role='delivery' if spec['mode']=='best_forward' else 'matched_control',dependencies=[]))
    if not specs:raise ValueError('Empty search package')
    h=specs[0]['host'];starts=representative_starts(h['architecture'],h['position'])
    expected={('best_forward',1)}|{('greedy',s) for s in starts}
    if len(specs)!=len(expected) or {(s['mode'],s.get('start_ordinal',1)) for s in specs}!=expected:
        raise ValueError('Incomplete representative search package')
    for spec in specs:
        if any(spec[k]!=specs[0][k] for k in ('host','source','pca','spatial_factors','latent_dims')):
            raise ValueError('Unmatched host or representation')
        if spec['host']['seed']!=3416:raise ValueError('First complete package must use seed3416')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    feed=dict(schema='look_search_work_feed_v1',tasks=tasks,test_access=False)
    policy=dict(schema='research_delivery_policy_v1',tasks=entries,test_access=False)
    for name,value in [('feed.json',feed),('delivery_policy.json',policy)]:
        path=out/name
        if path.exists() and read(path)!=value:raise ValueError('Immutable delivery registration changed')
        if not path.exists():atomic_write_json(value,path)
    receipt=dict(state='prepared_not_activated',sequence=str(sequence),sequence_sha256=file_sha256(sequence),
        tasks=len(tasks),test_access=False,requirements=['old_sequence_controller_drained','live_claims_reconciled',
        'immutable_scientific_sources_verified','resource_acceptance','existing_dispatcher_bound',
        'weekly_package_gate_preserved','real_downstream_progress'])
    atomic_write_json(receipt,out/'registration.json');return receipt


def register_tree_case(spec_path,run_dir,output):
    """Prepare one explicitly named first-seed tree task, outside the old matrix.

    This does not activate a dispatcher, release repeats, or certify a matched
    weekly package. Registration never rewrites an existing execution identity.
    """
    spec_path=Path(spec_path).resolve();spec=read(spec_path);validate(spec)
    if spec['mode']!=TREE_MODE or spec['host']['seed']!=3416:
        raise ValueError('Independent tree registration requires positive_forward_tree seed3416')
    run=Path(run_dir).resolve()
    if (run/'spec.json').exists() and read(run/'spec.json')!=spec:
        raise ValueError('Existing execution has a different scientific identity')
    task=dict(id='search/'+TREE_MODE+'/'+stable_hash(spec),spec=str(spec_path),
        spec_sha256=file_sha256(spec_path),run_dir=str(run),execution='look_search',search_mode=TREE_MODE)
    entry={k:task[k] for k in ('spec','spec_sha256','run_dir','execution')}
    entry.update(role='delivery',dependencies=[])
    feed=dict(schema='look_search_work_feed_v1',tasks=[task],test_access=False)
    policy=dict(schema='research_delivery_policy_v1',tasks=[entry],test_access=False)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    values=[('feed.json',feed),('delivery_policy.json',policy)]
    # Check all existing bindings before writing either half of a registration.
    for name,value in values:
        if (out/name).exists() and read(out/name)!=value:
            raise ValueError('Immutable delivery registration changed')
    for name,value in values:
        if not (out/name).exists():atomic_write_json(value,out/name)
    receipt=dict(state='prepared_not_activated',kind='independent_first_seed_tree',tasks=1,
        spec_sha256=task['spec_sha256'],test_access=False,weekly_package_released=False,
        requirements=['immutable_scientific_sources_verified','resource_acceptance',
            'existing_dispatcher_bound','weekly_package_gate_preserved','real_downstream_progress'])
    atomic_write_json(receipt,out/'registration.json');return receipt


def refresh(sequence,output):
    """Existing maintenance invokes this after completion; failures remain visible."""
    from look.analysis.search_delivery import report
    from look.analysis.search_report import report as matched
    from look.analysis.cumulative_delivery import collect,publish
    from look.studies.search_case import verify_case
    config=read(sequence);runs=[];specs=[];errors=[]
    for item in config['tasks']:
        if file_sha256(item['config'])!=item['sha256']:raise ValueError('Sequence changed')
        task=read(item['config'])['task'];run=Path(task['run_dir'])
        if file_sha256(task['spec'])!=task['spec_sha256']:raise ValueError('Registered spec changed')
        specs.append(read(task['spec']));runs.append(str(run))
        if (run/'accepted.json').exists():
            try:verify_case(run,specs[-1]);report(run)
            except Exception as error:errors.append(dict(run=str(run),error=repr(error)))
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # Failed acceptance must not overwrite the last good cumulative publication.
    if errors:
        result=dict(state='needs_review',errors=errors,test_access=False)
        atomic_write_json(result,out/'refresh_status.json');return result
    value=collect(sequence,verify_case);publication=publish(value,out/'publication')
    complete=all(c['state']=='accepted_delivery' for c in value['cases']) and bool(runs)
    if complete:
        matched(dict(host=specs[0]['host'],runs=runs,output=str(out/'matched_report'),test_access=False))
    result=dict(state='matched_delivery_accepted' if complete else 'partial_delivery',
                publication=publication,test_access=False)
    atomic_write_json(result,out/'refresh_status.json');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence');p.add_argument('--output',required=True)
    p.add_argument('--tree-spec');p.add_argument('--run-dir')
    p.add_argument('--refresh',action='store_true');a=p.parse_args()
    if a.tree_spec:
        if a.sequence or a.refresh or not a.run_dir:p.error('--tree-spec requires --run-dir and cannot use --sequence/--refresh')
        register_tree_case(a.tree_spec,a.run_dir,a.output)
    else:
        if not a.sequence or a.run_dir:p.error('--sequence required unless registering --tree-spec')
        (refresh if a.refresh else compile_sequence)(a.sequence,a.output)


if __name__=='__main__':main()
