"""Compile existing five-case sequence into independent shared-dispatch tasks.

No training, claims, allocation or owner replacement occurs here. Operators must
drain the old sequence controller and reconcile live claims before activation.
"""
import argparse
from pathlib import Path
from look.runtime.state import atomic_write_json,file_sha256
from look.runtime.delivery_policy import read
from look.studies.search_protocol import validate,representative_starts


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
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);p.add_argument('--output',required=True)
    p.add_argument('--refresh',action='store_true');a=p.parse_args()
    (refresh if a.refresh else compile_sequence)(a.sequence,a.output)


if __name__=='__main__':main()
