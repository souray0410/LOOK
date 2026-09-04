#!/usr/bin/env python3
"""Remove only the experiments named by a stopped reviewed LOOK study manifest."""
import argparse
import json
import os
from pathlib import Path
import shutil


def main():
    from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
    from look_core.state import atomic_write_json, stable_hash, utc_now
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    paths = resolve_runtime_arguments(args)
    runs = paths.runs_root.resolve()
    source = args.summary.resolve()
    if source.parent.parent != runs / 'reviewed_look' or source.name != 'summary.json':
        raise ValueError('Cleanup requires a reviewed LOOK summary inside the selected runs root')
    report_path = runs / 'maintenance' / f'cleanup_look__{stable_hash(str(source))[:12]}.json'
    if not source.exists() and report_path.exists():
        print('Already cleaned:', report_path)
        return
    summary = json.loads(source.read_text())
    if summary.get('status') == 'running':
        raise RuntimeError('Stop the old job before cleanup')
    plans = {x['plan_id'] for x in summary.get('completed_stages', {}).values()}
    if summary.get('active_plan_id'):
        plans.add(summary['active_plan_id'])
    remove = {source.parent}
    experiment_ids = set()
    for plan_id in plans:
        if len(plan_id) != 12 or any(c not in '0123456789abcdef' for c in plan_id):
            raise ValueError('Invalid plan ID')
        sweep = runs / 'sweeps' / f'validation__{plan_id}'
        plan = json.loads((sweep / 'study_plan.json').read_text())
        if plan['phase'] != 'validation' or not all(p.get('enabled') for p in plan['grid']['look_profiles']):
            raise ValueError('Refusing to delete non-LOOK or test experiments')
        remove.add(sweep)
        for eid in plan['experiment_ids']:
            if Path(eid).name != eid:
                raise ValueError('Invalid experiment ID')
            experiment_ids.add(eid)
            remove.add(runs / 'experiments' / eid)
            remove.add(paths.cache_root / 'pipeline_state' / f'experiment__{eid}__validation')
    state_dir = paths.cache_root / 'pipeline_state' / 'reviewed_look'
    if state_dir.exists():
        state = json.loads((state_dir / 'state.json').read_text())
        if state.get('progress', {}).get('summary') != str(source):
            raise RuntimeError('Reviewed LOOK state belongs to another job')
        remove.add(state_dir)
    if args.log:
        log = args.log.resolve()
        if log.parent != runs / 'logs' or not log.name.startswith('look-reviewed_'):
            raise ValueError('Only an explicit reviewed LOOK log may be removed')
        remove.add(log)
    for path in remove:
        if path.is_symlink():
            raise ValueError(f'Refusing symlink: {path}')
        for lock in path.rglob('stage.lock') if path.is_dir() else []:
            pid = int(json.loads(lock.read_text())['pid'])
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            raise RuntimeError(f'Live writer PID {pid}: {lock}')
    targets = [dict(path=str(p), bytes=sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                    if p.is_dir() else p.stat().st_size if p.exists() else 0)
               for p in sorted(remove)]
    report = dict(status='planned', reason='replace repeated per-pattern PCA with shared complete PCA',
        source_summary=str(source), paths=targets, experiment_ids=sorted(experiment_ids),
        preserves=['backbones', 'baseline_selection', 'dataset', 'preprocessed_pairs', 'generators'])
    if not args.execute:
        print(json.dumps(report, indent=2))
        return
    atomic_write_json(report, report_path)
    for p in sorted(remove):
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)
    registry_path = runs / 'experiment_registry.json'
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        registry['experiments'] = [r for r in registry['experiments'] if r['experiment_id'] not in experiment_ids]
        atomic_write_json(registry, registry_path)
    report.update(status='complete', completed_at_utc=utc_now())
    atomic_write_json(report, report_path)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
