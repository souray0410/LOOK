#!/usr/bin/env python3
"""Read only: scoped LOOK bank progress, completed comparisons, and GPU state."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--runs-root', type=Path)
    parser.add_argument('--session', default='look-joint')
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((args.project_root / 'project.json').read_text())
    runs = args.runs_root or Path(cfg['deployment_defaults']['data_root']) / cfg['directories']['runs']
    active = subprocess.run(['tmux', 'has-session', '-t', args.session], capture_output=True).returncode == 0
    print(f'session={args.session} running={active}')
    summaries = sorted((runs / 'joint_look').glob('*/summary.json'), key=lambda p: p.stat().st_mtime)
    if summaries:
        summary = json.loads(summaries[-1].read_text())
        print(f'summary={summaries[-1]}')
        for key in ('status', 'stage', 'next_stage', 'active_seed', 'updated_at_utc', 'active_plan_id', 'error', 'test_access'):
            print(f'{key}={summary.get(key)}')
        pca_progress = sorted((runs / 'pca').glob('*/progress.json'), key=lambda p: p.stat().st_mtime)
        for path in pca_progress[-3:]:
            print('shared_pca=' + json.dumps(json.loads(path.read_text())))
        plans = {s['plan_id'] for s in summary['completed_stages'].values()}
        if summary.get('active_plan_id'):
            plans.add(summary['active_plan_id'])
        for plan_id in sorted(plans):
            sweep = runs / 'sweeps' / f'validation__{plan_id}'
            plan = json.loads((sweep / 'study_plan.json').read_text())
            completed = 0
            for experiment_id in plan['experiment_ids']:
                root = runs / 'experiments' / experiment_id
                result = root / 'validation_result.json'
                if result.exists():
                    completed += 1
                    data = json.loads(result.read_text())
                    print(f'COMPLETED {experiment_id}')
                    for condition, metrics in data.get('validation', {}).items():
                        if isinstance(metrics, dict) and 'macro_f1' in metrics:
                            print(f"  {condition}: AUROC={metrics.get('macro_auroc_ovr', float('nan')):.4f} F1={metrics['macro_f1']:.4f}")
                elif root.exists():
                    print(f'IN PROGRESS {experiment_id}')
                    for bank in sorted((root / 'look').glob('*/factors/x*')):
                        decisions = [json.loads(p.read_text()) for p in sorted((bank / 'decisions').glob('*.json'))]
                        n = sum(d['enabled'] for d in decisions)
                        print(f'  decisions={len(decisions)} enabled={n}')
                        record = bank / 'bank_complete.json'
                        if record.exists():
                            d = json.loads(record.read_text())
                            print(f"  {bank.relative_to(root)} nodes={n} AUROC={d['primary_score']:.4f}")
                        else:
                            history = bank / 'search_history.json'
                            rows = json.loads(history.read_text()) if history.exists() else []
                            last_node = decisions[-1]['identity']['node'] if decisions else 'none yet'
                            print(f'  {bank.relative_to(root)} nodes={n} last_completed_node={last_node}')
            print(f'plan={plan_id} completed={completed}/{len(plan["experiment_ids"])}')
    print('GPU index, utilization, memory used, memory total:', flush=True)
    subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'])
    logs = sorted((runs / 'logs').glob(f'{args.session}_*.log'), key=lambda p: p.stat().st_mtime)
    if logs:
        print(f'log={logs[-1]}', flush=True)
        if args.follow:
            subprocess.run(['tail', '-n', '15', '-F', str(logs[-1])])


if __name__ == '__main__':
    main()
