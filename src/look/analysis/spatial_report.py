"""All three pairwise comparisons with shared participant bootstrap indices."""
from itertools import combinations
from pathlib import Path
import csv
import numpy as np
from look.runtime.state import atomic_write_json,file_sha256
from look.analysis.observed_report import simultaneous_bootstrap
from look.studies.spatial_protocol import ROUTES,protocol


def report(records,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    arrays={};rows=[]
    for record in records:
        path=Path(record['path'])
        if file_sha256(path)!=record['sha256']:raise ValueError('Changed spatial predictions')
        with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
        key=(record['method'],record['scenario'])
        if key in arrays:raise ValueError('Duplicate spatial view')
        arrays[key]=a
        rows.append(dict(method=key[0],scenario=key[1],macro_f1=record['metrics']['macro_f1']))
    required={(m,p) for m in ROUTES for p in ('oct_missing','cfp_missing')}
    if not required.issubset(arrays):raise ValueError('Incomplete three-route comparison')
    ref=next(iter(arrays.values()))
    for a in arrays.values():
        if not np.array_equal(a['participant_ids'],ref['participant_ids']) or not np.array_equal(a['labels'],ref['labels']):
            raise ValueError('Unpaired spatial evidence')
    keys=list(arrays);contrasts=[];definitions=[]
    for first,second in combinations(ROUTES,2):
        for scenario in ('oct_missing','cfp_missing','missing_average'):
            w=np.zeros(len(keys));patterns=('oct_missing','cfp_missing') if scenario=='missing_average' else (scenario,)
            for p in patterns:
                w[keys.index((first,p))]+=1/len(patterns);w[keys.index((second,p))]-=1/len(patterns)
            contrasts.append(w);definitions.append(dict(first=first,second=second,scenario=scenario,direction='first_minus_second'))
    p=protocol()
    stats=simultaneous_bootstrap(ref['labels'],np.stack([arrays[k]['logits'].argmax(1) for k in keys]),contrasts,p['bootstrap_iterations'],p['bootstrap_seed'])
    stats.update(definitions=definitions,family='nine_pairwise_contrasts_within_one_host',test_access=False)
    atomic_write_json(stats,out/'paired_statistics.json');atomic_write_json(records,out/'sources.json')
    with (out/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['method','scenario','macro_f1']);writer.writeheader();writer.writerows(rows)
    lines=['# LOOK spatial preprocessing — matched development evidence','',
        'All routes fit train only; the host is frozen. Each route has independent development selection under the same node/rank/ridge rules.',
        'Macro-F1 is final host prediction (higher is better), not a branch average. Differences are first minus second.',
        'Direct PCA has no spatial-factor search; interpolation and averaging each have three matching factors. This is not equal compute or identical search size.',
        'Averaging loses within-window detail; interpolation can alias; PCA preserves variance rather than guaranteed discriminative information.',
        'Pilot acceptance depends on integrity and completeness, never effect sign or significance. Intervals are conditional on development selection.',
        'No test was accessed. A single seed is not the final three-seed conclusion.','',
        '| Method | Input | Macro-F1 (%) |','|---|---|---|']
    lines += [f"| {r['method']} | {r['scenario']} | {100*r['macro_f1']:.3f} |" for r in rows]
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return stats


def group_report(runs,out):
    """Three complete seeds; average model F1s, never stack participants as independent."""
    from look.studies.project_case import read
    from look.studies.spatial_case import verify_case
    out=Path(out);out.mkdir(parents=True,exist_ok=True);seen=set();keys=[];pred=[];rows=[];reference=None;receipts={}
    for run in runs:
        root=Path(run);spec=read(root/'spec.json');verify_case(root,spec);seed=spec['host']['seed']
        if seed in seen:raise ValueError('Duplicate seed in spatial group')
        seen.add(seed);receipts[str(root)]=file_sha256(root/'accepted.json')
        for r in read(root/'development/suite.json')['records']:
            if r['method'] not in ROUTES or r['scenario'] not in ('oct_missing','cfp_missing'):continue
            if file_sha256(r['path'])!=r['sha256']:raise ValueError('Changed group predictions')
            with np.load(r['path'],allow_pickle=False) as a:
                if reference is None:reference={k:a[k].copy() for k in ('participant_ids','labels')}
                if any(not np.array_equal(reference[k],a[k]) for k in reference):raise ValueError('Seeds are not participant paired')
                pred.append(a['logits'].argmax(1));keys.append((seed,r['method'],r['scenario']))
                rows.append(dict(seed=seed,method=r['method'],scenario=r['scenario'],macro_f1=r['metrics']['macro_f1']))
    if seen!={3416,3417,3418} or len(keys)!=18:raise ValueError('Incomplete seed group')
    weights=[];definitions=[]
    for a,b in combinations(ROUTES,2):
        for scenario in ('oct_missing','cfp_missing','missing_average'):
            w=np.zeros(len(keys));patterns=('oct_missing','cfp_missing') if scenario=='missing_average' else (scenario,)
            for seed in sorted(seen):
                for pattern in patterns:
                    w[keys.index((seed,a,pattern))]+=1/(3*len(patterns));w[keys.index((seed,b,pattern))]-=1/(3*len(patterns))
            weights.append(w);definitions.append(dict(first=a,second=b,scenario=scenario))
    stats=simultaneous_bootstrap(reference['labels'],np.stack(pred),weights,10000,7341618)
    stats.update(definitions=definitions,family='one_three_seed_host_group_not_whole_study',test_access=False)
    atomic_write_json(stats,out/'paired_statistics.json');atomic_write_json(rows,out/'per_seed.json')
    aggregate=[]
    for method in ROUTES:
        for scenario in ('oct_missing','cfp_missing'):
            v=[r['macro_f1'] for r in rows if r['method']==method and r['scenario']==scenario]
            aggregate.append(dict(method=method,scenario=scenario,mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1))))
    atomic_write_json(aggregate,out/'mean_sd.json');atomic_write_json(receipts,out/'inputs.json')
