"""Single-final stage comparisons, explicitly distinct from progressive LOOK."""
import csv
from pathlib import Path
import numpy as np
from look.runtime.state import atomic_write_json,file_sha256
from look.analysis.observed_report import simultaneous_bootstrap
from look.analysis.linear_labels import METHOD_EXPLANATION, method_label
from look.methods.linear_operator import ARMS

COMPARISONS=[['rrr_shared_intercept','shared_pca_ridge'],['residual_rrr','rrr_shared_intercept'],
             ['residual_rrr','shared_pca_ridge'],*[ [m,'host'] for m in ARMS],['single_final','host']]


def report(records,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);arrays={};rows=[]
    for r in records:
        if file_sha256(r['path'])!=r['sha256']:raise ValueError('Changed predictions')
        rows.append(dict(method=r['method'],scenario=r['scenario'],macro_f1=r['metrics']['macro_f1']))
        if r['scenario'] not in ('oct_missing','cfp_missing'):continue
        with np.load(r['path'],allow_pickle=False) as f:a={k:f[k] for k in f.files}
        key=(r['method'],r['scenario'])
        if key in arrays:raise ValueError('Duplicate view')
        arrays[key]=a
    needed={(m,p) for pair in COMPARISONS for m in pair for p in ('oct_missing','cfp_missing')}
    if not needed.issubset(arrays):raise ValueError('Incomplete matched stage')
    keys=sorted(arrays);ref=arrays[keys[0]]
    for a in arrays.values():
        if any(not np.array_equal(a[k],ref[k]) for k in ('participant_ids','labels')):raise ValueError('Unpaired participants')
    weights=[];definitions=[]
    for first,second in COMPARISONS:
        for scenario in ('oct_missing','cfp_missing','missing_average'):
            ps=('oct_missing','cfp_missing') if scenario=='missing_average' else (scenario,)
            w=np.zeros(len(keys))
            for p in ps:w[keys.index((first,p))]+=1/len(ps);w[keys.index((second,p))]-=1/len(ps)
            weights.append(w);definitions.append(dict(first=first,second=second,scenario=scenario))
    stats=simultaneous_bootstrap(ref['labels'],np.stack([arrays[k]['logits'].argmax(1) for k in keys]),weights,10000,7341618)
    stats.update(definitions=definitions,family='21_terminal_stage_contrasts_not_global_LOOK',test_access=False)
    atomic_write_json(stats,out/'paired_statistics.json');atomic_write_json(records,out/'sources.json')
    with (out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['method','scenario','macro_f1']);w.writeheader();w.writerows(rows)
    lines=['# Accepted-host single-final mechanism stage','',
        'These are final host classification scores; higher macro-F1 is better. Differences are first minus second.',
        'This is a single terminal-node comparison, not full progressive LOOK or a spatial-reduction experiment.',
        'PCA selects its candidate rank by the original development grid and lambda by train GCV; all three maps share that rank/lambda.',
        'The original on/off-selected single_final result is separate from the candidate map even when disabled. No positive-result acceptance gate.',
        'All fitting uses full unaugmented train. All map predictions replay the complete development MHD forward.',
        'Intervals are conditional on these development-selected models. One seed cannot establish multi-seed stability. No test access.',
        '', METHOD_EXPLANATION, '', '| Method | Input | Macro-F1 (%) |','|---|---|---|']
    lines += [f"| {method_label(r['method'])} | {r['scenario']} | {100*r['macro_f1']:.3f} |" for r in rows]
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return stats
