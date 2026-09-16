"""Immutable paired forced-on versus independently dev-gated strategy reports."""
import csv
import json
from pathlib import Path
import numpy as np
from look.evaluation.correction_gate import fit_gate, apply_gate, macro_f1, PATTERNS
from look.analysis.observed_report import simultaneous_bootstrap
from look.studies.gate_manifest import validate_manifest
from look.runtime.state import atomic_write_json, file_sha256, stable_hash


def load_records(manifest):
    arrays = {}; records = {}; ref = None
    for r in manifest['records']:
        key = (r['method'], r['scenario'])
        if key in arrays: raise ValueError('Duplicate prediction view')
        if file_sha256(r['path']) != r['sha256']: raise ValueError('Prediction digest changed')
        with np.load(r['path'], allow_pickle=False) as f:
            a = {k:f[k].copy() for k in f.files}
        if ref is None: ref = a
        if any(not np.array_equal(ref[k], a[k]) for k in ('participant_ids', 'labels')):
            raise ValueError('Unpaired evidence')
        if len(np.unique(a['participant_ids'].astype(str))) != len(a['labels']):
            raise ValueError('Predictions must be one row per participant, not per eye')
        score = macro_f1(a['labels'], a['logits'])
        if not np.isclose(score, r['metrics']['macro_f1'], atol=1e-12, rtol=0):
            raise ValueError('Recorded metric does not match predictions')
        arrays[key] = a; records[key] = r
    return arrays, records


def compute(manifest, *, iterations=10000):
    """Retain forced outputs; select one global gate per method/missing direction."""
    arrays, records = load_records(manifest)
    methods = manifest['methods']; identity = manifest['identity']; decisions = {}; chosen = {}; rows = []
    for pattern in PATTERNS:
        host = arrays['host', pattern]
        rows.append(dict(method='host', scenario=pattern, view='baseline',
                         macro_f1=macro_f1(host['labels'], host['logits']), gate_enabled=None))
        for m in methods:
            a = arrays[m, pattern]
            g = fit_gate(a['labels'], a['logits'], host['logits'], data_role='development',
                         identity=identity, method=m, pattern=pattern)
            decisions[m, pattern] = g
            z = apply_gate(a['logits'], host['logits'], g, identity=identity, method=m, pattern=pattern)
            chosen[m, pattern] = dict(a, logits=z)
            for view, value in [('forced_on_mechanism', a['logits']), ('dev_selected_strategy', z)]:
                rows.append(dict(method=m, scenario=pattern, view=view,
                    macro_f1=macro_f1(a['labels'], value), gate_enabled=g['enabled']))
    # Preserve original mixed masks exactly, rather than regenerate or select per ratio/person.
    for m in methods:
        for method, scenario in sorted(arrays):
            if method != m or scenario in PATTERNS: continue
            if scenario == 'complete':
                a = arrays[method, scenario]
                host = arrays['host', 'complete']
                if not np.array_equal(a['logits'], host['logits']): raise ValueError('Complete input changed')
                z = a['logits'].copy()
            elif scenario.startswith('mixed_'):
                a = arrays[method, scenario]; z = a['logits'].copy()
                if 'patterns' not in a or not np.isin(a['patterns'], ['complete', *PATTERNS]).all():
                    raise ValueError('Missing or invalid participant missingness mask')
                for p in ('complete', *PATTERNS):
                    mask = a['patterns'] == p
                    raw = arrays['host', 'complete'] if p == 'complete' else arrays[m, p]
                    if not np.array_equal(a['logits'][mask], raw['logits'][mask]):
                        raise ValueError('Mixed predictions do not match fixed constituent views')
                    selected = raw if p == 'complete' else chosen[m, p]
                    z[mask] = selected['logits'][mask]
                # All methods must share the same participant-level mask.
                other = next(arrays[t, scenario] for t in methods if (t, scenario) in arrays)
                if not np.array_equal(other['patterns'], a['patterns']): raise ValueError('Unmatched mixed masks')
            else: continue
            chosen[m, scenario] = dict(a, logits=z)
            for view, value in [('forced_on_mechanism', a['logits']), ('dev_selected_strategy', z)]:
                rows.append(dict(method=m, scenario=scenario, view=view,
                    macro_f1=macro_f1(a['labels'], value), gate_enabled=None))
    # A single_final column must actually be its dev-selected PCA candidate.
    for p in PATTERNS:
        if ('single_final', p) in arrays:
            if not np.array_equal(arrays['single_final', p]['logits'], chosen['shared_pca_ridge', p]['logits']):
                raise ValueError('Original selected output disagrees with repaired gate')
    keys=[]; predictions=[]
    for p in PATTERNS:
        keys.append(('baseline','host',p)); predictions.append(arrays['host',p]['logits'].argmax(1))
        for m in methods:
            for view, d in [('forced_on_mechanism',arrays[m,p]), ('dev_selected_strategy',chosen[m,p])]:
                keys.append((view,m,p)); predictions.append(d['logits'].argmax(1))
    weights=[];definitions=[]
    for first,second in manifest['comparisons']:
        for scenario in (*PATTERNS,'missing_average'):
            ps=PATTERNS if scenario=='missing_average' else (scenario,);w=np.zeros(len(keys))
            for p in ps:
                for m,v in [(first,1),(second,-1)]:
                    view='baseline' if m=='host' else 'dev_selected_strategy'
                    w[keys.index((view,m,p))]+=v/len(ps)
            weights.append(w);definitions.append(dict(first=first,second=second,scenario=scenario,
                view='dev_selected_strategy',direction='first_minus_second'))
    for m in methods:
        for scenario in (*PATTERNS,'missing_average'):
            ps=PATTERNS if scenario=='missing_average' else (scenario,);w=np.zeros(len(keys))
            for p in ps:
                w[keys.index(('dev_selected_strategy',m,p))]+=1/len(ps)
                w[keys.index(('forced_on_mechanism',m,p))]-=1/len(ps)
            weights.append(w);definitions.append(dict(method=m,scenario=scenario,view='gate_effect',
                direction='dev_selected_strategy_minus_forced_on_mechanism'))
    stats=simultaneous_bootstrap(arrays['host',PATTERNS[0]]['labels'],np.stack(predictions),weights,iterations,7341618)
    stats.update(definitions=definitions,family='gate_amendment_one_host_not_global_research',
                 gate_refitted_in_bootstrap=False,selection_bias_removed=False,test_access=False)
    return rows, decisions, chosen, stats


def report(manifest, out, *, iterations=10000):
    validate_manifest(manifest)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    accepted=out/'accepted.json'
    if accepted.exists():
        r=json.loads(accepted.read_text())
        if r['identity']!=manifest['identity']:raise ValueError('Review identity changed')
        for name,sha in r['files'].items():
            p=(out/name).resolve()
            if not p.is_relative_to(out.resolve()) or file_sha256(p)!=sha:raise ValueError('Review evidence changed')
        load_records(manifest)
        return r
    rows, decisions, chosen, stats = compute(manifest, iterations=iterations)
    atomic_write_json(manifest,out/'manifest.json')
    atomic_write_json(list(decisions.values()),out/'locked_dev_gates.json')
    atomic_write_json(stats,out/'paired_statistics.json')
    atomic_write_json(rows,out/'metrics.json')
    with (out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['method','scenario','view','macro_f1','gate_enabled']);w.writeheader();w.writerows(rows)
    # Restricted prediction outputs remain in authorized artifact storage.
    for (m,p),a in chosen.items():
        target=out/'development'/f'{m}__{p}.npz';target.parent.mkdir(exist_ok=True)
        z=a['logits'];z0=z-z.max(1,keepdims=True);prob=np.exp(z0);prob/=prob.sum(1,keepdims=True)
        np.savez_compressed(target,participant_ids=a['participant_ids'],labels=a['labels'],logits=z,
                            probabilities=prob,patterns=a.get('patterns',np.full(len(z),p)))
    h=manifest['host']
    lines=['# LOOK：强制开启与独立开关必须分开报告','',
        f"宿主：{h['disease']} / {h['architecture']} / {h['position']} / {h['seed']}；范围：{manifest['scope']}。",'',
        '这里的强制开启仅指候选映射/候选bank：末级是一个节点；逐层固定位置对照是原LOOK选中位置。不是全部可用节点开启。',
        '每种方法、每种缺失方向独立使用同一dev macro-F1严格改善规则，平分或下降关闭。不是逐参与者挑正确预测。',
        '逐层场景这里仅为整套候选bank的最终回退；它不能冒充每种方法独立逐层贪心搜索。',
        '开关按纯缺失场景选一次，混合比例沿用这些开关及原参与者掩码；混合macro-F1不保证不下降。',
        '这是查看初步dev结果后补齐的策略比较，不伪称事前登记。原强制开启记录及统计保持不变。',
        '表中macro-F1单位%，越高越好；强制开启结果用于机制研究，dev_selected_strategy用于这套固定候选的策略比较。',
        'test尚未读取，届时必须固定dev开关，不得用test标签重新决定。无神经训练或重新拟合。',
        '配对区间条件于已选配置和开关，不消除dev选择偏差；仅为本补充比较族，不是全研究同时区间。','',
        '| 方法 | 缺失 | 强制开启 (%) | 开关后 (%) | 开关 |','|---|---|---:|---:|---|']
    for p in PATTERNS:
        host=next(r for r in rows if r['method']=='host' and r['scenario']==p)
        lines.append(f"| 原宿主 | {p} | {100*host['macro_f1']:.3f} | {100*host['macro_f1']:.3f} | 不适用 |")
        for m in manifest['methods']:
            raw=next(r for r in rows if (r['method'],r['scenario'],r['view'])==(m,p,'forced_on_mechanism'))
            final=next(r for r in rows if (r['method'],r['scenario'],r['view'])==(m,p,'dev_selected_strategy'))
            lines.append(f"| {m} | {p} | {100*raw['macro_f1']:.3f} | {100*final['macro_f1']:.3f} | {'开' if final['gate_enabled'] else '关'} |")
    (out/'README.zh-CN.md').write_text('\n'.join(lines)+'\n')
    validate_manifest(manifest)
    files={str(p.relative_to(out)):file_sha256(p) for p in out.rglob('*') if p.is_file() and p.name not in ['status.json','accepted.json','run.lock']}
    receipt=dict(schema='look_gate_review_v1',identity=manifest['identity'],state='accepted',test_access=False,
                 source_unchanged=True,new_neural_training=0,new_correction_fits=0,files=files)
    atomic_write_json(receipt,accepted);return receipt
