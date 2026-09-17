"""Matched best-forward versus four preregistered sequential starts, dev only."""
import csv
from pathlib import Path
import numpy as np
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.studies.project_case import read
from look.studies.search_protocol import representative_starts, PATTERNS
from look.analysis.observed_report import simultaneous_bootstrap


def route_key(spec):
    return 'best_forward' if spec['mode']=='best_forward' else 'sequential_start_'+str(spec.get('start_ordinal',1))


def verify_group(out):
    out=Path(out);r=read(out/'accepted.json')
    if r.get('state')!='accepted' or r.get('test_access') is not False or r.get('profile') is not False:
        raise ValueError('Search report is not accepted')
    for path,sha in r['inputs'].items():
        if file_sha256(path)!=sha:raise ValueError('Search evidence changed')
    for path,sha in r['files'].items():
        if file_sha256(out/path)!=sha:raise ValueError('Search report changed')
    return r


def report(manifest):
    from look.studies.search_case import verify_case
    if manifest.get('test_access') is not False:raise ValueError('Test sealed')
    out=Path(manifest['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'accepted.json').exists():return verify_group(out)
    h=manifest['host'];starts=representative_starts(h['architecture'],h['position'])
    expected={'best_forward'}|{'sequential_start_'+str(i) for i in starts}
    arrays={};metrics=[];diagnostics=[];inputs={};seen=set();source=None
    for root in map(Path,manifest['runs']):
        s=read(root/'spec.json');verify_case(root,s);key=route_key(s)
        if s['host']!=h or key in seen:raise ValueError('Mismatched or duplicate route')
        signature=stable_hash(dict(source=s['source'],pca=s['pca'],spatial_factors=s.get('spatial_factors'),latent_dims=s.get('latent_dims')))
        if source is not None and source!=signature:raise ValueError('Unmatched host or basis')
        source=signature;seen.add(key);inputs[str(root/'accepted.json')]=file_sha256(root/'accepted.json')
        selections={p:read(root/'corrections'/p/'factor_selection.json') for p in PATTERNS}
        traces={str(p.relative_to(root)):read(p) for p in (root/'corrections').rglob('selection.json')}
        diagnostics.append(dict(route=key,eligible_sites=s['eligible_sites'],factor_selection=selections,
                                search_traces=traces,costs=read(root/'costs.json')))
        for row in read(root/'development/suite.json')['records']:
            if row['scenario'] not in PATTERNS:continue
            if file_sha256(row['path'])!=row['sha256']:raise ValueError('Prediction changed')
            inputs[row['path']]=row['sha256']
            name=key if row['method']=='search' else 'host'
            with np.load(row['path'],allow_pickle=False) as f:a={k:f[k] for k in f.files}
            pair=(name,row['scenario'])
            if pair in arrays:
                if any(not np.array_equal(arrays[pair][k],a[k]) for k in ('participant_ids','labels','logits')):
                    raise ValueError('Uncorrected host predictions differ')
            else:
                arrays[pair]=a
                metrics.append(dict(route=name,pattern=row['scenario'],macro_f1=row['metrics']['macro_f1']))
    if seen!=expected:raise ValueError('Complete five-route matched group required')
    keys=sorted(arrays);ref=arrays[keys[0]]
    if any(not np.array_equal(a[k],ref[k]) for a in arrays.values() for k in ('participant_ids','labels')):
        raise ValueError('Participants or labels are not paired')
    weights=[];definitions=[]
    for pattern in PATTERNS:
        for other in ['host']+['sequential_start_'+str(i) for i in starts]:
            w=np.zeros(len(keys));w[keys.index(('best_forward',pattern))]=1;w[keys.index((other,pattern))]=-1
            weights.append(w);definitions.append(dict(method='best_forward',reference=other,pattern=pattern))
    stats=simultaneous_bootstrap(ref['labels'],np.stack([arrays[k]['logits'].argmax(1) for k in keys]),weights,10000,7341618)
    stats.update(definitions=definitions,family='one_host_search_comparison_not_global_study',test_access=False)
    atomic_write_json(stats,out/'paired_statistics.json');atomic_write_json(metrics,out/'results.json')
    atomic_write_json(diagnostics,out/'diagnostics.json');atomic_write_json(manifest,out/'manifest.json')
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['route','pattern','macro_f1']);w.writeheader();w.writerows(metrics)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['host','best_forward']+['sequential_start_'+str(i) for i in starts]
    fig,axs=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for ax,pattern in zip(axs,PATTERNS):
        lookup={r['route']:r['macro_f1'] for r in metrics if r['pattern']==pattern}
        ax.bar(np.arange(len(order)),[100*lookup[k] for k in order])
        ax.set_xticks(np.arange(len(order)),order,rotation=30,ha='right')
        ax.set_ylabel('Development macro-F1 (%)');ax.set_ylim(0,100);ax.set_title(pattern)
    fig.savefig(out/'comparison.svg');plt.close(fig)
    lines=['# LOOK 首种子搜索策略：完整匹配结果','',
        '仅开发集；首种子结果为初步发现，不是多种子复现或独立test结论。',
        '横轴为冻结宿主、最佳位置前向和四个固定起点的顺序逐级策略；纵轴为最终分类macro-F1百分比，越高越好。',
        '两个面板分别缺OCT、缺CFP。柱高不说明统计显著；差值及区间见paired_statistics.json。',
        '差值方向为最佳位置前向减对照，正数有利于前向策略。10000次参与者配对bootstrap的普通与本宿主族同时区间不代表全研究校正。',
        '所有策略保留不修正候选；顺序策略可跳过负收益位置，前向策略扫描所有剩余位置后仅接受最大正增益。',
        '四个起点固定为输入、首个特征阶段、候选节点序列中点、分类前末级特征；不按分数挑起点。',
        '诊断记录每个空间比例的实际开关路线、拟合尝试和评价次数，以及累计已知耗时和峰值；恢复前未闭合会话单列，不能据此夸大速度优势。',
        '本比较保持原PCA/GCV算子，只研究搜索顺序，不替代其他线性算子独立搜索的研究。','',
        '![开发集匹配结果](comparison.svg)','', '| 策略 | 缺失状态 | macro-F1 (%) |','|---|---|---|']
    lines += [f"| {r['route']} | {r['pattern']} | {100*r['macro_f1']:.3f} |" for r in metrics]
    (out/'README.zh-CN.md').write_text('\n'.join(lines)+'\n')
    names=['paired_statistics.json','results.json','diagnostics.json','manifest.json','results.csv','comparison.svg','README.zh-CN.md']
    atomic_write_json(dict(state='accepted',profile=False,test_access=False,host=h,identity=stable_hash(manifest),
                          inputs=inputs,files={n:file_sha256(out/n) for n in names},complete_routes=True),out/'accepted.json')
    return verify_group(out)
