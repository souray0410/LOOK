"""Complete suffix curves and dev-only start selection; no partial winner."""
import csv
from pathlib import Path
import numpy as np
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.studies.project_case import read, verify_case as verify_original
from look.studies.suffix_protocol import PATTERNS, sites
from look.analysis.observed_report import simultaneous_bootstrap


def choose(rows, count):
    if sorted(r['start_ordinal'] for r in rows)!=list(range(1,count+1)):
        raise ValueError('All unique starts must finish before selection')
    if not all(np.isfinite(r['macro_f1']) for r in rows):raise ValueError('Nonfinite score')
    return min(rows,key=lambda r:(-r['macro_f1'],r['start_ordinal']))


def verify_group(root):
    root=Path(root);r=read(root/'accepted.json')
    if r.get('state')!='accepted' or r.get('test_access') is not False or not r.get('all_starts_complete'):
        raise ValueError('Incomplete suffix pilot')
    for path,sha in r['inputs'].items():
        if file_sha256(path)!=sha:raise ValueError('Suffix report input changed')
    from look.studies.suffix_case import check_files
    check_files(root,r['files']);return r


def report(manifest):
    from look.studies.suffix_case import verify_case
    if manifest.get('test_access') is not False:raise ValueError('Test remains sealed')
    out=Path(manifest['output']);out.mkdir(parents=True,exist_ok=True);h=manifest['host'];ordered=sites(h['architecture'],h['position'])
    base_root=Path(manifest['source']['run_dir']);base=read(manifest['source']['spec_path']);verify_original(base_root,base)
    if (base['disease'],base['model']['name'],base['position'],base['seed'])!=tuple(h[k] for k in ('disease','architecture','position','seed')):
        raise ValueError('Wrong original host')
    inputs={str(base_root/'accepted.json'):file_sha256(base_root/'accepted.json')};records={}
    for r in read(base_root/'development/suite.json')['records']:
        if r['scenario'] not in PATTERNS:continue
        if r['method']=='look':records[(1,r['scenario'])]=r
        elif r['method']=='single_final':records[(len(ordered),r['scenario'])]=r
        elif r['method']=='host':records[(0,r['scenario'])]=r
    seen=set()
    for root in map(Path,manifest['runs']):
        s=read(root/'spec.json');verify_case(root,s)
        if s['host']!=h or s['source']['run_dir']!=str(base_root):raise ValueError('Unmatched suffix source')
        n=s['start_ordinal']
        if n in seen:raise ValueError('Duplicate suffix');
        seen.add(n);inputs[str(root/'accepted.json')]=file_sha256(root/'accepted.json')
        for r in read(root/'development/suite.json')['records']:
            if r['method']=='suffix':records[(n,r['scenario'])]=r
    if seen!=set(range(2,len(ordered))):raise ValueError('Incomplete internal suffix set')
    expected={(n,p) for n in range(len(ordered)+1) for p in PATTERNS}
    if set(records)!=expected:raise ValueError('Incomplete anchors or scenarios')
    arrays={};rows=[]
    for key,r in records.items():
        if file_sha256(r['path'])!=r['sha256']:raise ValueError('Prediction changed')
        inputs[r['path']]=r['sha256']
        with np.load(r['path'],allow_pickle=False) as f:arrays[key]={k:f[k] for k in f.files}
        a=arrays[key];from look.evaluation.stability import logit_metrics
        score=logit_metrics(a['labels'],a['logits'])['macro_f1']
        rows.append(dict(start_ordinal=key[0],allowed_start=ordered[key[0]-1] if key[0] else 'host',
            pattern=key[1],macro_f1=score,source='original_case_reused' if key[0] in (0,1,len(ordered)) else 'independent_suffix_refit'))
    keys=sorted(arrays);ref=arrays[keys[0]]
    if any(not np.array_equal(a[k],ref[k]) for a in arrays.values() for k in ('labels','participant_ids')):
        raise ValueError('Not participant-paired')
    weights=[];definitions=[]
    for n in range(2,len(ordered)+1):
        for p in PATTERNS:
            w=np.zeros(len(keys));w[keys.index((n,p))]=1;w[keys.index((1,p))]=-1
            weights.append(w);definitions.append(dict(first_start=n,reference_start=1,pattern=p))
    stats=simultaneous_bootstrap(ref['labels'],np.stack([arrays[k]['logits'].argmax(1) for k in keys]),weights,10000,7341618)
    stats.update(definitions=definitions,family='suffix_vs_earliest_within_host_not_global_study',test_access=False)
    selected={p:choose([r for r in rows if r['pattern']==p and r['start_ordinal']>0],len(ordered)) for p in PATTERNS}
    atomic_write_json(stats,out/'paired_statistics.json');atomic_write_json(selected,out/'selected_starts.json')
    atomic_write_json(rows,out/'results.json');atomic_write_json(manifest,out/'manifest.json')
    with (out/'curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(sorted(rows,key=lambda r:(r['pattern'],r['start_ordinal'])))
    lines=['# LOOK 不同校正起点：完整匹配曲线','',
        '横轴：允许开始校正的节点，按网络前向顺序；纵轴：dev最终分类macro-F1（%），越高越好。',
        '宿主融合位置固定；起点之前全部关闭，各后缀独立拟合，并非截掉原路线的前几个矩阵。',
        '从头与仅末级严格复用原完整研究，其他位置按相同PCA、维数、空间比例和开关规则拟合。',
        '贪心仍可跳过允许起点，因此允许起点不一定是第一个实际开启的节点。',
        '每个缺失方向单独在全部起点完成后按dev选优，同分选更早起点；不按缺失比例另选。',
        '这是dev选择后的探索性证据，选择最佳起点带来额外搜索预算，不是独立test优势。',
        '每个后缀减从头路线的10000次参与者配对bootstrap及本宿主族同时区间见paired_statistics.json。',
        '曲线低于从头路线是反例；不因负结果停止后续种子。不将本族区间冒充全研究同时区间。','',
        '| 允许起点 | 缺失模态 | macro-F1 (%) |','|---|---|---|']
    lines += [f"| {r['allowed_start']} | {r['pattern']} | {100*r['macro_f1']:.3f} |" for r in sorted(rows,key=lambda r:(r['pattern'],r['start_ordinal']))]
    (out/'README.zh-CN.md').write_text('\n'.join(lines)+'\n')
    files={p.name:file_sha256(p) for p in out.iterdir() if p.name in ['paired_statistics.json','selected_starts.json','results.json','manifest.json','curves.csv','README.zh-CN.md']}
    atomic_write_json(dict(state='accepted',host=h,all_starts_complete=True,test_access=False,inputs=inputs,files=files,
        identity=stable_hash(manifest),technical_gate_not_positive_performance=True),out/'accepted.json')
    return verify_group(out)
