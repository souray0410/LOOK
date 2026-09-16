"""Predeclared paired comparisons, including the mean-by-subspace interaction."""
import csv
from pathlib import Path
import numpy as np
from look.analysis.observed_report import simultaneous_bootstrap
from look.runtime.state import atomic_write_json, file_sha256
from look.studies.affine_protocol import COMPARISONS
from look.methods.affine_family import ARMS


def contrast_table(keys, seeds):
    weights=[];definitions=[]
    pairs=[(a+' minus '+b, {a:1., b:-1.}) for a,b in COMPARISONS]
    pairs.append(('mean_by_subspace_interaction',
        {'residual_rrr':1.,'rrr_shared_intercept':-1.,'pca_free_mean':-1.,'shared_pca_ridge':1.}))
    for name, coeff in pairs:
        for scenario in ('oct_missing','cfp_missing','missing_average'):
            patterns=('oct_missing','cfp_missing') if scenario=='missing_average' else (scenario,)
            w=np.zeros(len(keys))
            for seed in seeds:
                for p in patterns:
                    for method, value in coeff.items():w[keys.index((seed,method,p))]+=value/(len(seeds)*len(patterns))
            weights.append(w);definitions.append(dict(name=name,scenario=scenario,coefficients=coeff))
    return weights,definitions


def report(cases, out, *, iterations=10000):
    """cases: [(host, scope, records)]; three-seed groups may not pool people."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True);keys=[];pred=[];rows=[];ref=None;seen=set();group=None
    for host,scope,records in cases:
        current=(host['disease'],host['architecture'],host['position'],scope)
        if group is not None and current!=group:raise ValueError('Unmatched affine groups')
        group=current;seed=host['seed']
        if seed in seen:raise ValueError('Duplicate seed')
        seen.add(seed)
        for r in records:
            rows.append(dict(seed=seed,method=r['method'],scenario=r['scenario'],macro_f1=r['metrics']['macro_f1']))
            if r['scenario'] not in ('oct_missing','cfp_missing') or r['method'] not in (*ARMS,'host'):continue
            if file_sha256(r['path'])!=r['sha256']:raise ValueError('Prediction digest changed')
            with np.load(r['path'],allow_pickle=False) as f:
                if ref is None:ref={k:f[k].copy() for k in ('participant_ids','labels')}
                if any(not np.array_equal(ref[k],f[k]) for k in ref):raise ValueError('Unpaired evidence')
                key=(seed,r['method'],r['scenario'])
                if key in keys:raise ValueError('Duplicate view')
                keys.append(key);pred.append(f['logits'].argmax(1))
    if len(cases)>1 and seen!={3416,3417,3418}:raise ValueError('Incomplete replication group')
    if len(keys)!=len(seen)*2*(len(ARMS)+1):raise ValueError('Incomplete matched family')
    weights,definitions=contrast_table(keys,sorted(seen))
    stats=simultaneous_bootstrap(ref['labels'],np.stack(pred),weights,iterations,7341618)
    stats.update(definitions=definitions,test_access=False,seeds=sorted(seen),scope=group[-1],
                 family='51_affine_contrasts_within_host_group_not_global_research')
    atomic_write_json(stats,out/'paired_statistics.json')
    atomic_write_json([dict(host=h,scope=s,records=r) for h,s,r in cases],out/'sources.json')
    with (out/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','method','scenario','macro_f1']);writer.writeheader();writer.writerows(rows)
    lines=['# LOOK统一仿射修正：开发集配对比较','',
        '表中为宿主分类macro-F1，越高越好，但八方法列均强制应用候选映射/bank，不是每方法独立开关后的最终策略。',
        '末级仅一个节点；progressive仅原LOOK选中位置，不等于全部可用节点。独立开关结果见版本化gate-review报告。',
        '末级比较与逐层比较分别报告；宿主融合位置不等于校正起点。',
        '前四臂形成子空间×均值约束比较。无秩约束、对角、正交臂不是等参数或等秩实验。',
        'PLS-SVD方向后接岭回归，不等同于迭代PLSRegression。正交约束作用于标准化坐标，不声称原始特征距离保持。',
        '配置条件于原PCA选型，不能称所有方法独立优化后的最佳性能。拟合不用疾病标签，开发分类指标参与原配置选择。',
        '本报告采用10000次参与者配对bootstrap，三种子共用索引；同时区间属于本宿主比较族，不是整个研究全局区间。',
        '零方差单列；不显著不等于相等。±1 pp为研究参考范围。test未读取。',
        '需要同时阅读原宿主、负收益、特征误差、资源与均值/子空间交互，不能只展示最优排名。','',
        '| Seed | 方法 | 输入状态 | macro-F1 (%) |','|---|---|---|---|']
    lines += [f"| {r['seed']} | {r['method']} | {r['scenario']} | {100*r['macro_f1']:.3f} |" for r in rows]
    (out/'README.zh-CN.md').write_text('\n'.join(lines)+'\n')
    return stats
