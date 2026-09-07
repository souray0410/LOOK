"""Test reporting from frozen predictions; no ranking or model selection."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import numpy as np
from .dual_queue import read,record,verify,atomic
from .stable_metrics import logit_metrics
from .method_logit import load_bundle


def macro_f1_draws(labels, logits, draws):
    y=np.asarray(labels,dtype=int);p=np.argmax(logits,axis=1)
    tp=((y==1)&(p==1))[draws].sum(1);tn=((y==0)&(p==0))[draws].sum(1)
    fp=((y==0)&(p==1))[draws].sum(1);fn=((y==1)&(p==0))[draws].sum(1)
    d1=2*tp+fp+fn;d0=2*tn+fp+fn
    return (np.divide(2*tp,d1,out=np.zeros_like(tp,dtype=float),where=d1>0)+
            np.divide(2*tn,d0,out=np.zeros_like(tn,dtype=float),where=d0>0))/2


def paired_ci(pairs,iterations=2000):
    reference=pairs[0][0];ids=reference['participant_ids'].astype(str);y=reference['labels']
    draws=np.random.default_rng(3407).integers(0,len(y),size=(iterations,len(y)))
    deltas=[];point=[]
    for a,b in pairs:
        for v in (a,b):
            if not np.array_equal(ids,v['participant_ids'].astype(str)) or not np.array_equal(y,v['labels']):
                raise ValueError('Paired bootstrap participant order or labels differ')
        deltas.append(macro_f1_draws(y,a['logits'],draws)-macro_f1_draws(y,b['logits'],draws))
        point.append(a['metrics']['macro_f1']-b['metrics']['macro_f1'])
    values=np.mean(deltas,axis=0);low,high=np.quantile(values,[.025,.975])
    # Descriptive paired-bootstrap tail probability, not a permutation/exact test.
    p=min(1.,2*min((np.sum(values<=0)+1)/(iterations+1),(np.sum(values>=0)+1)/(iterations+1)))
    return dict(delta_macro_f1=float(np.mean(point)),ci_low=float(low),ci_high=float(high),
                bootstrap_tail_p=float(p),seed_count=len(pairs),participant_count=len(y))


def holm(values):
    order=np.argsort(values);out=np.zeros(len(values));last=0.
    for rank,index in enumerate(order):
        last=max(last,min(1.,float(values[index])*(len(values)-rank)));out[index]=last
    return out


def write_csv(path,rows):
    if not rows:raise ValueError('No completed report rows')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def report(root):
    root=Path(root);f=read(root/'frozen_manifest.json');out=root/'report';out.mkdir(exist_ok=True)
    results={};bundles={};metric_rows=[];provenance=[]
    for job in f['jobs']:
        p=root/'test'/job['id']/'complete.json';r=read(p)
        if r['status']!='complete' or r['phase']!='test' or r['n']!=290:raise ValueError('Incomplete test report')
        results[job['id']]=r;bundles[job['id']]={};provenance.append(record(p))
        for scenario,rec in r['predictions'].items():
            b=load_bundle(verify(rec));m=logit_metrics(b['labels'],b['logits'])
            for k,v in m.items():
                if isinstance(v,(int,float)) and not np.isclose(v,r['metrics'][scenario][k],rtol=0,atol=1e-12,equal_nan=True):
                    raise ValueError('Reported metric differs from saved predictions')
            bundles[job['id']][scenario]=b
            metric_rows.append(dict(job=job['id'],family=job['family'],fusion=job['fusion'],filling=job['filling'],
                seed=job['seed'],scenario=scenario,participants=len(b['labels']),
                **{k:v for k,v in m.items() if isinstance(v,(int,float))}))
    write_csv(out/'metrics_per_seed.csv',metric_rows)
    groups=defaultdict(list)
    for job in f['jobs']:
        if job['family']=='single_modality':continue
        own=bundles[job['id']]
        original=f'{job["filling"]}_{job["fusion"]}_{job["seed"]}'
        method=(read(job['validation_result']['path']).get('case',{}).get('method') or job['family'])
        if job['family']=='ablation':method=job['id'].split('_only_')[0]+'_only'
        for scenario in ('oct_missing','cfp_missing','random_0.2','random_0.4','random_0.6','random_0.8','random_1.0'):
            groups[(job['family'],method,job['fusion'],job['filling'],scenario,'filling')].append(
                (own['corrected_'+scenario],own['fill_'+scenario]))
            if job['family']!='original':
                groups[(job['family'],method,job['fusion'],job['filling'],scenario,'original_LOOK')].append(
                    (own['corrected_'+scenario],bundles[original]['corrected_'+scenario]))
    comparisons=[]
    for keys,pairs in sorted(groups.items()):
        family,method,fusion,filling,scenario,comparator=keys
        row=dict(family=family,method=method,fusion=fusion,filling=filling,scenario=scenario,comparator=comparator,
                 **paired_ci(pairs))
        row['interpretation']='three-seed paired test comparison' if len(pairs)==3 else 'single-seed ablation; exploratory'
        comparisons.append(row)
    for family,comparator in [('original','filling'),('selected','original_LOOK')]:
        indices=[i for i,r in enumerate(comparisons) if r['family']==family and r['comparator']==comparator and r['scenario'] in ('oct_missing','cfp_missing')]
        for i,p in zip(indices,holm([comparisons[i]['bootstrap_tail_p'] for i in indices])):
            comparisons[i]['holm_bootstrap_tail_p']=float(p)
    write_csv(out/'paired_comparisons.csv',comparisons)
    aggregates=defaultdict(list)
    for r in metric_rows:
        job=r['job']; label=job.rsplit('_',1)[0] # seed is final token for the fixed roster
        aggregates[(label,r['family'],r['fusion'],r['filling'],r['scenario'])].append(r)
    summary=[]
    for key,rows in sorted(aggregates.items()):
        metric_keys=[k for k in rows[0] if k not in ('job','family','fusion','filling','seed','scenario','participants')]
        n=len(rows)
        if len({r['seed'] for r in rows})!=n:raise ValueError('Duplicate seed in aggregation')
        summary.append(dict(zip(('configuration','family','fusion','filling','scenario'),key),seed_count=n,
            **{k+'_mean':float(np.mean([r[k] for r in rows])) for k in metric_keys},
            **{k+'_sd':float(np.std([r[k] for r in rows],ddof=1)) if n>1 else None for k in metric_keys}))
    write_csv(out/'metrics_summary.csv',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(13,5),sharey=True,layout='constrained')
    for ax,pattern in zip(axs,('oct_missing','cfp_missing')):
        rows=[r for r in comparisons if r['family']=='original' and r['scenario']==pattern]
        x=np.arange(len(rows));y=np.array([r['delta_macro_f1'] for r in rows])
        lo=np.array([r['ci_low'] for r in rows]);hi=np.array([r['ci_high'] for r in rows])
        ax.vlines(x,lo,hi,color='#286A8F',lw=2);ax.scatter(x,y,color='#286A8F',zorder=3)
        ax.axhline(0,color='#999999',lw=1);ax.set_xticks(x,[r['fusion']+'\n'+r['filling'] for r in rows],rotation=50,ha='right',fontsize=8)
        ax.set_title('OCT missing' if pattern=='oct_missing' else 'CFP missing');ax.set_ylabel('Test Macro-F1 difference: LOOK − filling')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Frozen LOOK • test (290 participants)\nMean of 3 seeds; paired participant bootstrap 95% interval',fontsize=13)
    fig.savefig(out/'test_overview.png',dpi=180);plt.close(fig)
    lines=['# LOOK test 实验报告','',f'已完成 {len(results)}/87 个冻结配置；test 共 290 名参与者。','',
        '这是经过筛选、类别平衡的内部测试集。不能据此推断自然患病率下的性能或外部临床泛化。',
        '所有参数来自 train 拟合与 validation 选择；本轮 test 不进行起点、节点、维度、阈值或超参数搜索。','',
        '![Test overview](test_overview.png)','',
        '## 核心比较','', '| 方法 | 融合位置 | Filling | 缺失方向 | 对照 | Δ Macro-F1 | 95% 配对区间 |',
        '|---|---|---|---|---|---:|---|']
    for r in comparisons:
        if r['scenario'] not in ('oct_missing','cfp_missing') or r['family'] not in ('original','selected'):continue
        if r['family']=='selected' and r['comparator']!='original_LOOK':continue
        lines.append(f'| {r["family"]} | {r["fusion"]} | {r["filling"]} | {r["scenario"]} | {r["comparator"]} | {r["delta_macro_f1"]:+.4f} | [{r["ci_low"]:+.4f}, {r["ci_high"]:+.4f}] |')
    lines += ['', '## 完整记录与解释边界','',
        '- `metrics_per_seed.csv`：所有种子、场景、原始指标，包括负结果。',
        '- `metrics_summary.csv`：均值与样本标准差；单种子消融不提供三种子结论。',
        '- `paired_comparisons.csv`：原始、选定起点、机制对照和消融的配对差值及区间。',
        '- 三个种子使用相同 290 人，不当作 870 个独立样本。Bootstrap 在参与者层重采样，三种子共享抽样。',
        '- F1、排序指标与 NLL/Brier/ECE 应同时查看；F1 增益不自动说明校准改善。',
        '- 校准结果只适用于本测试队列；不存在 natural test 或外部验证结果。',
        '- 原始 LOOK 和选定起点的固定方向比较分别做 Holm 校正。Bootstrap 尾概率为描述性近似，不能冒充精确检验。',
        '- 看过 test 后不回头调本轮方法；后续修改必须另立开发与评估方案。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    atomic(dict(status='complete',test_access=True,frozen_manifest=record(root/'frozen_manifest.json'),
        input_results=provenance,outputs=[record(p) for p in sorted(out.iterdir()) if p.is_file()]),out/'report_manifest.json')
    return out
