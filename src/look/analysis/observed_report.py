"""Participant-paired development reporting; no test access or model selection."""
import csv
from pathlib import Path
import numpy as np
from look.runtime.state import atomic_write_json,file_sha256


def f1_from_confusion(c):
    tn,fp,fn,tp=np.moveaxis(c,-1,0)
    first=np.divide(2*tn,2*tn+fn+fp,out=np.zeros_like(tn,dtype=float),where=(2*tn+fn+fp)>0)
    second=np.divide(2*tp,2*tp+fn+fp,out=np.zeros_like(tp,dtype=float),where=(2*tp+fn+fp)>0)
    return (first+second)/2


def simultaneous_bootstrap(labels,predictions,contrasts,iterations,seed):
    """Group identical outcome patterns: exactly the participant multinomial law.

    Each replicate uses one shared sample across all models. No cross-model rows
    are treated as independent participants. Contrasts may average model F1s.
    """
    y=np.asarray(labels);pred=np.asarray(predictions);weights=np.asarray(contrasts,dtype=float)
    if pred.ndim!=2 or pred.shape[1]!=len(y) or weights.shape[1]!=len(pred):raise ValueError('Bootstrap dimensions mismatch')
    if iterations<1 or not np.isin(y,[0,1]).all() or not np.isin(pred,[0,1]).all():raise ValueError('Invalid binary bootstrap inputs')
    patterns,counts=np.unique(np.column_stack([y,pred.T]),axis=0,return_counts=True)
    codes=2*patterns[:,0,None]+patterns[:,1:]
    indicator=(codes[:,:,None]==np.arange(4)).reshape(len(patterns),-1).astype(float)
    original=f1_from_confusion((counts@indicator).reshape(len(pred),4))
    estimate=weights@original
    rng=np.random.default_rng(seed);draws=[]
    for start in range(0,iterations,64):
        sampled=rng.multinomial(len(y),counts/counts.sum(),size=min(64,iterations-start))
        conf=(sampled@indicator).reshape(len(sampled),len(pred),4)
        draws.append(f1_from_confusion(conf)@weights.T)
    draws=np.concatenate(draws);standard=draws.std(0,ddof=1) if iterations>1 else np.zeros(len(weights))
    valid=standard>0
    maximum=np.max(np.abs((draws[:,valid]-draws[:,valid].mean(0))/standard[valid]),axis=1) if valid.any() else np.zeros(iterations)
    critical=float(np.quantile(maximum,.95))
    intervals=[]
    for i,point in enumerate(estimate):
        ordinary=np.quantile(draws[:,i],[.025,.975]).tolist()
        simultaneous=[float(point-critical*standard[i]),float(point+critical*standard[i])] if valid[i] else None
        intervals.append(dict(difference=float(point),ordinary_95=ordinary,simultaneous_95=simultaneous,
            bootstrap_sd=float(standard[i]),zero_variance=not bool(valid[i])))
    return dict(iterations=iterations,participants=len(y),seed=seed,critical=critical,
        scope='conditional_on_selected_models_and_development_selection',contrasts=intervals)


def write_report(records,output,iterations=10000,seed=3416):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[];all_data={}
    for r in records:
        if file_sha256(Path(r['path']))!=r['sha256']:raise ValueError('Changed prediction evidence')
        key=(r['method'],r['scenario'])
        if key in all_data:raise ValueError('Duplicate evaluation view')
        all_data[key]=np.load(r['path'],allow_pickle=False)
        rows.append(dict(method=r['method'],scenario=r['scenario'],**{k:v for k,v in r['metrics'].items() if isinstance(v,(int,float))}))
    fields=['method','scenario']+sorted(set().union(*(set(r)-{'method','scenario'} for r in rows)))
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    keys=list(all_data);reference=all_data[keys[0]]
    for item in all_data.values():
        if not np.array_equal(reference['participant_ids'],item['participant_ids']) or not np.array_equal(reference['labels'],item['labels']):
            raise ValueError('Unpaired report inputs')
    comparisons=[];weights=[]
    for pattern in ('oct_missing','cfp_missing'):
        for method in ('host','bias','affine','single_final','all_on','available_parent'):
            if ('look',pattern) not in all_data or (method,pattern) not in all_data:continue
            w=np.zeros(len(keys));w[keys.index(('look',pattern))]=1;w[keys.index((method,pattern))]=-1
            weights.append(w);comparisons.append(dict(method='look',reference=method,scenario=pattern))
    stats=simultaneous_bootstrap(reference['labels'],np.stack([all_data[k]['logits'].argmax(1) for k in keys]),weights,iterations,seed)
    stats['comparison_definitions']=comparisons;stats['family']='one_host_development_diagnostics_not_whole_study_family'
    atomic_write_json(stats,out/'paired_statistics.json')
    atomic_write_json(records,out/'source_records.json')
    lines=['# LOOK 自动匹配评价','', '本页为开发集结果，包含开发集选优影响，不是独立test证据。',
        '主指标是融合宿主最终预测的macro-F1，数值越大越好；不是两条分支F1的平均。',
        '完整输入时不启用修正，所以完整输入预测相同是设计保证，不是额外的性能发现。',
        '缺失比例使用固定参与者掩码；一个人的全部有效眼一起缺失，所有方法共用掩码。','',
        '| 方法 | 输入状态 | macro-F1 (%) |','|---|---|---|']
    for r in rows:lines.append(f"| {r['method']} | {r['scenario']} | {100*r['macro_f1']:.2f} |")
    lines+=['','差值方向统一为LOOK减对照，单位pp；大于0有利于LOOK。',
        'paired_statistics.json提供10000次参与者级配对重采样及本宿主比较族同时区间；不把本宿主区间当全研究校正。',
        '新旧模型、数据角色、所有失败与未收敛任务仍独立记账；结果不保证投稿录用。']
    (out/'README.zh-CN.md').write_text('\n'.join(lines)+'\n')
