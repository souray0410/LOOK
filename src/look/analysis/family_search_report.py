"""Complete one fitting-family configuration; never implies four-arm acceptance."""
import argparse
from pathlib import Path
import numpy as np
from look.runtime.state import atomic_write_json, file_sha256
from look.studies.project_case import read
from look.studies.search_protocol import PATTERNS
from look.analysis.observed_report import simultaneous_bootstrap


def validate_predictions(search, host):
    for key in ('participant_ids', 'labels'):
        if not np.array_equal(search[key], host[key]):
            raise ValueError('Unpaired report participants or labels')
    if len(np.unique(search['participant_ids'])) != len(search['participant_ids']):
        raise ValueError('Duplicate participant in delivery')


def report(root):
    from look.studies.family_search_case import verify_case, check_files
    root=Path(root);spec=read(root/'spec.json');verify_case(root,spec);ref=spec['reference_search']
    out=root/'delivery';out.mkdir(exist_ok=True)
    digest=file_sha256(root/'accepted.json')
    if (out/'accepted.json').exists():
        r=read(out/'accepted.json')
        if r['input_sha256']!=digest:raise ValueError('Delivery input changed')
        check_files(out,r['files']);return r
    rows=read(root/'development/suite.json')['records'];results=[];diagnostics=[];arrays={}
    for row in rows:
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('Prediction changed')
        key=(row['scenario'],row['method'])
        if key in arrays or key not in {(p,m) for p in PATTERNS for m in ('host','search')}:raise ValueError('Duplicate or unexpected prediction row')
        with np.load(row['path'],allow_pickle=False) as f:arrays[key]={k:f[k] for k in f.files}
        results.append(dict(pattern=row['scenario'],method=row['method'],metrics=row['metrics']))
    if len(arrays)!=4:raise ValueError("Both missing states and matched hosts required")
    reference=None
    predictions=[];weights=[]
    for i,pattern in enumerate(PATTERNS):
        a,b=arrays[pattern,'search'],arrays[pattern,'host'];validate_predictions(a,b)
        if reference is not None:validate_predictions(a,reference)
        reference=a
        predictions.extend([a['logits'].argmax(1),b['logits'].argmax(1)])
        w=np.zeros(2*len(PATTERNS));w[2*i]=1;w[2*i+1]=-1;weights.append(w)
        folder=root/'corrections'/pattern
        diagnostics.append(dict(pattern=pattern,selection=read(folder/'selection.json'),
            replay=read(folder/'replay.json')))
    stats=simultaneous_bootstrap(reference['labels'],np.stack(predictions),weights,10000,7341618)
    stats.update(patterns=list(PATTERNS),direction='search_minus_host',scope='one_selected_dev_configuration_not_independent_confirmation')
    atomic_write_json(results,out/'results.json');atomic_write_json(stats,out/'paired_statistics.json')
    atomic_write_json(dict(search=diagnostics,costs=read(root/'costs.json')),out/'diagnostics.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(PATTERNS),figsize=(9,4),layout='constrained')
    for ax,pattern in zip(axes,PATTERNS):
        values={r['method']:100*r['metrics']['macro_f1'] for r in results if r['pattern']==pattern}
        ax.bar(['Frozen host','LOOK'],[values['host'],values['search']]);ax.set_ylim(0,100)
        ax.set_title(pattern);ax.set_ylabel('Development macro-F1 (%)')
    fig.savefig(out/'comparison.svg');plt.close(fig)
    text=['# LOOK 单配置全流程交付','',f"配置：{ref['host']}；方法 {spec['arm']}；策略 {ref['mode']}；空间因子16；秩32。",
        '横轴：未修正冻结宿主与LOOK；纵轴：开发集macro-F1百分比，越高越好。两个面板分别表示缺OCT和缺CFP。',
        '误差分析、实际开启位置和候选评价数见diagnostics.json；差值及普通/本配置族同时区间见paired_statistics.json。',
        '首种子、开发集选优后的探索结果；区间未消除选择偏差，不是独立test或多种子复现。',
        '正则化在各自已接受前缀的训练投影统计上由原PCA参考GCV规则计算，不使用开发集选择λ；这不是各方法充分调参后的最优性能比较。',
        '本报告仅证明一个拟合方法的双缺失流程；四方法匹配包完成前不做方法排名。',
        '空间因子16指二维特征的高宽各除16；原网络输入未缩小。向量节点不进行空间插值。',
        ('每个已接受前缀独立评价全部下游位置，各位置只保留自身最佳候选，严格正增益扩展全部成分支；无正增益扩展不产生后代。最终报告开发集分数最高路径，全部路径及拒绝证据保留。' if ref['mode']=='positive_forward_tree' else '先扫描剩余下游位置，选择最大正收益；后续拟合使用此前已接受修正。无正收益保留当前模型并停止。' if ref['mode']=='best_forward' else '从指定起点按顺序逐级拟合；每级仅接受正收益修正，否则跳过该级继续。后续拟合使用此前已接受修正。'),'',
        '![单配置结果](comparison.svg)']
    (out/'README.zh-CN.md').write_text('\n\n'.join(text)+'\n')
    names=['results.json','paired_statistics.json','diagnostics.json','comparison.svg','README.zh-CN.md']
    receipt=dict(schema='look_family_search_delivery_v1',arm=spec['arm'],matched_family_complete=False,state='accepted',test_access=False,input_sha256=digest,files={n:file_sha256(out/n) for n in names})
    atomic_write_json(receipt,out/'accepted.json');return receipt


def _package_contrasts(keys,arms):
    definitions=[];weights=[]
    comparisons=[('rrr_free_minus_pca_free',{'residual_rrr':1,'pca_free_mean':-1}),
        ('rrr_constrained_minus_pca_constrained',{'rrr_shared_intercept':1,'shared_pca_ridge':-1}),
        ('pca_free_minus_constrained',{'pca_free_mean':1,'shared_pca_ridge':-1}),
        ('rrr_free_minus_constrained',{'residual_rrr':1,'rrr_shared_intercept':-1}),
        ('subspace_by_mean_interaction',{'residual_rrr':1,'rrr_shared_intercept':-1,'pca_free_mean':-1,'shared_pca_ridge':1})]
    comparisons=[(name,terms) for name,terms in comparisons if set(terms).issubset(arms)]
    comparisons += [(arm+'_minus_host',{arm:1,'host':-1}) for arm in arms]
    for pattern in PATTERNS:
        for name,terms in comparisons:
            w=np.zeros(len(keys))
            for method,coefficient in terms.items():w[keys.index((pattern,method))]=coefficient
            definitions.append(dict(name=name,pattern=pattern,coefficients=terms));weights.append(w)
    return definitions,weights


def refresh_package(feed_path,output,arms=('residual_rrr','pca_free_mean')):
    """Publish by atomic pointer only after complete matched evidence validates.

    Incomplete coverage has no result ranking. Failed refreshes preserve the last
    successful snapshot and its original evidence identity, and expose failure.
    This is reporting only: no training, task admission or seed release.
    """
    import fcntl
    from look.runtime.state import stable_hash
    from look.studies.family_search_protocol import ARMS,validate
    from look.studies.family_search_case import verify_case,check_files
    arms=tuple(arms)
    if arms not in (('residual_rrr','pca_free_mean'),tuple(ARMS)):
        raise ValueError('Only approved free-mean core or full four-arm package allowed')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with (out/'refresh.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            feed=read(feed_path)
            if feed.get('schema')!='look_family_search_feed_v1' or feed.get('test_access') is not False:
                raise ValueError('Sealed family feed required')
            tasks=feed['tasks'];by_arm={};signature=None;completed=[];receipts={}
            for task in tasks:
                if task.get('execution')!='look_family_search' or task.get('test_access') is not False:
                    raise ValueError('Unexpected family task')
                if file_sha256(task['spec'])!=task['spec_sha256']:raise ValueError('Family task spec changed')
                s=read(task['spec']);validate(s);arm=s['arm'];run=Path(task['run_dir'])
                if arm in by_arm or task.get('arm')!=arm:raise ValueError('Duplicate or mismatched family arm')
                key=stable_hash({k:s[k] for k in ('schema','protocol','host','reference_search','candidates',
                    'penalty_policy','candidate_provenance','source_pins')})
                if signature is not None and key!=signature:raise ValueError('Unmatched family scientific identity')
                signature=key;by_arm[arm]=(s,run)
                if arm in arms and (run/'accepted.json').exists():
                    if read(run/'spec.json')!=s:raise ValueError('Accepted run spec differs from registered task')
                    verify_case(run,s);report(run)
                    completed.append(arm);receipts[arm]=file_sha256(run/'accepted.json')
            if set(by_arm)!=set(ARMS):raise ValueError('Exactly four registered fitting arms required')
            inputs=dict(feed_sha256=file_sha256(feed_path),science_identity=signature,case_receipts=receipts,required_arms=list(arms))
            input_digest=stable_hash(inputs)
            if len(completed)!=len(arms):
                state=dict(schema='look_family_package_coverage_v1',state='incomplete',test_access=False,
                    completed=[a for a in arms if a in completed],waiting=[a for a in arms if a not in completed],
                    deferred=[a for a in ARMS if a not in arms],
                    ranking_available=False,replication_released=False,input_digest=input_digest)
                atomic_write_json(state,out/'status.json');return state
            arrays={};rows=[];costs={};reference=None
            for arm in arms:
                s,run=by_arm[arm];seen=set();costs[arm]=read(run/'costs.json')
                for row in read(run/'development/suite.json')['records']:
                    local=(row['scenario'],row['method'])
                    if local in seen or local not in {(p,m) for p in PATTERNS for m in ('host','search')}:
                        raise ValueError('Duplicate or unexpected package prediction')
                    seen.add(local)
                    if file_sha256(row['path'])!=row['sha256']:raise ValueError('Package prediction changed')
                    with np.load(row['path'],allow_pickle=False) as f:a={k:f[k] for k in f.files}
                    validate_predictions(a,a)
                    if reference is not None:validate_predictions(a,reference)
                    reference=a
                    if a['logits'].shape!=(len(a['labels']),2) or not np.isfinite(a['logits']).all():
                        raise ValueError('Finite binary participant logits required')
                    method=arm if row['method']=='search' else 'host';key=(row['scenario'],method)
                    if key in arrays:
                        if not np.array_equal(arrays[key]['logits'],a['logits']):raise ValueError('Frozen host logits differ')
                        continue
                    arrays[key]=a
                    from sklearn.metrics import f1_score
                    score=float(f1_score(a['labels'],a['logits'].argmax(1),labels=[0,1],average='macro',zero_division=0))
                    if not np.isclose(score,row['metrics']['macro_f1'],rtol=0,atol=1e-12):raise ValueError('Reported macro-F1 differs from predictions')
                    rows.append(dict(pattern=row['scenario'],method=method,participants=len(a['labels']),metrics=row['metrics']))
                if len(seen)!=4:raise ValueError('Both missing states required for each fitting arm')
            current=out/'current.json'
            if current.exists():
                old=read(current)
                if old.get('input_digest')==input_digest:
                    snapshot=(out/old['snapshot']).resolve()
                    if not snapshot.is_relative_to(out.resolve()):raise ValueError('Package snapshot escaped output')
                    if file_sha256(snapshot/'accepted.json')!=old['accepted_sha256']:raise ValueError('Package receipt changed')
                    receipt=read(snapshot/'accepted.json');check_files(snapshot,receipt['files'])
                    atomic_write_json(dict(state='accepted',input_digest=input_digest,snapshot=old['snapshot'],test_access=False),out/'status.json')
                    return receipt
            keys=sorted(arrays);definitions,weights=_package_contrasts(keys,arms)
            stats=simultaneous_bootstrap(reference['labels'],np.stack([arrays[k]['logits'].argmax(1) for k in keys]),weights,10000,7341618)
            stats.update(definitions=definitions,model_order=[list(k) for k in keys],test_access=False,
                family='one_host_registered_fitting_package_two_missing_states_not_global_study',
                limitations='conditional_on_dev_selected_paths; bootstrap_does_not_remove_selection_bias')
            snapshot=out/'snapshots'/input_digest;snapshot.mkdir(parents=True,exist_ok=True)
            atomic_write_json(rows,snapshot/'results.json');atomic_write_json(stats,snapshot/'paired_statistics.json')
            atomic_write_json(inputs,snapshot/'inputs.json');atomic_write_json(costs,snapshot/'costs.json')
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            order=['host']+[a for a in ['shared_pca_ridge','pca_free_mean','rrr_shared_intercept','residual_rrr'] if a in arms]
            fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
            for ax,pattern in zip(axes,PATTERNS):
                lookup={r['method']:r['metrics']['macro_f1'] for r in rows if r['pattern']==pattern}
                ax.bar(order,[100*lookup[k] for k in order]);ax.tick_params(axis='x',labelrotation=30)
                ax.set_ylabel('Development macro-F1 (%)');ax.set_title(pattern);ax.set_ylim(0,100)
            fig.savefig(snapshot/'comparison.svg');plt.close(fig)
            text=[f'# LOOK {len(arms)}种拟合方法：完整首种子匹配包','',
                '同一冻结宿主、3416、插值16倍、秩32；各方法独立正收益树，不继承其他方法选中的修正位置。',
                '每个前缀只用train投影统计按PCA参考GCV选择正则化；不等于各方法充分调参的性能上限。',
                ('自由均值固定：比较PCA自由均值与RRR自由均值；约束均值两臂继续登记，当前包不要求完成，不产生均值交互结论。' if len(arms)==2 else 'PCA约束/自由均值与RRR约束/自由均值构成匹配比较。交互项为(RRR自由−RRR约束)−(PCA自由−PCA约束)。'),
                f'差值由单模型macro-F1计算，共用参与者重采样；10000次配对bootstrap及{len(definitions)}个比较族内同时区间，不是全研究校正。',
                'dev用于路径选择，因此区间有选择偏差；首种子结果不证明跨宿主、跨疾病或独立test有效。',
                '横轴为固定方法顺序，不按性能排名；纵轴macro-F1百分比越高越好。其他指标及反例在results.json，不能仅凭F1判断校准。',
                'costs.json记录各任务成本；缓存、迁移与共享资源不同，不构成严格冷启动速度比较。', '',
                '| 方法 | 缺失状态 | N | macro-F1 (%) |','|---|---|---:|---:|']
            for pattern in PATTERNS:
                for method in order:
                    row=next(r for r in rows if r['pattern']==pattern and r['method']==method)
                    text.append(f"| {method} | {pattern} | {row['participants']} | {100*row['metrics']['macro_f1']:.4f} |")
            text += ['','![拟合方法比较](comparison.svg)']
            (snapshot/'README.zh-CN.md').write_text('\n\n'.join(text)+'\n')
            names=['results.json','paired_statistics.json','inputs.json','costs.json','comparison.svg','README.zh-CN.md']
            receipt=dict(schema='look_family_package_delivery_v1',state='accepted',test_access=False,
                matched_package_complete=True,matched_family_complete=(len(arms)==4),required_arms=list(arms),replication_released=False,input_digest=input_digest,
                inputs=inputs,files={n:file_sha256(snapshot/n) for n in names})
            atomic_write_json(receipt,snapshot/'accepted.json')
            pointer=dict(input_digest=input_digest,snapshot=str(snapshot.relative_to(out)),accepted_sha256=file_sha256(snapshot/'accepted.json'))
            atomic_write_json(pointer,current)
            atomic_write_json(dict(state='accepted',input_digest=input_digest,snapshot=pointer['snapshot'],test_access=False),out/'status.json')
            return receipt
        except Exception as e:
            atomic_write_json(dict(state='needs_review',error=repr(e),test_access=False,
                last_success_preserved=(out/'current.json').exists()),out/'status.json')
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--run');group.add_argument('--feed');p.add_argument('--output');p.add_argument('--arms',nargs='+',default=['residual_rrr','pca_free_mean']);a=p.parse_args()
    if a.feed:
        if not a.output:p.error('--feed requires --output')
        refresh_package(a.feed,a.output,a.arms)
    else:report(a.run)
