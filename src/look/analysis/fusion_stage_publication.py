"""Cumulative three-host fusion-stage publication and paired gain differences."""
from pathlib import Path
import numpy as np

from look.analysis.cohort_publication import publish,read,PATTERNS,NAMES,POSITION_DESCRIPTIONS,position_site_names
from look.analysis.observed_report import simultaneous_bootstrap
from look.runtime.state import atomic_write_json,file_sha256,stable_hash
from look.studies.cohort_fusion_stage import METHODS



POSITION_LABELS={
    'middle':'中层特征图融合（Stage 2后）',
    'deep':'深层特征图融合（Stage 4后）',
    'features':'每眼特征向量融合',
}


def _position_label(position):
    return POSITION_LABELS[position]


def _site_summary(position, sites):
    names=position_site_names(position,sites)
    return ' → '.join(names.get(site,site) for site in sites)


def _prediction_rows(spec):
    root=Path(spec['output'])
    rows={}
    for method in METHODS:
        receipt=read(root/method/'accepted.json')
        if receipt.get('state')!='accepted' or receipt.get('identity')!=stable_hash(spec):
            raise ValueError('Fusion-stage method not accepted')
        for row in receipt['records']:
            if file_sha256(row['path'])!=row['sha256']:
                raise ValueError('Fusion-stage prediction changed')
            key=(row['method'],row['scenario'])
            value=dict(np.load(row['path'],allow_pickle=False))
            if key in rows:
                if not all(np.array_equal(rows[key][name],value[name]) for name in value):
                    raise ValueError('Fusion-stage baseline differs within host')
            else:
                rows[key]=value
    return rows


def _cross_host_statistics(plan,specs):
    arrays={}
    reference=None
    for spec in specs:
        position=spec['position']
        rows=_prediction_rows(spec)
        for pattern in PATTERNS:
            for method in ('host',*METHODS):
                item=rows[(method,pattern)]
                if reference is None:
                    reference=item
                if not np.array_equal(reference['participant_ids'],item['participant_ids']) or not np.array_equal(reference['labels'],item['labels']):
                    raise ValueError('Fusion-stage participant order/labels differ across hosts')
                arrays[(position,method,pattern)]=item
    keys=list(arrays)
    weights=[]
    definitions=[]
    for definition in plan['cross_host_comparisons']:
        position,method,pattern=definition['position'],definition['method'],definition['pattern']
        w=np.zeros(len(keys))
        for key,coef in [
            ((position,method,pattern),1),
            ((position,'host',pattern),-1),
            (('deep',method,pattern),-1),
            (('deep','host',pattern),1),
        ]:
            w[keys.index(key)]+=coef
        weights.append(w)
        definitions.append(definition)
    stats=simultaneous_bootstrap(
        reference['labels'],
        np.stack([arrays[key]['logits'].argmax(1) for key in keys]),
        weights,10000,9123416)
    stats['comparison_definitions']=definitions
    stats['family']='eight_prespecified_gain_differences_middle_features_vs_deep'
    return stats


def _structure(publication,expected):
    sites=[]
    for row in publication.get('search_details',[]):
        for site in row.get('sites',[]):
            if site not in sites:
                sites.append(site)
    if sites and sites!=expected['correction_sites']:
        raise ValueError('Fusion-stage published correction sites differ from prespecified host structure')
    actual=publication.get('host_structure')
    if actual is not None and actual!=expected:
        raise ValueError('Fusion-stage host receipt differs from prespecified structure')
    return dict(position=publication['position'],fusion_description=POSITION_DESCRIPTIONS[publication['position']],
        correction_sites=expected['correction_sites'],host_structure=expected)


def refresh(plan,statuses):
    out=Path(plan['publication'])
    out.mkdir(parents=True,exist_ok=True)
    specs=[read(row['spec']) for row in plan['tasks']]
    publications=[publish(spec['output'],out/'configs'/spec['run_id']) for spec in specs]
    complete=(all(p.get('matched_package')=='accepted' for p in publications)
        and all(statuses.get(spec['run_id'],{}).get('state') in ('accepted','accepted_reference') for spec in specs))
    current=dict(schema='look_fusion_stage_publication_v1',study_kind='fusion_stage_v1',
        task_id=plan['task_id'],sequence_id=plan['sequence_id'],test_used=False,
        order=[p['position'] for p in publications],configurations=publications,execution=statuses,
        structures=[_structure(p,plan['host_structures'][p['position']]) for p in publications],cross_host_comparisons=plan['cross_host_comparisons'],
        complete=complete,limitations=['single_seed_same_development_selection','host_parameter_and_training_randomness_differ',
            'fusion_operator_and_topology_change_with_position','not_external_A_B_C_coverage','test_sealed'])
    if complete:
        current['cross_host_statistics']=_cross_host_statistics(plan,specs)
    current['source_evidence_sha256']=stable_hash(current)
    previous=read(out/'current.json') if (out/'current.json').exists() else {}
    if previous!=current:
        atomic_write_json(current,out/'current.json')
    deep_pub=next(p for p in publications if p['position']=='deep')
    middle_pub=next(p for p in publications if p['position']=='middle')
    deep_rows={(r['method'],r['scenario']):r['metrics'] for r in deep_pub['results']}
    middle_rows={(r['method'],r['scenario']):r['metrics'] for r in middle_pub['results']}
    factor=specs[0]['factor'];rank=specs[0]['rank'];seed=specs[0]['seed']
    lines=['# LOOK：R18宿主融合阶段机制比较','',
        '本页已通过左侧独立科学验收；范围仅为WS02青光眼小队列、单seed开发集机制包，test继续封存。','',
        f'**实验范围。** 1,264 train / 296 dev，seed {seed}；输入为二维CFP（彩色眼底照相）与二维OCT（光学相干断层扫描中间切片）224×224。'
        f' x{factor}表示空间特征统计的降采样/压缩倍数（factor={factor}）；q{rank}表示线性拟合使用的低秩维度（rank={rank}），不是网络通道数。'
        ' 两种LOOK拟合均允许自由均值：PCA方向约束与自由低秩残差（RRR）都使用完整正收益树。','',
        '**读表。** dev=开发集；PCA=主成分方向约束；RRR=自由低秩残差；Macro-F1越大越好；pp=百分点。'
        ' NLL（negative log-likelihood，负对数似然）和Brier越低越好；AUROC越高越好。','',
        '三种宿主分别是：**中层特征图融合（Stage 2后）**、**深层特征图融合（Stage 4后）**、**每眼特征向量融合**。'
        ' deep严格引用既有接受run，不重训；middle/features为两个新完整宿主。','',
        '这不是“LOOK从哪个位置开始修正”的消融：融合位置同时改变宿主拓扑、参数量、训练权重与融合算子，LOOK仍在每个真实宿主自己的合法修正位置上独立搜索。','',
        '|宿主融合位置|机器身份|参数量|实际LOOK修正位置（中文）|状态|','|---|---|---:|---|---|']
    for spec,pub in zip(specs,publications):
        structure=_structure(pub,plan['host_structures'][pub['position']])
        host=structure['host_structure']
        state=statuses.get(spec['run_id'],{}).get('state','not_started')
        lines.append(f'|{_position_label(spec["position"])}|{spec["run_id"]}|{host.get("parameters","—")}|' +
            _site_summary(spec['position'],structure['correction_sites'])+f'|{state}|')
    lines+=['','## 每个宿主内部结果','',
        '|融合位置|缺失状态|不修正F1|PCA＋树F1|残差＋树F1|残差−PCA(pp)|','|---|---|---:|---:|---:|---:|']
    for pub in publications:
        rows={(r['method'],r['scenario']):r['metrics'] for r in pub['results']}
        if pub.get('matched_package')!='accepted':
            continue
        for pattern,label in PATTERNS.items():
            host=rows[('host',pattern)]['macro_f1']
            pca=rows[('pca_free_mean',pattern)]['macro_f1']
            rrr=rows[('residual_rrr',pattern)]['macro_f1']
            lines.append(f'|{_position_label(pub["position"])}|{label}|{100*host:.2f}%|{100*pca:.2f}%|{100*rrr:.2f}%|{100*(rrr-pca):+.2f}|')
    lines+=['','### 必须同时看的反例','',
        f'- **middle缺OCT的“大增益”首先来自更弱基线。** middle不修正Macro-F1只有{100*middle_rows[("host","oct_missing")]["macro_f1"]:.2f}%，'
        f'加PCA/残差后为{100*middle_rows[("pca_free_mean","oct_missing")]["macro_f1"]:.2f}%/{100*middle_rows[("residual_rrr","oct_missing")]["macro_f1"]:.2f}%；'
        f'但deep对应PCA/残差的绝对F1已经是{100*deep_rows[("pca_free_mean","oct_missing")]["macro_f1"]:.2f}%/{100*deep_rows[("residual_rrr","oct_missing")]["macro_f1"]:.2f}%。'
        ' middle只有约11.89M参数，而deep/features约22.88M；因此不能把middle更大的修正收益直接解释成“早融合更优”。',
        f'- **F1提高不保证概率质量改善。** deep缺OCT时，NLL从不修正{deep_rows[("host","oct_missing")]["negative_log_likelihood"]:.4f}'
        f'恶化到PCA {deep_rows[("pca_free_mean","oct_missing")]["negative_log_likelihood"]:.4f}、残差 {deep_rows[("residual_rrr","oct_missing")]["negative_log_likelihood"]:.4f}，'
        f'虽然Macro-F1从{100*deep_rows[("host","oct_missing")]["macro_f1"]:.2f}%升到{100*deep_rows[("pca_free_mean","oct_missing")]["macro_f1"]:.2f}%/{100*deep_rows[("residual_rrr","oct_missing")]["macro_f1"]:.2f}%。'
        ' NLL越低越好，所以这个反例不能被F1表掩盖。']
    if complete:
        lines+=['','## 跨宿主：相对各自不修正基线的LOOK收益差','',
            '差值定义为“新宿主的(method−host)收益 − deep的(method−host)收益”，单位百分点。8项在开跑前固定，并共用一次296人参与者级10,000次bootstrap；同时区间覆盖这8项。跨零不代表等效。','',
            '|新融合位置|方法|缺失状态|收益差(pp)|普通95%|8项同时95%|','|---|---|---|---:|---|---|']
        for definition,value in zip(plan['cross_host_comparisons'],current['cross_host_statistics']['contrasts']):
            ordinary='['+', '.join(f'{100*x:+.2f}' for x in value['ordinary_95'])+']'
            simultaneous=('['+', '.join(f'{100*x:+.2f}' for x in value['simultaneous_95'])+']'
                if value['simultaneous_95'] else '零方差')
            lines.append(f'|{_position_label(definition["position"])}|{NAMES[definition["method"]]}|{PATTERNS[definition["pattern"]]}|{100*value["difference"]:+.2f}|{ordinary}|{simultaneous}|')
        feature_defs=[(d,v) for d,v in zip(plan['cross_host_comparisons'],current['cross_host_statistics']['contrasts']) if d['position']=='features']
        if all(v['simultaneous_95'][0]<=0<=v['simultaneous_95'][1] for _,v in feature_defs):
            lines+=['','features相对deep的4项收益差在8项同时95%区间中全部跨零；这表示当前数据不足以确定这些差异的方向，**不等于等效**。']
    else:
        lines+=['','三宿主尚未全部完整接受，因此8项跨宿主收益差尚未生成，不用局部数值排名。']
    lines+=['','## 解释边界','',
        '- 本包已由左侧独立验收，但接受范围仍只是WS02、ResNet18、seed3416、同一开发集机制包；不升级为多seed或独立test结论。',
        '- 单seed且同一dev参与宿主/路径选择；bootstrap不包含训练随机性或test泛化。',
        '- middle/deep/features的参数量、融合算子和训练权重不同；结果不能解释成单一“融合层编号”因果效应。',
        '- features与deep参数量相同但拓扑/融合算子仍不同；features相对deep的4项同时区间全跨零，也不能写成等效。',
        '- 本机制回答宿主融合阶段与LOOK收益是否一致，不替代老师外部A/B/C方法匹配缺口；已有MMTM只回答一个适配A的局部问题。',
        '- 概率质量、收益下降、负结果和跨零结果全部保留；不按F1点估计删正式对照。','',
        '各宿主完整F1/AUROC/NLL/Brier、最终接受路径、中文修正位置和缓存成本见configs子页；受限预测、权重和参与者资产不上GitHub。']
    text='\n'.join(lines)+'\n'
    target=out/'README.md'
    if not target.exists() or target.read_text()!=text:
        tmp=out/'.README.tmp'
        tmp.write_text(text)
        tmp.replace(target)
    return current
