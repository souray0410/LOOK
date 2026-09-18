"""Cumulative three-host fusion-stage publication and paired gain differences."""
from pathlib import Path
import numpy as np

from look.analysis.cohort_publication import publish,read,PATTERNS,NAMES,POSITION_DESCRIPTIONS
from look.analysis.observed_report import simultaneous_bootstrap
from look.runtime.state import atomic_write_json,file_sha256,stable_hash
from look.studies.cohort_fusion_stage import METHODS


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
    lines=['# LOOK：R18宿主融合阶段机制比较','',
        '同一小队列、同一ResNet18公开ImageNet初始化类别与完全相同训练/停止规则，只改变原宿主融合发生在Stage2后（middle）、Stage4后（deep）或每眼向量（features）。deep严格引用既有接受run，不重训；middle/features是两个新完整宿主。','',
        '这不是“LOOK从哪个位置开始修正”的消融：融合位置改变了宿主拓扑、参数量、训练权重与融合算子，LOOK仍在每个真实宿主自己的合法correction_sites上独立搜索。','',
        '|宿主融合阶段|执行身份|参数量|实际LOOK correction sites|状态|','|---|---|---:|---|---|']
    for spec,pub in zip(specs,publications):
        structure=_structure(pub,plan['host_structures'][pub['position']])
        host=structure['host_structure']
        state=statuses.get(spec['run_id'],{}).get('state','not_started')
        lines.append(f'|{spec["position"]}|{spec["run_id"]}|{host.get("parameters","—")}|' +
            ' → '.join(structure['correction_sites'])+f'|{state}|')
    lines+=['','## 每个宿主内部结果','',
        '|融合阶段|缺失状态|不修正F1|PCA＋树F1|残差＋树F1|残差−PCA(pp)|','|---|---|---:|---:|---:|---:|']
    for pub in publications:
        rows={(r['method'],r['scenario']):r['metrics'] for r in pub['results']}
        if pub.get('matched_package')!='accepted':
            continue
        for pattern,label in PATTERNS.items():
            host=rows[('host',pattern)]['macro_f1']
            pca=rows[('pca_free_mean',pattern)]['macro_f1']
            rrr=rows[('residual_rrr',pattern)]['macro_f1']
            lines.append(f'|{pub["position"]}|{label}|{100*host:.2f}%|{100*pca:.2f}%|{100*rrr:.2f}%|{100*(rrr-pca):+.2f}|')
    if complete:
        lines+=['','## 跨宿主：相对各自不修正基线的LOOK收益差','',
            '差值定义为“新宿主的(method−host)收益 − deep的(method−host)收益”，单位百分点。8项在开跑前固定，并共用一次296人参与者级10,000次bootstrap；同时区间覆盖这8项。跨零不代表等效。','',
            '|新融合阶段|方法|缺失状态|收益差(pp)|普通95%|8项同时95%|','|---|---|---|---:|---|---|']
        for definition,value in zip(plan['cross_host_comparisons'],current['cross_host_statistics']['contrasts']):
            ordinary='['+', '.join(f'{100*x:+.2f}' for x in value['ordinary_95'])+']'
            simultaneous=('['+', '.join(f'{100*x:+.2f}' for x in value['simultaneous_95'])+']'
                if value['simultaneous_95'] else '零方差')
            lines.append(f'|{definition["position"]}|{NAMES[definition["method"]]}|{PATTERNS[definition["pattern"]]}|{100*value["difference"]:+.2f}|{ordinary}|{simultaneous}|')
    else:
        lines+=['','三宿主尚未全部完整接受，因此8项跨宿主收益差尚未生成，不用局部数值排名。']
    lines+=['','## 解释边界','',
        '- 单seed且同一dev参与宿主/路径选择；bootstrap不包含训练随机性或test泛化。',
        '- middle/deep/features的参数量、融合算子和训练权重不同；结果不能解释成单一“融合层编号”因果效应。',
        '- 本机制回答宿主融合阶段与LOOK收益是否一致，不替代老师外部A/B/C方法匹配缺口。',
        '- 概率质量、收益下降、负结果和跨零结果全部保留；不按F1点估计删正式对照。','',
        '各宿主完整F1/AUROC/NLL/Brier、路径和缓存成本见configs子页；受限预测、权重和参与者资产不上GitHub。']
    text='\n'.join(lines)+'\n'
    target=out/'README.md'
    if not target.exists() or target.read_text()!=text:
        tmp=out/'.README.tmp'
        tmp.write_text(text)
        tmp.replace(target)
    return current
