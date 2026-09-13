"""Predefined cross-seed/cross-architecture contrasts on one UKB development cohort."""
import json
from pathlib import Path
import numpy as np
from look.runtime.state import atomic_write_json,file_sha256,stable_hash
from look.analysis.observed_report import simultaneous_bootstrap


def summarize(tasks,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True);groups={};findings=[]
    for task in tasks:
        run=Path(task['run_dir'])
        if not (run/'accepted.json').exists():continue
        spec=json.loads(Path(task['spec']).read_text())
        from look.studies.project_case import verify_case
        verify_case(run,spec)
        groups.setdefault(spec['disease'],[]).append((spec,run))
    for disease,cases in sorted(groups.items()):
        expected={(m,p,s) for m in ('resnet50','densenet121','swin_b') for p in ('middle','deep','features') for s in (3416,3417,3418)}
        actual={(s['model']['name'],s['position'],s['seed']) for s,r in cases}
        if actual!=expected or len(cases)!=len(expected):continue
        signature=stable_hash(sorted(file_sha256(r/'accepted.json') for s,r in cases))
        path=root/(disease+'.json')
        if path.exists() and json.loads(path.read_text()).get('signature')==signature:
            findings.append(json.loads(path.read_text()));continue
        predictions=[];keys=[];identity=None;raw=[]
        for spec,run in sorted(cases,key=lambda x:(x[0]['model']['name'],x[0]['position'],x[0]['seed'])):
            for record in json.loads((run/'report/source_records.json').read_text()):
                if file_sha256(Path(record['path']))!=record['sha256']:raise ValueError('Changed matched prediction')
                with np.load(record['path'],allow_pickle=False) as z:
                    ids=(z['participant_ids'].copy(),z['labels'].copy())
                    if identity is None:identity=ids
                    elif any(not np.array_equal(a,b) for a,b in zip(identity,ids)):raise ValueError('Cross-host identity mismatch')
                    key=(spec['model']['name'],spec['position'],spec['seed'],record['method'],record['scenario'])
                    if key in keys:raise ValueError('Duplicate model view')
                    keys.append(key);predictions.append(z['logits'].argmax(1))
                    raw.append(dict(architecture=key[0],position=key[1],seed=key[2],method=key[3],scenario=key[4],macro_f1=record['metrics']['macro_f1']))
        definitions=[];weights=[]
        for architecture in ('all','resnet50','densenet121','swin_b'):
            models=('resnet50','densenet121','swin_b') if architecture=='all' else (architecture,)
            for scenario in ('oct_missing','cfp_missing'):
                for reference in ('host','bias','affine','single_final','all_on','available_parent'):
                    w=np.zeros(len(keys))
                    for m in models:
                        for p in ('middle','deep','features'):
                            for s in (3416,3417,3418):
                                w[keys.index((m,p,s,'look',scenario))]+=1/(9*len(models))
                                w[keys.index((m,p,s,reference,scenario))]-=1/(9*len(models))
                    definitions.append(dict(architecture=architecture,scenario=scenario,left='look',right=reference,positions='equal_mean_three',seeds='equal_mean_three'));weights.append(w)
        stats=simultaneous_bootstrap(identity[1],np.stack(predictions),weights,10000,73621)
        stats.update(signature=signature,disease=disease,definitions=definitions,raw_seed_rows=raw,
            family='48_predefined_contrasts_within_disease',test_access=False,
            interpretation='single_UKB_development_cohort; architecture_replication_is_not_external_validation')
        for row in stats['contrasts']:
            ci=row['simultaneous_95'];row['judgement']='尚不能分辨'
            if ci:
                if ci[0]>.01:row['judgement']='支持实质提升'
                elif ci[1]<-.01:row['judgement']='支持实质下降'
                elif ci[0]>=-.01 and ci[1]<=.01:row['judgement']='支持实际接近'
        atomic_write_json(stats,path);findings.append(stats)
    lines=['# LOOK 多模型机制研究','',f'三模型×三位置×三种子齐全的疾病：{len(findings)}/3。',
        '当前只比较同一UKB队列；完整输入、缺失OCT、缺失CFP及混合缺失比例分开呈现。',
        '主指标是宿主最终预测macro-F1，越高越好。正差值表示LOOK优于对应对照。',
        '跨模型平均先算每个宿主指标，再等权平均；不拼接不同模型的预测来增加样本量。','',
        '| 疾病 | 模型 | 缺失状态 | 对照 | 差值 pp | 同时区间 pp | 判断 |','|---|---|---|---|---:|---|---|']
    for g in findings:
        for d,r in zip(g['definitions'],g['contrasts']):
            ci=r['simultaneous_95'];ci='未定义' if ci is None else f'[{ci[0]*100:.2f}, {ci[1]*100:.2f}]'
            lines.append(f"| {g['disease']} | {d['architecture']} | {d['scenario']} | {d['right']} | {r['difference']*100:.2f} | {ci} | {r['judgement']} |")
    lines+=['','每个疾病内48个预定对比作同时校正；不把此区间称为跨所有疾病的全局区间。',
        '始终保留逐种子、逐位置及模型反例；父配方和LOOK启用均有开发选优影响，最终结论仍需锁定后的独立评价。']
    (root/'README.zh-CN.md').write_text('\n'.join(lines)+'\n')
    result=dict(complete_diseases=len(findings),expected_diseases=3,test_access=False,study_complete=len(findings)==3)
    atomic_write_json(result,root/'coverage.json');return result
