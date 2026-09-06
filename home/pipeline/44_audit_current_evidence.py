#!/usr/bin/env python3
"""CPU-only cohort provenance and exploratory validation sensitivity report."""
import argparse,csv,json,re
from collections import Counter
from pathlib import Path
import numpy as np
from look_core.start_study import read_json,source_result,file_record
from look_core.start_report import write_csv
from look_core.stable_metrics import logit_metrics
from look_core.state import atomic_write_json,atomic_write_text,utc_now


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh audit output')
    parent=read_json(a.parent_summary)
    if parent['test_access'] is not False:raise ValueError('Only sealed validation audit allowed')
    original=source_result(parent,a.parent_summary.parents[2],'normalized_mean_layer3_3407')
    config=read_json(original.with_name('validation_manifest.json'))['config']
    labels=Path(config['labels_csv']);root=labels.parents[4]
    candidates=root/'phenotypes/record_phenotype_candidates.csv'
    with labels.open() as f:rows=list(csv.DictReader(f))
    with candidates.open() as f:candidate_rows=list(csv.DictReader(f))
    def sources(row):return set(re.findall(r'41270|41271|41272|20002|20004|6148|glaucoma_treatment',row['evidence_sources']))
    case_rows=[r for r in rows if r['label_id']=='1'];val={r['participant_id']:r for r in rows if r['split']=='validation'}
    counts=dict(paired_candidates=len(candidate_rows),candidate_statuses=dict(Counter(r['candidate_status'] for r in candidate_rows)),
        matched_participants=len(rows),split_labels=dict(Counter(r['split']+':'+r['label_id'] for r in rows)),
        cases=len(case_rows),hospital_icd_cases=sum(bool(sources(r)&{'41270','41271'}) for r in case_rows),
        treatment_flag_cases=sum('glaucoma_treatment' in sources(r) for r in case_rows),
        treatment_flag_only_cases=sum(sources(r)=={'glaucoma_treatment'} for r in case_rows))
    groups={
        'all_evidence':lambda s:True,
        'hospital_icd_positive':lambda s:bool(s&{'41270','41271'}),
        'concordant_selfreport_positive':lambda s:{'6148','20002'}<=s,
        'excluding_treatment_only_positive':lambda s:s!={'glaucoma_treatment'},
    }
    sensitivity=[];tradeoffs=[];provenance=[file_record(labels),file_record(candidates),file_record(a.parent_summary)]
    for filling in ('normalized_mean','raw_zero','paired_cgan'):
      for fusion in ('layer3','feature','layer2'):
       for seed in (3407,3408,3409):
        path=source_result(parent,a.parent_summary.parents[2],f'{filling}_{fusion}_{seed}')
        r=read_json(path);provenance.append(file_record(path))
        for pattern in ('oct_missing','cfp_missing'):
         before=r['validation']['fill_'+pattern];after=r['validation']['look_after_fill_'+pattern]
         tradeoffs.append(dict(fusion=fusion,filling=filling,seed=seed,pattern=pattern,
            delta_f1=after['macro_f1']-before['macro_f1'],delta_auroc=after['macro_auroc_ovr']-before['macro_auroc_ovr'],
            delta_nll=after['negative_log_likelihood']-before['negative_log_likelihood'],source=str(path)))
         if fusion!='layer3' or filling!='normalized_mean':continue
         for method,scenario in [('filling','fill_'+pattern),('original_LOOK','look_after_fill_'+pattern)]:
          pred=path.parent/'predictions'/f'validation__{scenario}.npz';provenance.append(file_record(pred))
          with np.load(pred,allow_pickle=False) as bundle:
           ids=bundle['participant_ids'].astype(str);y=bundle['labels'];z=bundle['logits']
           if any(pid not in val for pid in ids):raise ValueError('Non-validation participant encountered')
           for group,select in groups.items():
            mask=np.array([val[pid]['label_id']=='0' or select(sources(val[pid])) for pid in ids])
            ys=y[mask];zs=z[mask];metrics=logit_metrics(ys,zs) if len(np.unique(ys))==2 else {}
            for metric in ('macro_f1','macro_auroc_ovr','macro_auprc_ovr','ece_15','multiclass_brier','negative_log_likelihood'):
             sensitivity.append(dict(group=group,method=method,seed=seed,pattern=pattern,n_controls=int((ys==0).sum()),
                n_cases=int((ys==1).sum()),metric=metric,value=metrics.get(metric),source=str(pred)))
    a.output.mkdir(parents=True)
    write_csv(sensitivity,a.output/'validation_label_sensitivity.csv');write_csv(tradeoffs,a.output/'metric_tradeoffs.csv')
    atomic_write_json(dict(counts=counts,generated_at_utc=utc_now(),test_access=False,provenance=provenance),a.output/'audit.json')
    atomic_write_text(f'''# LOOK 当前证据审计

生成时间：{utc_now()}。只分析validation预测，未解封test。

## 人群与标签

- 配对候选参与者：{counts['paired_candidates']}；匹配人群：{len(rows)}（病例{len(case_rows)}，对照{len(rows)-len(case_rows)}）。
- 病例中含住院ICD来源：{counts['hospital_icd_cases']}；含治疗标记：{counts['treatment_flag_cases']}；仅治疗标记：{counts['treatment_flag_only_cases']}。
- 5326/5327含青光眼或高眼压治疗史，不能统称医生确诊；完整字段来源说明见项目导师问题台账。
- 筛选过程可复现不等于合理性已被证明；任务选择曾参考短预算validation表现，必须披露。

## 指标冲突与敏感性

- 54个固定方向比较中，F1提高{sum(r['delta_f1']>0 for r in tradeoffs)}个；AUROC下降{sum(r['delta_auroc']<0 for r in tradeoffs)}个；NLL上升（恶化）{sum(r['delta_nll']>0 for r in tradeoffs)}个。
- 标签敏感性固定layer3/normalized_mean/三种子，复用原预测，不重新拟合或选配置。每个来源病例子集都使用同一批validation对照；分组可重叠，病例比例改变，F1/AUPRC不能跨组当作纯标签质量效应比较。
- ICD子组病例可能很少，结果仅为探索性描述；不用于挑选报告范围或改变原标签。
- 所有结果为配置选择所用验证集的开发证据。线性、逐层、联合信息等机制结论需等待相应对照完成。

## 本轮重点

起点实验保留完整代表性设置，以展示起点影响现象，不追求当前模型最优。15个方法对照及3个自身输入对照接续；跨数据、模型和SSF+LOOK互补性仍未验证。见ADVISOR_QUESTIONS.md逐项跟踪。
''',a.output/'当前证据审计.md')
    print(json.dumps(dict(output=str(a.output),counts=counts),ensure_ascii=False))


if __name__=='__main__':main()
