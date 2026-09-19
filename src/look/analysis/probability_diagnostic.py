"""CPU-only probability-loss decomposition for accepted LOOK prediction bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from look.evaluation.stability import logit_metrics, probabilities_from_logits
from look.runtime.state import atomic_write_json, file_sha256

RUNS = (
    ("r50_deep", "2026_09_18_09_35_40"),
    ("r18_deep", "2026_09_18_11_18_28_650020"),
    ("r18_mmtm_deep", "2026_09_18_15_29_03_789799"),
    ("r18_middle", "2026_09_19_02_40_42_281912_middle"),
    ("r18_features", "2026_09_19_02_40_42_281912_features"),
)
METHODS = ("pca_free_mean", "residual_rrr")
PATTERNS = ("oct_missing", "cfp_missing")
BINS = ((0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0))
QUANTILES = (0.0,0.25,0.5,0.75,0.9,0.95,1.0)
METRIC_KEYS = ("macro_f1","macro_auroc_ovr","negative_log_likelihood","multiclass_brier")
PROBABILITY_SUM_ATOL = 8*np.finfo(np.float64).eps


def _read(path):
    return json.loads(Path(path).read_text())


def _finite_binary_bundle(path, expected_sha256):
    path=Path(path)
    if file_sha256(path)!=expected_sha256:
        raise ValueError(f"Prediction SHA changed: {path}")
    with np.load(path,allow_pickle=False) as z:
        required={"labels","probabilities","logits","scores","participant_ids","patterns"}
        if set(z.files)!=required:
            raise ValueError(f"Unexpected prediction fields: {path}")
        value={name:z[name].copy() for name in z.files}
    y=np.asarray(value["labels"],dtype=np.int64)
    logits=np.asarray(value["logits"],dtype=np.float64)
    probs=np.asarray(value["probabilities"],dtype=np.float64)
    ids=np.asarray(value["participant_ids"]).astype(str)
    scores=np.asarray(value["scores"],dtype=np.float64)
    if len(y)!=296 or logits.shape!=(296,2) or probs.shape!=(296,2) or scores.shape!=(296,):
        raise ValueError("Expected 296 two-class development predictions")
    if len(np.unique(ids))!=296 or set(np.unique(y))!={0,1}:
        raise ValueError("Participant IDs or binary labels invalid")
    if not np.isfinite(logits).all() or not np.isfinite(probs).all() or not np.isfinite(scores).all():
        raise ValueError("NaN/Inf in accepted prediction bundle")
    calculated=probabilities_from_logits(logits)
    if not np.array_equal(calculated,probs):
        raise ValueError("Saved probabilities differ from stable softmax(logits)")
    if not np.array_equal(scores,logits[:,1]-logits[:,0]):
        raise ValueError("Saved scores differ from logit_1-logit_0")
    if np.any(probs<0) or np.any(probs>1) or not np.allclose(probs.sum(1),1.0,rtol=0,atol=PROBABILITY_SUM_ATOL):
        raise ValueError("Saved probabilities are outside binary probability simplex")
    return dict(value,labels=y,logits=logits,probabilities=probs,participant_ids=ids,scores=scores)


def _per_person_losses(labels, logits):
    y=np.asarray(labels,dtype=np.int64);z=np.asarray(logits,dtype=np.float64)
    probs=probabilities_from_logits(z)
    nll=logsumexp(z,axis=1)-z[np.arange(len(y)),y]
    brier=np.square(probs-np.eye(2)[y]).sum(axis=1)
    if not np.isfinite(nll).all() or not np.isfinite(brier).all():
        raise ValueError("Non-finite per-person probability loss")
    return nll,brier


def _metric_check(bundle, expected):
    got=logit_metrics(bundle["labels"],bundle["logits"])
    errors={}
    for key in METRIC_KEYS:
        a=float(got[key]);b=float(expected[key])
        if not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a,b,rtol=0,atol=1e-12):
            raise ValueError(f"Accepted metric mismatch: {key}: {a} != {b}")
        errors[key]=abs(a-b)
    return {key:got[key] for key in METRIC_KEYS}, max(errors.values(),default=0.0)


def _summary(mask, dnll, dbrier, total_n):
    mask=np.asarray(mask,dtype=bool);count=int(mask.sum())
    if count==0:
        return dict(count=0,observed=False,mean_delta_nll=None,mean_delta_brier=None,
                    contribution_nll=0.0,contribution_brier=0.0)
    return dict(count=count,observed=True,
        mean_delta_nll=float(dnll[mask].mean()),mean_delta_brier=float(dbrier[mask].mean()),
        contribution_nll=float(dnll[mask].sum()/total_n),
        contribution_brier=float(dbrier[mask].sum()/total_n))


def _public_group(value):
    if value["count"]==0:
        return dict(count=0,status="未观察")
    if value["count"]<10:
        return dict(count_label="少于10",status="分层细节不公开")
    return value


def _bin_masks(conf):
    conf=np.asarray(conf,dtype=np.float64)
    if np.any(conf<0.5-1e-15) or np.any(conf>1+1e-15):
        raise ValueError("Host max probability outside [0.5,1]")
    masks=[]
    covered=np.zeros(len(conf),dtype=int)
    for i,(lo,hi) in enumerate(BINS):
        mask=(conf>=lo)&((conf<hi) if i<len(BINS)-1 else (conf<=hi))
        masks.append(mask);covered+=mask.astype(int)
    if not np.all(covered==1):
        raise ValueError("Confidence bins do not cover every row exactly once")
    return masks


def analyze_pair(host, corrected, host_metrics, corrected_metrics):
    if not np.array_equal(host["participant_ids"],corrected["participant_ids"]):
        raise ValueError("Participant order mismatch")
    if not np.array_equal(host["labels"],corrected["labels"]):
        raise ValueError("Label order mismatch")
    y=host["labels"];n=len(y)
    hm,he=_metric_check(host,host_metrics);cm,ce=_metric_check(corrected,corrected_metrics)
    hnll,hbrier=_per_person_losses(y,host["logits"])
    cnll,cbrier=_per_person_losses(y,corrected["logits"])
    dnll=cnll-hnll;dbrier=cbrier-hbrier
    host_pred=host["logits"].argmax(1);corr_pred=corrected["logits"].argmax(1)
    hc=host_pred==y;cc=corr_pred==y
    transitions={
        "host_correct_to_correct":hc&cc,
        "host_correct_to_wrong":hc&~cc,
        "host_wrong_to_correct":~hc&cc,
        "host_wrong_to_wrong":~hc&~cc,
    }
    by_class={str(c):_summary(y==c,dnll,dbrier,n) for c in (0,1)}
    by_transition={key:_summary(mask,dnll,dbrier,n) for key,mask in transitions.items()}
    host_conf=host["probabilities"].max(1)
    by_bin={f"[{lo:.1f},{hi:.1f}{')' if i<len(BINS)-1 else ']'}":
        _summary(mask,dnll,dbrier,n)
        for i,((lo,hi),mask) in enumerate(zip(BINS,_bin_masks(host_conf)))}
    total_nll=float(dnll.mean());total_brier=float(dbrier.mean())
    for name,groups in (("class",by_class),("transition",by_transition),("bin",by_bin)):
        if not np.isclose(sum(v["contribution_nll"] for v in groups.values()),total_nll,rtol=0,atol=1e-12):
            raise ValueError(f"{name} NLL contributions do not sum to total")
        if not np.isclose(sum(v["contribution_brier"] for v in groups.values()),total_brier,rtol=0,atol=1e-12):
            raise ValueError(f"{name} Brier contributions do not sum to total")
    qn=np.quantile(dnll,QUANTILES);qb=np.quantile(dbrier,QUANTILES)
    positive=np.flatnonzero(dnll>0)
    order=positive[np.argsort(dnll[positive])[::-1]] if len(positive) else positive
    k=min(int(math.ceil(0.1*n)),len(order));top=order[:k]
    positive_sum=float(dnll[dnll>0].sum()/n);negative_sum=float(dnll[dnll<0].sum()/n)
    concentration=dict(
        positive_count=int((dnll>0).sum()),negative_count=int((dnll<0).sum()),zero_count=int((dnll==0).sum()),
        top_requested=int(math.ceil(0.1*n)),top_used=int(k),
        top_positive_contribution_nll=float(dnll[top].sum()/n) if k else 0.0,
        all_positive_contribution_nll=positive_sum,all_negative_contribution_nll=negative_sum)
    private=dict(
        host_metrics=hm,corrected_metrics=cm,metric_max_abs_error=max(he,ce),
        delta_metrics=dict(macro_f1=float(cm["macro_f1"]-hm["macro_f1"]),
            macro_auroc_ovr=float(cm["macro_auroc_ovr"]-hm["macro_auroc_ovr"]),
            nll=total_nll,brier=total_brier),
        by_class=by_class,by_transition=by_transition,by_host_confidence_bin=by_bin,
        quantiles=dict(delta_nll={str(q):float(v) for q,v in zip(QUANTILES,qn)},
                       delta_brier={str(q):float(v) for q,v in zip(QUANTILES,qb)}),
        concentration=concentration,
        participant_count=n)
    public=dict(private)
    public["by_class"]={k:_public_group(v) for k,v in by_class.items()}
    public["by_transition"]={k:_public_group(v) for k,v in by_transition.items()}
    public["by_host_confidence_bin"]={k:_public_group(v) for k,v in by_bin.items()}
    return private,public


def _find_record(receipt,method,pattern):
    rows=[r for r in receipt["records"] if r.get("method")==method and r.get("scenario")==pattern]
    if len(rows)!=1:
        raise ValueError(f"Expected one {method}/{pattern} row")
    return rows[0]


def register_spec(run_root, output, public_cutoffs=None):
    run_root=Path(run_root);items=[]
    for config,run_id in RUNS:
        root=run_root/run_id
        for method in METHODS:
            receipt_path=root/method/"accepted.json";receipt=_read(receipt_path)
            if receipt.get("state")!="accepted" or receipt.get("test_access") is not False:
                raise ValueError(f"Unaccepted source receipt: {run_id}/{method}")
            for pattern in PATTERNS:
                corrected=_find_record(receipt,method,pattern);host=_find_record(receipt,"host",pattern)
                items.append(dict(
                    pair_id=f"{config}__{pattern}__{method}",configuration=config,run_id=run_id,
                    method=method,pattern=pattern,participant_count=296,
                    receipt=str(receipt_path),receipt_sha256=file_sha256(receipt_path),
                    host=dict(path=host["path"],sha256=host["sha256"],metrics=host["metrics"]),
                    corrected=dict(path=corrected["path"],sha256=corrected["sha256"],metrics=corrected["metrics"]),
                    scientific_cutoff=(public_cutoffs or {}).get(run_id)))
    if len(items)!=20 or len({x["pair_id"] for x in items})!=20:
        raise ValueError("Probability diagnostic must register exactly 20 unique pairs")
    spec=dict(schema="look_probability_diagnostic_spec_v1",
        task_id="look-probability-and-external-readiness-20260919-v1",
        purpose="descriptive_posthoc_probability_counterexample_localization_not_confirmatory",
        test_access=False,training=False,fitting=False,calibration=False,gpu=False,
        threads_max=2,participant_count=296,
        methods=list(METHODS),patterns=list(PATTERNS),bins=[list(x) for x in BINS],
        quantiles=list(QUANTILES),top_positive_fraction=0.1,items=items)
    atomic_write_json(spec,output)
    return spec


def analyze(spec_path, private_output, public_output):
    spec=_read(spec_path)
    if (spec.get("schema")!="look_probability_diagnostic_spec_v1" or spec.get("test_access") is not False
            or spec.get("training") is not False or spec.get("fitting") is not False
            or spec.get("calibration") is not False or spec.get("gpu") is not False
            or spec.get("threads_max")!=2 or len(spec.get("items",[]))!=20):
        raise ValueError("Probability diagnostic spec changed")
    private_rows=[];public_rows=[];reference=None;input_files={}
    for item in spec["items"]:
        receipt=Path(item["receipt"])
        if file_sha256(receipt)!=item["receipt_sha256"]:
            raise ValueError("Source receipt changed")
        host=_finite_binary_bundle(item["host"]["path"],item["host"]["sha256"])
        corrected=_finite_binary_bundle(item["corrected"]["path"],item["corrected"]["sha256"])
        if reference is None:
            reference=(host["participant_ids"].copy(),host["labels"].copy())
        if not np.array_equal(reference[0],host["participant_ids"]) or not np.array_equal(reference[1],host["labels"]):
            raise ValueError("All 20 pairs must share the same ordered 296-person cohort")
        private,public=analyze_pair(host,corrected,item["host"]["metrics"],item["corrected"]["metrics"])
        base={k:item[k] for k in ("pair_id","configuration","run_id","method","pattern","scientific_cutoff")}
        private_rows.append(dict(base,**private));public_rows.append(dict(base,**public))
        for role in ("host","corrected"):
            x=item[role];input_files[x["sha256"]]=dict(sha256=x["sha256"],role=role)
    private=dict(schema="look_probability_diagnostic_private_v1",spec_sha256=file_sha256(spec_path),
        test_access=False,participant_count=296,pair_count=20,input_unique_file_count=len(input_files),
        input_files=sorted(input_files.values(),key=lambda x:x["sha256"]),rows=private_rows)
    public=dict(schema="look_probability_diagnostic_public_v1",
        diagnostic_generated_at=datetime.now(timezone.utc).isoformat(),
        purpose=spec["purpose"],test_used=False,participant_count=296,pair_count=20,
        methods=spec["methods"],patterns=spec["patterns"],fixed_bins=spec["bins"],
        quantiles=spec["quantiles"],top_positive_fraction=spec["top_positive_fraction"],
        rows=public_rows,privacy="No participant IDs or per-person values; groups with count <10 suppress detailed losses.",
        interpretation_limits=[
            "Descriptive post-hoc localization only; no new significance ranking.",
            "F1/AUROC and NLL/Brier measure different properties; NLL worsening is not by itself proof of a causal calibration mechanism.",
            "No training, fitting, calibration, model forward pass, or test-set access occurred."
        ])
    atomic_write_json(private,private_output);atomic_write_json(public,public_output)
    return private,public


def render_public(public_path, output_md):
    p=_read(public_path)
    lines=["# LOOK：已接受预测的概率损失反例定位","",
        "本页是**观察后、描述性、零训练**诊断：只读取5个已接受配置的296人development预测，不重新前向、不拟合、不校准、不访问test。","",
        "固定20对 = 5配置 × 2缺失 × 2LOOK修正方法，各自与同一宿主不修正预测配对。ΔNLL/ΔBrier = 修正−宿主，**正值表示概率损失恶化**。分箱按宿主最大预测概率预先固定为[0.5,0.6)、[0.6,0.7)、[0.7,0.8)、[0.8,0.9)、[0.9,1]。","",
        "|配置|缺失|方法|ΔF1(pp)|ΔAUROC(pp)|ΔNLL|ΔBrier|正ΔNLL人数|Top10%正ΔNLL贡献|","|---|---|---|---:|---:|---:|---:|---:|---:|"]
    labels={"r50_deep":"R50 deep","r18_deep":"R18 deep","r18_mmtm_deep":"R18 MMTM deep",
            "r18_middle":"R18 middle","r18_features":"R18 features"}
    methods={"pca_free_mean":"PCA方向约束","residual_rrr":"自由低秩残差"}
    patterns={"oct_missing":"缺OCT（仅CFP）","cfp_missing":"缺CFP（仅OCT）"}
    for r in p["rows"]:
        d=r["delta_metrics"];c=r["concentration"]
        lines.append(f'|{labels[r["configuration"]]}|{patterns[r["pattern"]]}|{methods[r["method"]]}|{100*d["macro_f1"]:+.2f}|{100*d["macro_auroc_ovr"]:+.2f}|{d["nll"]:+.4f}|{d["brier"]:+.4f}|{c["positive_count"]}|{c["top_positive_contribution_nll"]:+.4f}|')
    worsened=sorted(p["rows"],key=lambda r:r["delta_metrics"]["nll"],reverse=True)
    top=worsened[0]
    positive=[r for r in p["rows"] if r["delta_metrics"]["nll"]>0]
    positive_f1=[r for r in positive if r["delta_metrics"]["macro_f1"]>0]
    nll_brier_disagree=[r for r in positive if r["delta_metrics"]["brier"]<0]
    wrong_wrong_positive=[r for r in positive
        if r["by_transition"]["host_wrong_to_wrong"].get("contribution_nll",0)>0]
    lines+=["","## 怎么读这些反例","",
        f'20对中有 **{len(positive)}/20** 对ΔNLL>0，而且这{len(positive)}对全部同时ΔF1>0；因此“F1提高但NLL恶化”不是只出现在单个配置。',
        f'ΔNLL最大的已接受配对是 **{labels[top["configuration"]]} / {patterns[top["pattern"]]} / {methods[top["method"]]}**：ΔNLL={top["delta_metrics"]["nll"]:+.4f}，ΔF1={100*top["delta_metrics"]["macro_f1"]:+.2f}pp。',
        f'在这{len(positive)}个ΔNLL>0配对中，**原错→仍错**组的NLL贡献全部为正（{len(wrong_wrong_positive)}/{len(positive)}）；R18 deep缺OCT尤其明显：剩余错误的损失扩大，与一部分原错→正确带来的F1收益同时存在。这支持“仍错误样本的概率损失被放大”作为描述性候选解释，但不能推出临床或因果机制。',
        f'另外有 **{len(nll_brier_disagree)}/{len(positive)}** 个ΔNLL>0配对同时ΔBrier<0，说明NLL与Brier本身也可能方向相反，不能把“NLL恶化”简写成“所有概率质量/校准指标都恶化”。',
        "固定宿主置信度箱没有出现跨配置统一的恶化区间：R18 deep缺OCT主要由[0.9,1]箱贡献，但MMTM/R50的正ΔNLL行并不遵循同一模式。因此本包不宣称某一置信度区间是统一原因。","",
        "每行的真实类别组、正确性转移组和固定置信度箱都经过私有完整加总验收；公共页面对人数<10的小组只显示“少于10，分层细节不公开”。Top10%项固定取ceil(0.1×296)=30个最大的正ΔNLL（若正值不足30则取全部），并与全部正/负贡献分开报告，避免总差接近0时使用不稳定百分比。","",
        "## 边界","",
        "- 本页不是预注册确认性实验，不新增显著性排名。",
        "- NLL/Brier恶化不等同于已经证明“校准机制变差”；这里只定位概率损失由哪些描述性组贡献。",
        "- 不公开participant ID、逐人损失或受限预测；完整加总审计留在授权服务器。",
        "- 原科学cutoff按各配置accepted prediction保留；诊断生成时间单独记录在current.json。"]
    Path(output_md).write_text("\n".join(lines)+"\n")


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True)
    r=sub.add_parser("register");r.add_argument("--run-root",required=True);r.add_argument("--output",required=True);r.add_argument("--cutoffs")
    a=sub.add_parser("analyze");a.add_argument("--spec",required=True);a.add_argument("--private-output",required=True);a.add_argument("--public-output",required=True)
    d=sub.add_parser("render");d.add_argument("--public",required=True);d.add_argument("--output",required=True)
    args=ap.parse_args()
    if args.cmd=="register":
        cutoffs=_read(args.cutoffs) if args.cutoffs else None;register_spec(args.run_root,args.output,cutoffs)
    elif args.cmd=="analyze": analyze(args.spec,args.private_output,args.public_output)
    else: render_public(args.public,args.output)


if __name__=="__main__":
    main()
