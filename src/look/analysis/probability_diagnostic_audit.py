"""Independent CPU audit of probability_diagnostic outputs.

Does not call probability_diagnostic.analyze_pair or logit_metrics.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import f1_score,roc_auc_score


BINS=((0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0))
QUANTILES=(0.0,0.25,0.5,0.75,0.9,0.95,1.0)


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path,digest):
    if sha(path)!=digest: raise ValueError("Input prediction SHA changed")
    with np.load(path,allow_pickle=False) as z:
        y=z["labels"].astype(np.int64);logits=z["logits"].astype(np.float64)
        probs=z["probabilities"].astype(np.float64);ids=z["participant_ids"].astype(str)
    shifted=logits-logsumexp(logits,axis=1,keepdims=True)
    calc=np.exp(shifted)
    if not np.array_equal(probs,calc): raise ValueError("Saved probabilities differ from independent softmax")
    return y,logits,probs,ids


def metrics(y,z):
    pred=z.argmax(1);score=z[:,1]-z[:,0]
    probs=np.exp(z-logsumexp(z,axis=1,keepdims=True))
    return dict(macro_f1=float(f1_score(y,pred,average="macro")),
        macro_auroc_ovr=float(roc_auc_score(y,score)),
        negative_log_likelihood=float((logsumexp(z,axis=1)-z[np.arange(len(y)),y]).mean()),
        multiclass_brier=float(np.square(probs-np.eye(2)[y]).sum(1).mean()))


def losses(y,z):
    probs=np.exp(z-logsumexp(z,axis=1,keepdims=True))
    return logsumexp(z,axis=1)-z[np.arange(len(y)),y],np.square(probs-np.eye(2)[y]).sum(1)


def contribution(mask,d,n):
    mask=np.asarray(mask,bool);return float(d[mask].sum()/n) if mask.any() else 0.0


def close(a,b,atol=1e-12):
    return bool(np.isclose(float(a),float(b),rtol=0,atol=atol))


def audit(spec_path,private_path,output_path):
    spec=read(spec_path);private=read(private_path)
    rows={r["pair_id"]:r for r in private["rows"]}
    if len(spec["items"])!=20 or set(rows)!={x["pair_id"] for x in spec["items"]}: raise ValueError("20-pair audit coverage changed")
    ref=None;max_error=0.0;checked=0
    for item in spec["items"]:
        h_y,h_z,h_p,h_ids=load(item["host"]["path"],item["host"]["sha256"])
        c_y,c_z,c_p,c_ids=load(item["corrected"]["path"],item["corrected"]["sha256"])
        if not np.array_equal(h_ids,c_ids) or not np.array_equal(h_y,c_y): raise ValueError("Pair order changed")
        if ref is None: ref=(h_ids.copy(),h_y.copy())
        if not np.array_equal(ref[0],h_ids) or not np.array_equal(ref[1],h_y): raise ValueError("Cross-pair cohort order changed")
        row=rows[item["pair_id"]];hm=metrics(h_y,h_z);cm=metrics(c_y,c_z)
        for prefix,got in (("host_metrics",hm),("corrected_metrics",cm)):
            for key,value in got.items():
                err=abs(value-row[prefix][key]);max_error=max(max_error,err)
                if err>1e-12: raise ValueError(f"Independent metric mismatch {item['pair_id']} {prefix} {key}")
        hn,hb=losses(h_y,h_z);cn,cb=losses(c_y,c_z);dn=cn-hn;db=cb-hb;n=len(h_y)
        expected=row["delta_metrics"]
        for key,value in (("nll",dn.mean()),("brier",db.mean()),
                          ("macro_f1",cm["macro_f1"]-hm["macro_f1"]),
                          ("macro_auroc_ovr",cm["macro_auroc_ovr"]-hm["macro_auroc_ovr"])):
            err=abs(float(value)-expected[key]);max_error=max(max_error,err)
            if err>1e-12: raise ValueError(f"Independent delta mismatch {item['pair_id']} {key}")
        hp=h_z.argmax(1);cp=c_z.argmax(1);hc=hp==h_y;cc=cp==h_y
        groups={"host_correct_to_correct":hc&cc,"host_correct_to_wrong":hc&~cc,
            "host_wrong_to_correct":~hc&cc,"host_wrong_to_wrong":~hc&~cc}
        for key,mask in groups.items():
            r=row["by_transition"][key]
            if int(mask.sum())!=r["count"]: raise ValueError("Transition count mismatch")
            for field,d in (("contribution_nll",dn),("contribution_brier",db)):
                value=contribution(mask,d,n);err=abs(value-r[field]);max_error=max(max_error,err)
                if err>1e-12: raise ValueError("Transition contribution mismatch")
        conf=h_p.max(1);cover=np.zeros(n,int)
        for i,(lo,hi) in enumerate(BINS):
            mask=(conf>=lo)&((conf<hi) if i<4 else (conf<=hi));cover+=mask
            key=f"[{lo:.1f},{hi:.1f}{')' if i<4 else ']'}";r=row["by_host_confidence_bin"][key]
            if int(mask.sum())!=r["count"]: raise ValueError("Confidence-bin count mismatch")
            for field,d in (("contribution_nll",dn),("contribution_brier",db)):
                value=contribution(mask,d,n);err=abs(value-r[field]);max_error=max(max_error,err)
                if err>1e-12: raise ValueError("Confidence-bin contribution mismatch")
        if not np.all(cover==1): raise ValueError("Confidence-bin coverage mismatch")
        for name,d in (("delta_nll",dn),("delta_brier",db)):
            q=np.quantile(d,QUANTILES)
            for level,value in zip(QUANTILES,q):
                err=abs(float(value)-row["quantiles"][name][str(level)]);max_error=max(max_error,err)
                if err>1e-12: raise ValueError("Quantile mismatch")
        checked+=1
    result=dict(schema="look_probability_diagnostic_independent_audit_v1",state="accepted_right_audit",
        pair_count=checked,participant_count=296,ordered_cohort_exact=True,metrics_independent=True,
        transition_contributions_independent=True,confidence_bins_independent=True,quantiles_independent=True,
        max_abs_error=max_error,spec_sha256=sha(spec_path),private_sha256=sha(private_path),
        test_access=False,training=False,fitting=False,calibration=False,gpu=False)
    Path(output_path).write_text(json.dumps(result,indent=2)+"\n");return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--spec",required=True);p.add_argument("--private",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();audit(a.spec,a.private,a.output)
if __name__=="__main__": main()
