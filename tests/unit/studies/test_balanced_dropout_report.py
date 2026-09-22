import json
from pathlib import Path

import numpy as np
import pytest

from look.runtime.state import file_sha256
from look.studies import balanced_dropout_delivery as module


def _metrics():
    return {
        "macro_f1": 0.5,
        "macro_auroc_ovr": 0.5,
        "negative_log_likelihood": 0.7,
        "multiclass_brier": 0.5,
    }


def _write_prediction(path, participant_ids, labels, logits):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        participant_ids=np.asarray(participant_ids),
        labels=np.asarray(labels, dtype=np.int64),
        logits=np.asarray(logits, dtype=np.float64),
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "metrics": _metrics(),
    }


def _make_report_root(tmp_path, *, mismatch_host=False):
    root=tmp_path/"run"
    root.mkdir()
    identity="fixed-identity"
    labels=np.asarray([0,1]*10,dtype=np.int64)
    ids=np.asarray([f"p{i:02d}" for i in range(len(labels))])
    host=np.column_stack((np.where(labels==0,2.0,-1.0),np.where(labels==1,2.0,-1.0)))
    delta=np.column_stack((np.linspace(-0.2,0.2,len(labels)),np.linspace(0.2,-0.2,len(labels))))
    (root/"host").mkdir()
    (root/"host/accepted.json").write_text(json.dumps({
        "identity":identity,"state":"accepted","best_epoch":0,"stop_epoch":15,
        "metrics":_metrics(),"files":{"best.pt":"frozen-host-best"},
    }))
    for arm_index,arm in enumerate(module.ARMS):
        records=[]
        for pattern_index,pattern in enumerate(module.PATTERNS):
            method_logits=host.copy()
            # Leave one registered contrast exactly degenerate to exercise the
            # zero-variance simultaneous=null contract.
            if not (arm_index==0 and pattern_index==0):
                method_logits=host + (arm_index+pattern_index+1)*delta
            method=_write_prediction(
                root/arm/"development"/f"{arm}_{pattern}.npz",ids,labels,method_logits)
            method.update(method=arm,scenario=pattern)
            records.append(method)
            host_logits=host.copy()
            if mismatch_host and arm_index==1 and pattern_index==0:
                host_logits=host_logits+0.01
            baseline=_write_prediction(
                root/arm/"development"/f"host_{pattern}.npz",ids,labels,host_logits)
            baseline.update(method="host",scenario=pattern)
            records.append(baseline)
        (root/arm).mkdir(parents=True,exist_ok=True)
        (root/arm/"accepted.json").write_text(json.dumps({
            "identity":identity,"state":"accepted","records":records,
            "host_best_sha256":"frozen-host-best","replay_exact":True,
        }))
    return root,identity


def test_registered_contrasts_bind_method_and_same_pattern_host():
    definitions=module._registered_contrast_definitions()
    assert len(definitions)==4
    assert {(x["arm"],x["pattern"]) for x in definitions}=={
        (arm,pattern) for arm in module.ARMS for pattern in module.PATTERNS}
    assert all(x["method_key"]==(x["arm"],x["pattern"]) for x in definitions)
    assert all(x["reference_key"]==("host",x["pattern"]) for x in definitions)


def test_report_entry_builds_four_contrasts_and_zero_variance_null(tmp_path,monkeypatch):
    root,identity=_make_report_root(tmp_path)
    monkeypatch.setattr(module,"validate",lambda spec:None)
    monkeypatch.setattr(module,"stable_hash",lambda spec:identity)
    module.report({},root)
    payload=json.loads((root/"delivery/results.json").read_text())
    assert payload["state"]=="self_checked_pending_independent_review"
    for metric in module.REGISTERED_METRICS:
        rows=payload["paired_statistics"][metric]["rows"]
        assert len(rows)==4
        assert all("method_key" in row and "reference_key" in row for row in rows)
    first=payload["paired_statistics"]["macro_f1"]["rows"][0]
    assert first["favorable_improvement"]==0.0
    assert first["bootstrap_sd"]==0.0
    assert first["simultaneous_95"] is None
    accepted=json.loads((root/"delivery/accepted.json").read_text())
    assert accepted["state"]=="self_checked_pending_independent_review"
    assert accepted["test_access"] is False


def test_report_entry_rejects_cross_arm_host_mismatch(tmp_path,monkeypatch):
    root,identity=_make_report_root(tmp_path,mismatch_host=True)
    monkeypatch.setattr(module,"validate",lambda spec:None)
    monkeypatch.setattr(module,"stable_hash",lambda spec:identity)
    with pytest.raises(ValueError,match="baseline differs"):
        module.report({},root)
