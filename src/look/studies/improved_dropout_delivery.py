"""Frozen improved-dropout A -> two missing states x two LOOK families -> report."""
from __future__ import annotations
import argparse, copy, fcntl, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from mhd_framework.models import create_model

from look.data.array_pair import ArrayPair
from look.data.observed_pair import collate_observed
from look.models.observed_participant import ObservedParticipantModel
from look.models.improved_modality_dropout import build_improved_dropout_host
from look.methods.improved_dropout_family import (
    ARMS,PATTERNS,SITES,evaluate,fit_pca_bank,fit_family_trajectory,load_bank
)
from look.runtime.state import atomic_write_json,file_sha256,stable_hash
from look.evaluation.evaluator import save_prediction_bundle
from look.studies.embracenet_delivery import _simultaneous_contrasts,REGISTERED_METRICS

SOURCE_A_COMMIT="77f94be3ecc3a7632bf34aad5fc288d542036ad6"
AUTHOR_COMMIT="8040d96b2dec48cf8fc7d13b45e15af0d07952ed"
A_RECEIPT_SHA="47dcc6a8276332e06003228e894aff61a17d392f6187ff06f1c1d555f409d1da"
A_BEST_SHA="edc6f5f26befdda3c742a2683b77aa9f629e7cfcdde60c198b83e7c70c534530"
A_PRED_SHA="212e1e4e32a25bf5873b64c66504b84166d0be9a1262f127ba49f4b9f3a6e0ca"


def read(p): return json.loads(Path(p).read_text())


def continuation_identity(root,source_commit):
    root=Path(root);spec=read(root/"spec.json");receipt=read(root/"host/accepted.json")
    if file_sha256(root/"host/accepted.json")!=A_RECEIPT_SHA or receipt["files"]["best.pt"]!=A_BEST_SHA or receipt["files"]["development_predictions.npz"]!=A_PRED_SHA:
        raise ValueError("Frozen IMD A assets changed")
    if receipt.get("best_epoch")!=13 or receipt.get("stop_epoch")!=28 or receipt.get("state")!="accepted":
        raise ValueError("Frozen IMD A training identity changed")
    return {
        "schema":"look_imd_component_A_plus_LOOK_v1","source_A_commit":SOURCE_A_COMMIT,
        "continuation_source_commit":source_commit,"author_commit":AUTHOR_COMMIT,
        "A_receipt_sha256":A_RECEIPT_SHA,"A_best_sha256":A_BEST_SHA,"A_predictions_sha256":A_PRED_SHA,
        "A_identity":receipt["identity"],"data_root":spec["data_root"],"data_audit_sha256":spec["data_audit_sha256"],
        "seed":3416,"arms":list(ARMS),"patterns":list(PATTERNS),"sites":list(SITES),"rank":32,
        "selection":"development macro-F1 positive-forward tree; strict improvement",
        "method_identity":"simultaneous-modality-dropout + learnable-token target-task component adaptation; contrastive pretraining omitted",
        "encoder_boundary":"weights frozen but BatchNorm running buffers changed during target training; not author-frozen encoder-state reproduction",
        "test_access":False,
    }


def build_graph(root,device):
    root=Path(root)
    config=dict(name="resnet18",spatial_dims=2,in_channels=3,num_classes=2,views=1,granularity="block")
    parents=[ObservedParticipantModel(create_model(config)) for _ in range(2)]
    graph=build_improved_dropout_host(*parents,device=str(device),hidden_dropout=.1,classifier_dropout=.1)
    state=torch.load(root/"host/best.pt",map_location="cpu",weights_only=False)
    graph.load_state_dict(state["model"],strict=True);graph.to(device);graph.eval()
    for p in graph.parameters(): p.requires_grad_(False)
    return graph


def loader(ds,batch=16):
    return DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0,collate_fn=collate_observed,
        generator=torch.Generator().manual_seed(3416))


def init(root,source_commit):
    root=Path(root);identity=continuation_identity(root,source_commit);target=root/"look_continuation"
    target.mkdir(exist_ok=True)
    path=target/"contract.json"
    if path.exists() and read(path)!=identity: raise ValueError("IMD LOOK continuation identity changed")
    atomic_write_json(identity,path)
    # Method-identity audit is part of the continuation, not a mutation of the frozen A.
    audit={
      "schema":"look_imd_method_identity_audit_v1","state":"accepted_identity_correction","test_access":False,
      "paper":"MICCAI 2025 / arXiv:2509.18284",
      "author_commit":AUTHOR_COMMIT,
      "paper_staging":{
        "contrastive_pretraining":"projector head; supervised sigmoid contrastive; Lhat_con = L_oct,cfp + L_oct,fused + L_cfp,fused",
        "target_training":"classifier head; simultaneous modality dropout Eq1; learnable modality tokens",
        "order":"paper ablation explicitly adds contrastive learning before target training with L_smd + tokens"
      },
      "current_A":{
        "target_training_executed":True,"contrastive_pretraining_executed":False,
        "loss":"complete CE + oct-missing CE + cfp-missing CE, lambda=1",
        "learnable_tokens":True,"encoders_parameter_gradients":False,
        "encoder_bn_buffers_changed":True,
        "encoder_bn_changed_keys":60,
        "label":"IMD target-task component adaptation, not the complete Learning Contrastive Multimodal Fusion method"
      },
      "frozen_A_assets":{"accepted_sha256":A_RECEIPT_SHA,"best_sha256":A_BEST_SHA,"predictions_sha256":A_PRED_SHA},
      "look_pairing":"retain this A as a matched frozen component baseline; do not retrofit contrastive pretraining into the completed A"
    }
    atomic_write_json(audit,target/"method_identity_audit.json")
    return identity


def run(root,source_commit):
    root=Path(root);target=root/"look_continuation";identity=init(root,source_commit);sid=stable_hash(identity)
    with (target/"pipeline.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_write_json({"state":"running","identity":sid,"test_access":False,"time":time.time()},target/"status.json")
        device=torch.device("cuda:0");graph=build_graph(root,device)
        train=ArrayPair(identity["data_root"],"train");dev=ArrayPair(identity["data_root"],"development")
        if len(train)!=1264 or len(dev)!=296 or set(train.participant_ids)&set(dev.participant_ids):
            raise ValueError("IMD continuation split identity changed")
        pca=fit_pca_bank(graph,loader(train),device,target/"pca",sid,32)
        atomic_write_json({"state":"accepted","identity":sid,"test_access":False,
            "files":{s:file_sha256(target/"pca"/f"{s}.pt") for s in SITES}},target/"pca_acceptance.json")
        records=[]
        for arm in ARMS:
            arm_records=[]
            for pattern in PATTERNS:
                folder=target/arm/pattern
                if (folder/"accepted.json").exists():
                    rec=read(folder/"accepted.json")
                    if rec.get("identity")!=sid: raise ValueError("IMD LOOK arm identity changed")
                    arm_records.extend(rec["records"]);continue
                bank,selection=fit_family_trajectory(graph,loader(train),loader(dev),arm=arm,pattern=pattern,bases=pca,
                    identity=sid,output=folder,device=device,workspace_bytes=8*1024**3,rank=32)
                a=evaluate(graph,loader(dev),device,pattern,bank)
                b=evaluate(graph,loader(dev),device,pattern,())
                rows=[]
                for method,result in ((arm,a),("host",b)):
                    path=target/arm/"development"/f"{method}_{pattern}.npz"
                    save_prediction_bundle(result,path)
                    rows.append({"method":method,"scenario":pattern,"path":str(path),"sha256":file_sha256(path),"metrics":result["metrics"]})
                rec={"schema":"look_imd_formal_arm_v1","state":"accepted","identity":sid,"arm":arm,"pattern":pattern,
                    "selection_sha256":file_sha256(folder/"selection.json"),"bank_sha256":file_sha256(folder/"bank.pt"),
                    "records":rows,"test_access":False}
                atomic_write_json(rec,folder/"accepted.json");arm_records.extend(rows)
            atomic_write_json({"state":"accepted","identity":sid,"arm":arm,"records":arm_records,"test_access":False},target/arm/"accepted.json")
            records.extend(arm_records)
        report(root,source_commit)
        atomic_write_json({"state":"completed","identity":sid,"test_access":False,"time":time.time()},target/"status.json")


def report(root,source_commit):
    root=Path(root);target=root/"look_continuation";identity=continuation_identity(root,source_commit);sid=stable_hash(identity)
    arrays={};rows=[]
    for arm in ARMS:
        rec=read(target/arm/"accepted.json")
        if rec.get("identity")!=sid or rec.get("state")!="accepted":raise ValueError("Unaccepted IMD LOOK arm")
        for row in rec["records"]:
            if file_sha256(row["path"])!=row["sha256"]:raise ValueError("IMD LOOK prediction changed")
            a=dict(np.load(row["path"],allow_pickle=False));key=(row["method"],row["scenario"])
            if key in arrays:
                if any(not np.array_equal(arrays[key][n],a[n]) for n in ("participant_ids","labels","logits")):raise ValueError("Shared IMD baseline differs")
            else: arrays[key]=a;rows.append(row)
    ref=next(iter(arrays.values()))
    logits={k:v["logits"].astype(np.float64) for k,v in arrays.items()}
    definitions=[{"arm":arm,"pattern":pattern,"method_key":(arm,pattern),"reference_key":("host",pattern)} for pattern in PATTERNS for arm in ARMS]
    stats=_simultaneous_contrasts(ref["labels"].astype(np.int64),logits,definitions,10000,3416)
    delivery=target/"delivery";delivery.mkdir(exist_ok=True)
    host=read(root/"host/accepted.json")
    payload={"schema":"look_imd_component_A_plus_LOOK_results_v1","state":"self_checked_pending_independent_review",
      "identity":sid,"test_access":False,"method_identity":"IMD target-task component adaptation; no contrastive pretraining; encoder BN buffers adapted",
      "host_complete_metrics":host["metrics"],"development_results":rows,"paired_statistics":stats,
      "selection_bias":"same 296-person development selects A and LOOK trees and estimates effects",
      "uncertainty_limit":"single seed; participant bootstrap excludes training-seed variation"}
    atomic_write_json(payload,delivery/"results.json")
    lines=["# IMD target-task组件 A + LOOK（单种子候选）","",
      "重要身份边界：当前A执行了simultaneous modality dropout + learnable token的target-task CE训练；**没有执行论文的contrastive multimodal fusion pretraining**。此外encoder参数未更新，但训练态BN running buffers发生变化，因此不是论文所述完整frozen-encoder-state复现。","",
      f"冻结A：best epoch {host['best_epoch']} / stop {host['stop_epoch']}；complete-dev Macro-F1 {100*host['metrics']['macro_f1']:.3f}%。","",
      "|方法|状态|Macro-F1|AUROC|NLL|Brier|","|---|---|---:|---:|---:|---:|"]
    for row in rows:
        m=row["metrics"];lines.append(f"|{row['method']}|{row['scenario']}|{100*m['macro_f1']:.3f}%|{100*m['macro_auroc_ovr']:.3f}%|{m['negative_log_likelihood']:.5f}|{m['multiclass_brier']:.5f}|")
    for metric in REGISTERED_METRICS:
        lines+=["",f"### {metric}"]
        for row in stats[metric]["rows"]:
            sim=row["simultaneous_95"];s="未定义（零方差）" if sim is None else f"[{sim[0]:.5f}, {sim[1]:.5f}]"
            lo,hi=row["ordinary_95"];lines.append(f"- {row['arm']} / {row['pattern']}: favorable {row['favorable_improvement']:.5f}; ordinary95 [{lo:.5f},{hi:.5f}]; simultaneous95 {s}; Holm p={row['holm_p_value']:.4g}.")
    (delivery/"README.zh-CN.md").write_text("\n".join(lines)+"\n")
    atomic_write_json({"state":"self_checked_pending_independent_review","identity":sid,"scientific_acceptance":False,"test_access":False,
      "files":{"results.json":file_sha256(delivery/"results.json"),"README.zh-CN.md":file_sha256(delivery/"README.zh-CN.md"),
      "method_identity_audit.json":file_sha256(target/"method_identity_audit.json")}},delivery/"accepted.json")


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--source-commit",required=True);p.add_argument("--action",choices=["init","run","report"],required=True)
    a=p.parse_args()
    if a.action=="init": print(json.dumps(init(a.root,a.source_commit),indent=2))
    elif a.action=="run": run(a.root,a.source_commit)
    else: report(a.root,a.source_commit)


if __name__=="__main__":main()
