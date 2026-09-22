"""Balanced modality-dropout A followed by the ordinary matched LOOK family pipeline."""
from __future__ import annotations
import argparse, copy, fcntl, hashlib, json, os, random, shutil, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from look.data.array_pair import ArrayPair
from look.data.observed_pair import collate_observed
from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle
from look.methods.joint import correction_sites
from look.methods.operator import prepare_complete_pca_bank
from look.methods.family_greedy import fit_family_trajectory
from look.studies.family_search_case import load_bank
from look.studies.project_case import CheckedLoader
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.training.mechanism_training import state_equal
from look.training.balanced_dropout_host import train_balanced_dropout_host, profile_resume
from look.studies.embracenet_delivery import _simultaneous_contrasts, REGISTERED_METRICS

ARMS=("residual_rrr","pca_free_mean");PATTERNS=("oct_missing","cfp_missing")


def read(p): return json.loads(Path(p).read_text())


def init_spec(base_spec_path,source_run,source_root,output,physical_output):
    base_path=Path(base_spec_path);source_run=Path(source_run);source_root=Path(source_root);physical=Path(physical_output);logical=Path(output)
    base=read(base_path);host=read(source_run/"host/accepted.json")
    if file_sha256(base_path)!="f09a452f8584861ff5291cb262894e16f4ba4c7c962f43f6ba4cfbccb18929e8": raise ValueError("Accepted ordinary base spec changed")
    if file_sha256(source_run/"host/accepted.json")!="daef1a7f0871ba21f7c97c42c44bf7ecb11614901dc5a56f61d9c4afaf8c9a7d": raise ValueError("Accepted ordinary host receipt changed")
    if host.get("best_epoch")!=35 or host.get("stop_epoch")!=50 or host.get("test_access") is not False: raise ValueError("Accepted ordinary host identity changed")
    best=source_run/"host/best.pt"
    if file_sha256(best)!="f37eacbc49b7dd45797f63b440781de3dbd419b47ffb9f12deb736206cb297c4": raise ValueError("Accepted ordinary best changed")
    if physical.exists() or logical.exists() or logical.is_symlink(): raise FileExistsError("Balanced-dropout output already exists")
    physical.mkdir(parents=True);logical.parent.mkdir(parents=True,exist_ok=True);logical.symlink_to(physical)
    pins=[]
    for p in sorted((source_root/"src/look").rglob("*.py")): pins.append({"path":str(p.resolve()),"sha256":file_sha256(p)})
    framework=read(source_root/"framework.lock.json")
    for item in framework["files"]:
        p=source_root/item["local_path"]
        if file_sha256(p)!=item["sha256"]: raise ValueError("Framework pin changed")
        pins.append({"path":str(p.resolve()),"sha256":item["sha256"]})
    spec={"schema":"look_fresh_cohort_delivery_v1","study_kind":"balanced_modality_dropout_v1","architecture":"resnet18","position":"deep",
        "factor":16,"rank":32,"seed":3416,"arms":list(ARMS),"search":"positive_forward_tree","test_access":False,
        "data_root":base["data_root"],"data_audit_sha256":base["data_audit_sha256"],"training":base["training"],
        "workspace_bytes":base["workspace_bytes"],"gpu_budget_bytes":base["gpu_budget_bytes"],"gpu_reserve_bytes":base["gpu_reserve_bytes"],
        "ram_budget_bytes":base["ram_budget_bytes"],"disk_reserve_bytes":base.get("disk_reserve_bytes"),"devices":[0],
        "lock_root":"/data/mengh/models/locks","output":str(logical),"physical_output":str(physical),
        "initialization":{"kind":"accepted_native_host_checkpoint","path":str(best.resolve()),"sha256":file_sha256(best),
            "receipt_path":str((source_run/"host/accepted.json").resolve()),"receipt_sha256":file_sha256(source_run/"host/accepted.json"),
            "base_spec_path":str(base_path.resolve()),"base_spec_sha256":file_sha256(base_path),"base_identity":host["identity"],
            "construction":base["initialization"]},
        "source_pins":pins,"source_commit":subprocess.check_output(["git","-C",str(source_root),"rev-parse","HEAD"],text=True).strip(),
        "framework_commit":framework["upstream_commit"],"mask_policy":{"states":["complete","oct_missing","cfp_missing"],"probability_each":1/3,
            "participant_level":True,"all_eyes_same_state":True,"filler":"normalized_mean_zero_before_encoder","bn":"ordinary_train_mode_including_zeroed_branch"},
        "source_host":{"run":str(source_run),"best_epoch":35,"stop_epoch":50,"best_sha256":file_sha256(best),
            "receipt_sha256":file_sha256(source_run/"host/accepted.json"),
            "development_predictions_path":str((source_run/"host/development_predictions.npz").resolve()),
            "development_predictions_sha256":host["files"]["development_predictions.npz"]}}
    atomic_write_json(spec,logical/"spec.json")
    atomic_write_json({"schema":"look_balanced_dropout_init_v1","state":"accepted","identity":stable_hash(spec),"test_access":False,
        "spec_sha256":file_sha256(logical/"spec.json"),"source_host_best_sha256":file_sha256(best)},logical/"init_accepted.json")
    return spec


def _base_spec(s):
    init=s["initialization"];p=Path(init["base_spec_path"])
    if file_sha256(p)!=init["base_spec_sha256"]: raise ValueError("Base spec changed")
    return read(p)


def validate(s):
    from look.training.observed_host import validate_config
    if s.get("schema")!="look_fresh_cohort_delivery_v1" or s.get("study_kind")!="balanced_modality_dropout_v1" or s.get("test_access") is not False:
        raise ValueError("Undeclared balanced-dropout study")
    if (s["architecture"],s["position"],s["factor"],s["rank"],s["seed"],s["arms"],s["search"])!=("resnet18","deep",16,32,3416,list(ARMS),"positive_forward_tree"):
        raise ValueError("Balanced-dropout scope changed")
    validate_config(s["training"])
    if s["mask_policy"]!={"states":["complete","oct_missing","cfp_missing"],"probability_each":1/3,"participant_level":True,
        "all_eyes_same_state":True,"filler":"normalized_mean_zero_before_encoder","bn":"ordinary_train_mode_including_zeroed_branch"}:
        raise ValueError("Balanced-dropout mask policy changed")
    init=s["initialization"]
    if init["kind"]!="accepted_native_host_checkpoint" or file_sha256(init["path"])!=init["sha256"] or file_sha256(init["receipt_path"])!=init["receipt_sha256"]:
        raise ValueError("Balanced-dropout source checkpoint changed")
    receipt=read(init["receipt_path"])
    if receipt.get("best_epoch")!=35 or receipt.get("stop_epoch")!=50 or receipt.get("identity")!=init["base_identity"] or receipt["files"]["best.pt"]!=init["sha256"]:
        raise ValueError("Balanced-dropout source receipt changed")
    source_host=s["source_host"]
    if (receipt["files"]["development_predictions.npz"]!=source_host["development_predictions_sha256"]
            or file_sha256(source_host["development_predictions_path"])!=source_host["development_predictions_sha256"]):
        raise ValueError("Balanced-dropout source development predictions changed")
    base=_base_spec(s)
    if base["training"]!=s["training"] or base["data_root"]!=s["data_root"] or base["data_audit_sha256"]!=s["data_audit_sha256"]:
        raise ValueError("Balanced-dropout source training/data changed")
    if file_sha256(Path(s["data_root"])/"accepted.json")!=s["data_audit_sha256"]: raise ValueError("Data audit changed")
    for row in s["source_pins"]:
        if file_sha256(row["path"])!=row["sha256"]: raise ValueError("Source pin changed")


def make_graph(s):
    from look.studies.cohort_delivery import make_graph as make_fresh
    base=_base_spec(s);g=make_fresh(base)
    state=torch.load(s["initialization"]["path"],map_location="cpu",weights_only=False)
    ids=[(n.id,n.name) for n in sorted(g.nodes,key=lambda n:n.id)]
    if state["node_ids"]!=ids: raise ValueError("Accepted ordinary host node IDs changed")
    g.load_state_dict(state["model"],strict=True)
    g.native_host_provenance=copy.deepcopy(g.native_host_provenance)
    g.native_host_provenance["balanced_dropout_initialization"]={"source_best_sha256":s["initialization"]["sha256"],"source_best_epoch":35}
    return g


def load_selected(s,root):
    r=read(root/"host/accepted.json")
    if r["identity"]!=stable_hash(s) or r["state"]!="accepted": raise ValueError("Balanced-dropout host unaccepted")
    for name,digest in r["files"].items():
        if file_sha256(root/"host"/name)!=digest: raise ValueError("Balanced-dropout host evidence changed")
    g=make_graph(s);state=torch.load(root/"host/best.pt",map_location="cpu",weights_only=False)
    g.load_state_dict(state["model"],strict=True);g.eval()
    for p in g.parameters():p.requires_grad_(False)
    return g,r


def _loader(ds,s,check=lambda:False):
    return CheckedLoader(DataLoader(ds,batch_size=s["training"]["microbatch"],shuffle=False,num_workers=0,
        collate_fn=collate_observed,generator=torch.Generator().manual_seed(s["seed"])),check)


def work(s,root,stage):
    import psutil
    validate(s);root=Path(root);identity=stable_hash(s);stop=False
    def request(*_):
        nonlocal stop;stop=True
    signal.signal(signal.SIGTERM,request);signal.signal(signal.SIGUSR1,request)
    props=torch.cuda.get_device_properties(0)
    if torch.cuda.mem_get_info()[0]<s["gpu_reserve_bytes"]+s["gpu_budget_bytes"]: raise MemoryError("GPU0 exclusive budget unavailable")
    torch.cuda.set_per_process_memory_fraction(s["gpu_budget_bytes"]/props.total_memory);torch.set_num_threads(2)
    torch.manual_seed(s["seed"]);np.random.seed(s["seed"]);random.seed(s["seed"]);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def check():
        if s.get("disk_reserve_bytes") and shutil.disk_usage(root).free<s["disk_reserve_bytes"]: raise OSError("Disk reserve")
        if psutil.Process().memory_info().rss>.85*s["ram_budget_bytes"]: raise MemoryError("Host reserve")
        if torch.cuda.mem_get_info()[0]<s["gpu_reserve_bytes"]: raise MemoryError("Device reserve")
        return stop
    train=ArrayPair(s["data_root"],"train",augment=True,seed=s["seed"]);fit=ArrayPair(s["data_root"],"train");dev=ArrayPair(s["data_root"],"development")
    if set(fit.participant_ids)&set(dev.participant_ids): raise ValueError("Train/dev overlap")
    if stage=="profile":
        # Before any update, prove the accepted ordinary A is reproduced exactly.
        source=np.load(s["source_host"]["development_predictions_path"],allow_pickle=False)
        initial=evaluate_missing(make_graph(s),_loader(dev,s,check),torch.device("cuda:0"),fixed_pattern="complete")
        if (not np.array_equal(source["participant_ids"].astype(str),initial["participant_ids"].astype(str))
                or not np.array_equal(source["labels"],initial["labels"])
                or not np.array_equal(source["logits"],initial["logits"])):
            raise ValueError("Balanced-dropout epoch0 does not exactly replay accepted ordinary A")
        atomic_write_json({"schema":"look_balanced_dropout_epoch0_replay_v1","state":"accepted","identity":identity,
            "source_predictions_sha256":s["source_host"]["development_predictions_sha256"],
            "participants":len(initial["participant_ids"]),"logits_exact":True,"test_access":False},
            root/"profile/epoch0_replay.json")
        r=profile_resume(lambda:make_graph(s),train,dev,s["training"],s["seed"],root/"profile",identity,torch.device("cuda:0"),check)
        if r["peak_reserved_bytes"]>s["gpu_budget_bytes"]: raise MemoryError("Balanced-dropout profile budget")
    elif stage=="host":
        if read(root/"profile/accepted.json")["identity"]!=identity: raise ValueError("Profile identity changed")
        g=make_graph(s);r=train_balanced_dropout_host(g,train,dev,s["training"],s["seed"],root/"host",identity,torch.device("cuda:0"),check)
        if r.get("state")!="accepted": raise RuntimeError("Balanced-dropout host not accepted")
    else:
        g,h=load_selected(s,root);frozen={k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in g.state_dict().items()}
        bank=prepare_complete_pca_bank(g,_loader(fit,s,check),correction_sites(g),[16],32,torch.device("cuda:0"),root/"pca",
            {"case":identity,"host_best_sha256":h["files"]["best.pt"]},root/"quarantine",load_only=stage!="pca",strict_rank=True)
        if stage=="pca": pass
        elif stage=="fit_profile":
            # Use the standard ordinary-host fitting profile on the new frozen A.
            from look.studies.fusion_cache_revalidation import revalidate_or_fit
            records=[];raw=root/"profile/fitting";reval=root/"profile/revalidated";reval.mkdir(parents=True,exist_ok=True)
            def role_loader(role): return _loader(fit if role=="train" else dev,s,check)
            def fresh_graph(): return load_selected(s,root)[0]
            for arm in ARMS:
                for pattern in PATTERNS:
                    source=raw/arm/pattern;target=reval/arm/pattern
                    arts,result,receipt=revalidate_or_fit(source=source,target=target,graph=g,loader=role_loader,device=torch.device("cuda:0"),
                        arm=arm,pattern=pattern,sites=correction_sites(g),factor=16,pca_bank=bank,identity=identity,
                        workspace_bytes=s["workspace_bytes"],should_pause=check,load_fresh_graph=fresh_graph)
                    receipt_path=target.parent/"audit"/target.name/"accepted.json"
                    records.append({"arm":arm,"pattern":pattern,"revalidated_root":str(target.relative_to(root)),
                        "revalidation_receipt":str(receipt_path.relative_to(root)),"revalidation_receipt_sha256":file_sha256(receipt_path),
                        "selection_sha256":file_sha256(target/"selection.json"),"bank_sha256":file_sha256(target/"bank.pt")})
            atomic_write_json({"schema":"look_balanced_dropout_fit_profile_v1","state":"accepted","identity":identity,"test_access":False,
                "methods":list(ARMS),"patterns":list(PATTERNS),"development":len(dev),"records":records,
                "gpu_peak_reserved_bytes":torch.cuda.max_memory_reserved()},root/"profile/fitting/accepted.json")
        elif stage in ARMS:
            records=[]
            for pattern in PATTERNS:
                correction=root/"corrections"/stage/pattern
                arts,_=fit_family_trajectory(g,_loader(fit,s,check),_loader(dev,s,check),arm=stage,pattern=pattern,sites=correction_sites(g),
                    factor=16,candidates=[{"rank":32,"ridge_lambda":None}],pca_bank=bank,identity=identity,output=correction,
                    device=torch.device("cuda:0"),workspace_bytes=s["workspace_bytes"],mode="positive_forward_tree",should_pause=check,
                    penalty_policy="prefix_train_pca_gcv")
                a=evaluate_missing(g,_loader(dev,s,check),torch.device("cuda:0"),fixed_pattern=pattern,artifact_banks={pattern:arts})
                b=evaluate_missing(g,_loader(dev,s,check),torch.device("cuda:0"),fixed_pattern=pattern)
                for method,result in ((stage,a),("host",b)):
                    p=root/stage/"development"/f"{method}_{pattern}.npz";save_prediction_bundle(result,p)
                    records.append({"method":method,"scenario":pattern,"path":str(p),"sha256":file_sha256(p),"metrics":result["metrics"]})
            atomic_write_json({"state":"accepted","identity":identity,"test_access":False,"records":records,
                "host_best_sha256":h["files"]["best.pt"],"replay_exact":True},root/stage/"accepted.json")
        else: raise ValueError("Unknown balanced-dropout stage")
        current=g.state_dict()
        for k,v in frozen.items():
            if torch.is_tensor(v) and not torch.equal(v,current[k].detach().cpu()): raise ValueError("Frozen A changed during LOOK stage")
    atomic_write_json({"stage":stage,"state":"completed","identity":identity,"time":time.time(),"test_access":False},root/(stage+"_status.json"))


def _registered_contrast_definitions():
    return [
        {
            "arm": arm,
            "pattern": pattern,
            "method_key": (arm, pattern),
            "reference_key": ("host", pattern),
        }
        for pattern in PATTERNS
        for arm in ARMS
    ]


def report(s,root):
    validate(s);root=Path(root);identity=stable_hash(s);rows=[];arrays={}
    for arm in ARMS:
        r=read(root/arm/"accepted.json")
        if r["identity"]!=identity or r["state"]!="accepted": raise ValueError("Unaccepted balanced-dropout arm")
        for x in r["records"]:
            if file_sha256(x["path"])!=x["sha256"]: raise ValueError("Balanced-dropout prediction changed")
            a=dict(np.load(x["path"],allow_pickle=False));key=(x["method"],x["scenario"])
            if key in arrays:
                if not all(np.array_equal(arrays[key][k],a[k]) for k in a): raise ValueError("Balanced-dropout baseline differs")
            else: arrays[key]=a;rows.append({"method":key[0],"scenario":key[1],"metrics":x["metrics"],"sha256":x["sha256"]})
    ref=next(iter(arrays.values()))
    for a in arrays.values():
        for k in ("participant_ids","labels"):
            if not np.array_equal(ref[k],a[k]): raise ValueError("Balanced-dropout report identity mismatch")
    definitions=_registered_contrast_definitions()
    for definition in definitions:
        if definition["method_key"] not in arrays or definition["reference_key"] not in arrays:
            raise ValueError("Balanced-dropout registered contrast predictions missing")
    logits={(m,p):v["logits"].astype(np.float64) for (m,p),v in arrays.items()}
    stats=_simultaneous_contrasts(ref["labels"].astype(np.int64),logits,definitions,10000,7341618)
    out=root/"delivery";out.mkdir(exist_ok=True)
    host=read(root/"host/accepted.json")
    payload={"schema":"look_balanced_dropout_results_v1","state":"self_checked_pending_independent_review","identity":identity,
        "test_access":False,"host_complete_metrics":host["metrics"],"development_results":rows,"paired_statistics":stats,
        "selection_bias":"same 296-person development selects A checkpoint and LOOK trees; intervals conditional/exploratory",
        "uncertainty_limit":"single seed; participant bootstrap excludes training-seed variation"}
    atomic_write_json(payload,out/"results.json")
    lines=["# 三态缺失鲁棒微调 A + LOOK 单种子候选","",f"冻结A：best epoch {host['best_epoch']} / stop {host['stop_epoch']}；complete-dev Macro-F1 {100*host['metrics']['macro_f1']:.3f}%。",
        "A由accepted ordinary R18 deep best35继续微调；train每participant完整/缺OCT/缺CFP各1/3随机三态，缺失输入在encoder前置零；同participant各眼一致。没有full-only再训练对照，因此不宣称微调A本身优于普通训练。","",
        "|方法|状态|Macro-F1|AUROC|NLL|Brier|","|---|---|---:|---:|---:|---:|"]
    for row in rows:
        m=row["metrics"];lines.append(f"|{row['method']}|{row['scenario']}|{100*m['macro_f1']:.3f}%|{100*m['macro_auroc_ovr']:.3f}%|{m['negative_log_likelihood']:.5f}|{m['multiclass_brier']:.5f}|")
    lines+=["","四个主比较均为同一冻结A下A+LOOK−A；开发集同时选A/tree并估效应，不能视作独立确认。"]
    for metric in REGISTERED_METRICS:
        lines+=["",f"### {metric}"]
        for row in stats[metric]["rows"]:
            sim=row["simultaneous_95"];st="未定义（零方差恒等对比）" if sim is None else f"[{sim[0]:.5f}, {sim[1]:.5f}]"
            lo,hi=row["ordinary_95"];lines.append(f"- {row['arm']} / {row['pattern']}: 改善 {row['favorable_improvement']:.5f}; ordinary95 [{lo:.5f},{hi:.5f}]; 同族同时95 {st}; Holm p={row['holm_p_value']:.4g}.")
    (out/"README.zh-CN.md").write_text("\n".join(lines)+"\n")
    atomic_write_json({"schema":"look_balanced_dropout_delivery_v1","state":"self_checked_pending_independent_review","identity":identity,
        "scientific_acceptance":False,"test_access":False,"files":{"results.json":file_sha256(out/"results.json"),
        "README.zh-CN.md":file_sha256(out/"README.zh-CN.md")}},out/"accepted.json")


def stage_receipt(root,stage):
    return {"profile":root/"profile/accepted.json","host":root/"host/accepted.json","pca":root/"pca/accepted.json",
        "fit_profile":root/"profile/fitting/accepted.json","residual_rrr":root/"residual_rrr/accepted.json",
        "pca_free_mean":root/"pca_free_mean/accepted.json","report":root/"delivery/accepted.json"}[stage]


def pipeline(spec_path):
    s=read(spec_path);validate(s);root=Path(s["output"])
    with (root/"pipeline.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_write_json({"state":"running","identity":stable_hash(s),"time":time.time(),"test_access":False},root/"status.json")
        try:
            for stage in ("profile","host","pca","fit_profile","residual_rrr","pca_free_mean"):
                rp=stage_receipt(root,stage)
                if rp.exists():
                    if read(rp).get("identity")!=stable_hash(s) or read(rp).get("state")!="accepted": raise ValueError("Existing stage receipt changed: "+stage)
                    continue
                env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONHASHSEED=str(s["seed"]))
                with (root/f"{stage}.log").open("a") as log:
                    code=subprocess.run([sys.executable,"-m","look.studies.balanced_dropout_delivery","--spec",str(spec_path),"--stage",stage],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
                if code: raise RuntimeError(f"Balanced-dropout stage {stage} failed ({code})")
            report(s,root)
            atomic_write_json({"state":"completed","identity":stable_hash(s),"time":time.time(),"test_access":False},root/"status.json")
        except Exception as e:
            import traceback
            atomic_write_json({"state":"needs_review","identity":stable_hash(s),"error":repr(e),"traceback":traceback.format_exc(),"time":time.time(),"test_access":False},root/"status.json")
            raise


def main():
    p=argparse.ArgumentParser();p.add_argument("--spec");p.add_argument("--stage",choices=["init","pipeline","profile","host","pca","fit_profile",*ARMS,"report"])
    p.add_argument("--base-spec");p.add_argument("--source-run");p.add_argument("--source-root");p.add_argument("--output");p.add_argument("--physical-output")
    a=p.parse_args()
    if a.stage=="init":
        value=init_spec(a.base_spec,a.source_run,a.source_root,a.output,a.physical_output);print(json.dumps(value,ensure_ascii=False,indent=2));return
    if not a.spec: raise ValueError("--spec required")
    s=read(a.spec);root=Path(s["output"])
    if a.stage=="pipeline": pipeline(a.spec)
    elif a.stage=="report": report(s,root)
    else:
        uuid=subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader","-i",os.environ["CUDA_VISIBLE_DEVICES"]],text=True).strip()
        locks=Path(s["lock_root"]);locks.mkdir(parents=True,exist_ok=True)
        with (locks/(uuid+".lock")).open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);work(s,root,a.stage)


if __name__=="__main__": main()
