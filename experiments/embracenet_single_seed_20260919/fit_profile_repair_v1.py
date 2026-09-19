"""Management-only R3 fit-profile acceptance overlay for a frozen EmbraceNet science commit."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--spec",required=True)
    parser.add_argument("--scientific-source",required=True)
    args=parser.parse_args()
    spec_path=Path(args.spec).resolve()
    source=Path(args.scientific_source).resolve()
    sys.path.insert(0,str(source/"src"))

    from look.data.array_pair import ArrayPair
    from look.evaluation.embracenet import availability_matrix,evaluate_single_missing
    from look.methods.affine_family import FamilyArtifact,fit_map
    from look.methods.embracenet_family import EmbraceNetFamilyStatistics
    from look.methods.independent_greedy import SelectionPaused
    from look.methods.joint import correction_sites
    from look.methods.operator import LOOKArtifact,_gcv_lambda
    from look.methods.linear_operator import fingerprint
    from look.runtime.host_checkpoint import atomic_save
    from look.runtime.state import atomic_write_json,stable_hash
    from look.studies import embracenet_delivery as delivery
    from look.studies.embracenet_adapter import forward_embracenet_with_look

    spec=json.loads(spec_path.read_text())
    delivery.validate(spec)
    source_head=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    if source_head!=spec["source_commit"]:
        raise ValueError("Frozen scientific source changed")

    overlay_root=Path(__file__).resolve().parents[2]
    overlay_commit=subprocess.check_output(["git","-C",str(overlay_root),"rev-parse","HEAD"],text=True).strip()
    critical=[
        "src/look/models/embracenet.py",
        "src/look/evaluation/embracenet.py",
        "src/look/training/embracenet_host.py",
        "src/look/methods/embracenet_family.py",
        "src/look/studies/embracenet_delivery.py",
    ]
    scientific_shas={rel:file_sha(source/rel) for rel in critical}
    overlay_shas={rel:file_sha(overlay_root/rel) for rel in critical}
    if scientific_shas!=overlay_shas:
        raise ValueError("Overlay changed scientific/numerical modules")

    root=Path(spec["output"])
    target=root/"fit_profile"
    target.mkdir(parents=True,exist_ok=True)
    if (target/"accepted.json").exists():
        receipt=json.loads((target/"accepted.json").read_text())
        if receipt.get("identity")!=stable_hash(spec):
            raise ValueError("Existing fit-profile identity differs")
        print(json.dumps({"reused":True,"receipt_sha256":file_sha(target/"accepted.json")}))
        return

    stop=False
    def request(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request);signal.signal(signal.SIGUSR1,request)
    def check():
        return delivery._resource_guard(spec,root,lambda:stop)

    uuid=subprocess.check_output(
        ["nvidia-smi","--query-gpu=uuid","--format=csv,noheader","-i","0"],text=True
    ).strip()
    lock_root=Path(spec["lock_root"]);lock_root.mkdir(parents=True,exist_ok=True)
    with (lock_root/(uuid+".lock")).open("a") as device_lock:
        fcntl.flock(device_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if check(): raise SelectionPaused()

        torch.set_num_threads(2);torch.manual_seed(spec["seed"]);np.random.seed(spec["seed"]);random.seed(spec["seed"])
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.reset_peak_memory_stats()

        graph,_=delivery._load_selected(spec,root,torch.device("cuda:0"))
        fit=ArrayPair(spec["data_root"],"train")
        dev=ArrayPair(spec["data_root"],"development")
        pca=delivery._pca_bank(spec,root,graph,fit,check)
        sites=correction_sites(graph)
        bases={site:next(b for (name,_),b in pca.items() if name==site) for site in sites}
        max_site=max(sites,key=lambda site:bases[site].std.numel())
        max_dim=int(bases[max_site].std.numel())
        effective_site="embraced_feature"
        effective_dim=int(bases[effective_site].std.numel())
        rank=spec["look"]["rank"]
        if min(max_dim,effective_dim)<rank:
            raise ValueError("R3 representative sites cannot support rank32")

        subset=delivery.ProbeSubset(fit,64)
        raw_loader=delivery._loader(subset,spec)

        class StopAfterOne:
            def __init__(self):self.calls=0;self.fired=False
            def __call__(self):
                if check():return True
                self.calls+=1
                if not self.fired and self.calls>=2:
                    self.fired=True;return True
                return False

        resume_root=target/"overlay_resume_probe"
        continuous_root=target/"overlay_continuous_probe"
        identity={"case":stable_hash(spec),"purpose":"fit_profile_overlay_resume","site":max_site,"rank":rank}
        interrupted=EmbraceNetFamilyStatistics(
            graph,raw_loader,{max_site:bases[max_site]},torch.device("cuda:0"),
            resume_root,identity,spec["workspace_bytes"],rank,StopAfterOne(),projected_ranks=(rank,)
        )
        try:
            interrupted.statistics("oct_missing",(),[max_site])
            raise ValueError("R3 interruption did not pause")
        except SelectionPaused:
            if check():raise
        partials=list(resume_root.glob("*.pt"))
        if len(partials)!=1:raise ValueError("R3 interruption did not create one resume artifact")
        partial=torch.load(partials[0],map_location="cpu",weights_only=False)["payload"]
        partial_batches=len(partial["batch_stamps"])
        if partial.get("complete") is not False or partial_batches<1:
            raise ValueError("R3 interruption checkpoint is not a nonempty partial state")

        resumed=EmbraceNetFamilyStatistics(
            graph,raw_loader,{max_site:bases[max_site]},torch.device("cuda:0"),
            resume_root,identity,spec["workspace_bytes"],rank,check,projected_ranks=(rank,)
        )
        resumed_stats=resumed.statistics("oct_missing",(),[max_site])
        continuous=EmbraceNetFamilyStatistics(
            graph,raw_loader,{max_site:bases[max_site]},torch.device("cuda:0"),
            continuous_root,identity,spec["workspace_bytes"],rank,check,projected_ranks=(rank,)
        )
        continuous_stats=continuous.statistics("oct_missing",(),[max_site])
        if fingerprint(vars(resumed_stats[max_site]))!=fingerprint(vars(continuous_stats[max_site])):
            raise ValueError("R3 resumed full moments differ from uninterrupted")
        if fingerprint(vars(resumed.projected[max_site][rank]))!=fingerprint(vars(continuous.projected[max_site][rank])):
            raise ValueError("R3 resumed projected moments differ from uninterrupted")

        dev_loader=delivery._loader(dev,spec)
        first_batch=next(iter(dev_loader))
        availability=availability_matrix(len(first_batch["label"]),"oct_missing",torch.device("cuda:0"))
        max_base=forward_embracenet_with_look(
            graph,first_batch["oct"].cuda(),first_batch["cfp"].cuda(),first_batch["counts"],
            availability,stop_node=max_site
        )["output"].detach()

        max_basis=bases[max_site];stats=resumed_stats[max_site]
        q_basis=max_basis.components[:rank].double()
        projected=resumed.projected[max_site][rank]
        ridge=_gcv_lambda(
            q_basis@stats.cxx@q_basis.T,q_basis@stats.cxy@q_basis.T,
            float(projected.syy),stats.count,rank
        )
        max_rows=[]
        for arm in ("pca_free_mean","residual_rrr"):
            mapping=fit_map(stats,rank,ridge,arm=arm,basis=max_basis.components[:rank])
            base=LOOKArtifact(
                max_site,"oct_missing","availability_zero_mask",max_basis.factor,rank,
                max_basis.feature_shape,max_basis.downsample_shape,max_basis.mean,max_basis.std,
                max_basis.pca_mean,max_basis.components[:rank],torch.zeros(rank,rank),torch.zeros(rank),
                ridge,0.,0.,float(max_basis.explained_variance_ratio[:rank].sum()),
                max_basis.fit_seconds,max_basis.peak_rss_bytes,max_basis.source_id,
                max_basis.member_names,max_basis.member_shapes,max_basis.protocol,max_basis.split_rule,
                max_basis.spatial_method
            )
            artifact=FamilyArtifact(base,mapping)
            path=target/f"overlay_max_{arm}_{max_site}_rank{rank}.pt"
            atomic_save(path,artifact.record())
            loaded=FamilyArtifact.from_record(torch.load(path,map_location="cpu",weights_only=False))
            corrected=loaded.apply_feature(max_base)
            delta=float((corrected-max_base).abs().max())
            if not math.isfinite(delta) or delta<=0:raise ValueError("Max-dimensional solver/writeback is zero")
            max_rows.append({"arm":arm,"site":max_site,"dimension":max_dim,"rank":rank,
                "ridge_lambda":float(ridge),"artifact":str(path),"sha256":file_sha(path),
                "site_max_abs_change":delta})

        effective_identity={"case":stable_hash(spec),"purpose":"fit_profile_overlay_effective","site":effective_site,"rank":rank}
        effective=EmbraceNetFamilyStatistics(
            graph,raw_loader,{effective_site:bases[effective_site]},torch.device("cuda:0"),
            target/"overlay_effective_moments",effective_identity,spec["workspace_bytes"],rank,check,
            projected_ranks=(rank,)
        )
        effective_stats=effective.statistics("oct_missing",(),[effective_site])
        e_basis=bases[effective_site];e_stats=effective_stats[effective_site]
        e_q=e_basis.components[:rank].double();e_projected=effective.projected[effective_site][rank]
        e_ridge=_gcv_lambda(
            e_q@e_stats.cxx@e_q.T,e_q@e_stats.cxy@e_q.T,
            float(e_projected.syy),e_stats.count,rank
        )
        baseline=evaluate_single_missing(
            graph,dev_loader,torch.device("cuda:0"),"oct_missing",should_pause=check
        )
        effective_rows=[]
        for arm in ("pca_free_mean","residual_rrr"):
            mapping=fit_map(e_stats,rank,e_ridge,arm=arm,basis=e_basis.components[:rank])
            base=LOOKArtifact(
                effective_site,"oct_missing","availability_zero_mask",e_basis.factor,rank,
                e_basis.feature_shape,e_basis.downsample_shape,e_basis.mean,e_basis.std,e_basis.pca_mean,
                e_basis.components[:rank],torch.zeros(rank,rank),torch.zeros(rank),e_ridge,0.,0.,
                float(e_basis.explained_variance_ratio[:rank].sum()),e_basis.fit_seconds,
                e_basis.peak_rss_bytes,e_basis.source_id,e_basis.member_names,e_basis.member_shapes,
                e_basis.protocol,e_basis.split_rule,e_basis.spatial_method
            )
            artifact=FamilyArtifact(base,mapping)
            path=target/f"overlay_effective_{arm}_{effective_site}_rank{rank}.pt"
            atomic_save(path,artifact.record())
            loaded=FamilyArtifact.from_record(torch.load(path,map_location="cpu",weights_only=False))
            corrected=evaluate_single_missing(
                graph,dev_loader,torch.device("cuda:0"),"oct_missing",[loaded],should_pause=check
            )
            replay=evaluate_single_missing(
                graph,dev_loader,torch.device("cuda:0"),"oct_missing",
                [FamilyArtifact.from_record(torch.load(path,map_location="cpu",weights_only=False))],
                should_pause=check
            )
            if not np.array_equal(corrected["logits"],replay["logits"]):
                raise ValueError("Effective profile save/load replay differs")
            delta=float(np.max(np.abs(corrected["logits"]-baseline["logits"])))
            if not math.isfinite(delta) or delta<=0:raise ValueError("Effective profile full-dev correction is zero")
            effective_rows.append({"arm":arm,"site":effective_site,"dimension":effective_dim,"rank":rank,
                "ridge_lambda":float(e_ridge),"artifact":str(path),"sha256":file_sha(path),
                "full_dev_max_abs_logit_change":delta,"metrics":corrected["metrics"],
                "save_load_replay_exact":True})

        if check():raise SelectionPaused()
        peak=int(torch.cuda.max_memory_reserved())
        if peak>spec["gpu_budget_bytes"]:raise MemoryError("R3 overlay profile exceeded GPU budget")
        receipt={
            "schema":"look_embracenet_fit_profile_overlay_v1","state":"accepted",
            "identity":stable_hash(spec),"scientific_source_commit":source_head,
            "overlay_commit":overlay_commit,"overlay_source_sha256":file_sha(__file__),
            "critical_scientific_shas":scientific_shas,"overlay_matches_scientific":True,
            "probe_train_participants":len(subset),"resume_partial_batches":partial_batches,
            "fit_resume_vs_uninterrupted_exact":True,"projected_resume_vs_uninterrupted_exact":True,
            "representative_max_site":max_site,"representative_max_dimension":max_dim,
            "max_dimension_solver_rows":max_rows,
            "effective_site":effective_site,"effective_dimension":effective_dim,
            "effective_full_dev_rows":effective_rows,
            "baseline_full_dev_metrics":baseline["metrics"],
            "gpu_peak_reserved_bytes":peak,"gpu_budget_bytes":spec["gpu_budget_bytes"],
            "storage_policy":spec["storage"],"test_access":False,
            "scientific_acceptance":False,
            "topology_note":"Pre-embrace corrections can change the missing-branch site feature yet be masked by EmbraceNet availability; full-dev nonzero propagation is therefore additionally required and demonstrated at embraced_feature."
        }
        atomic_write_json(receipt,target/"accepted.json")
        atomic_write_json({"state":"completed","identity":stable_hash(spec),"time":time.time(),"test_access":False},
                          root/"fit_profile_status.json")
        print(json.dumps({"accepted":True,"receipt_sha256":file_sha(target/"accepted.json"),
                          "peak":peak,"max_site":max_site,"effective_site":effective_site}))
if __name__=="__main__":
    try:main()
    except SelectionPaused:
        raise SystemExit(75)
