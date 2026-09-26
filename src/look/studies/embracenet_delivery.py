"""Authorized finite WS02 EmbraceNet single-seed delivery: A -> four LOOK trees -> report."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import logsumexp
from torch.utils.data import DataLoader, Dataset

from look.data.array_pair import ArrayPair
from look.data.observed_pair import collate_observed
from look.evaluation.embracenet import (
    EvaluationPaused,
    evaluate_complete_analytical,
    evaluate_single_missing,
)
from look.evaluation.evaluator import save_prediction_bundle
from look.evaluation.stability import logit_metrics, probabilities_from_logits
from look.evaluation.metrics import holm_adjust
from look.methods.embracenet_family import (
    EmbraceNetFamilyStatistics,
    fit_embracenet_family_trajectory,
    load_embracenet_bank,
    prepare_embracenet_pca_bank,
)
from look.methods.affine_family import FamilyArtifact, fit_map
from look.methods.independent_greedy import SelectionPaused
from look.methods.operator import LOOKArtifact, _gcv_lambda
from look.methods.joint import correction_sites
from look.models.embracenet import (
    build_embracenet_host,
    capture_embracenet_sampling,
    restore_embracenet_sampling,
)
from look.models.observed_participant import ObservedParticipantModel
from look.runtime.host_checkpoint import atomic_save, capture_rng, cpu_tree, restore_rng, read_selected
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.studies.project_case import CheckedLoader, Paused as LoaderPaused
from look.training.embracenet_host import train_embracenet_host


SCHEMA = "look_embracenet_single_seed_v1"
TASK_ID = "look-embracenet-single-seed-20260919-v1"
PACKET_SHA = "fd3300c0cdbfa3d168eca5e9e4563b4c5e3d9fdf4dc7e0111bdea89ccd3339f1"
AUTHORIZATION_SHA = "bad1710372f092f662c87e4d41e60c591decb0ab8dccadb0a75934e91313bd01"
DECISION_SHA = "0893255b914979e67212876b653cb601e7573c8413b9490b1d16f30cfc81d14c"
PHD_RESEARCH_SHA = "1d24eddf2f939a82ca3ed821c803739a727b8f403d20457e22291a28530ddb2e"
ARMS = ("pca_free_mean", "residual_rrr")
PATTERNS = ("oct_missing", "cfp_missing")
RUN_ID = "embracenet_single_seed_20260919_v1"
REGISTERED_METRICS = ("macro_f1", "macro_auroc_ovr", "negative_log_likelihood", "multiclass_brier")


class StagePaused(Exception):
    """Owning signal/resource callback requested a resumable stage pause."""


def read(path):
    return json.loads(Path(path).read_text())


def _source_pins(source_root):
    source_root = Path(source_root).resolve()
    paths = [
        *sorted((source_root / "src/look").rglob("*.py")),
        source_root / "pyproject.toml",
        source_root / "workspace/PHD_RESEARCH_STANDARD.md",
    ]
    import mhd_framework
    mhd_root = Path(mhd_framework.__file__).resolve().parent
    paths += sorted(mhd_root.rglob("*.py"))
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({"path": str(path), "sha256": file_sha256(path)})
    return rows


def initialize_spec(base_spec, source_root, output, physical_output):
    base = read(base_spec)
    expected_training = {
        "epochs": 100, "patience": 15, "minimum_epochs": 8, "warmup_epochs": 5,
        "microbatch": 16, "effective_batch": 128, "pretrained_lr": 1e-4,
        "new_layer_lr": 1e-3, "weight_decay": 1e-4, "clip": 5.0,
        "precision": "fp32", "num_workers": 0, "loss": "unweighted_cross_entropy",
        "primary_metric": "macro_f1",
    }
    if base.get("seed") != 3416 or base.get("training") != expected_training or base.get("test_access") is not False:
        raise ValueError("Accepted base R18 training identity changed")
    data_name = Path(base.get("data_root", "")).name
    if base.get("architecture") != "resnet18" or data_name != "ukb_small_20260909_v1":
        raise ValueError("Accepted small-cohort identity changed")
    source_root = Path(source_root).resolve()
    commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    phd = source_root / "workspace/PHD_RESEARCH_STANDARD.md"
    if file_sha256(phd) != PHD_RESEARCH_SHA:
        raise ValueError("Synced PHD research standard changed")
    logical_output = Path(output).absolute()
    physical_output = Path(physical_output).resolve()
    backup_root = Path("/backup/mengh/LOOK").resolve()
    if not physical_output.is_relative_to(backup_root):
        raise ValueError("EmbraceNet formal artifacts must use the accepted /backup/mengh/LOOK physical policy")
    physical_output.mkdir(parents=True, exist_ok=True)
    logical_output.parent.mkdir(parents=True, exist_ok=True)
    if logical_output.exists() or logical_output.is_symlink():
        if not logical_output.is_symlink() or logical_output.resolve() != physical_output:
            raise ValueError("Existing logical EmbraceNet run path does not bind the declared backup target")
    else:
        logical_output.symlink_to(physical_output, target_is_directory=True)
    spec = {
        "schema": SCHEMA, "task_id": TASK_ID, "run_id": RUN_ID, "test_access": False,
        "seed": 3416, "architecture": "resnet18", "embracement_size": 256,
        "data_name": data_name, "data_root": base["data_root"],
        "data_audit_sha256": base["data_audit_sha256"],
        "initialization": base["initialization"], "training": base["training"],
        "source_commit": commit, "source_root": str(source_root),
        "framework_commit": base["framework_commit"], "source_pins": _source_pins(source_root),
        "packet_sha256": PACKET_SHA, "authorization_sha256": AUTHORIZATION_SHA,
        "trial_decision_sha256": DECISION_SHA, "phd_research_sha256": PHD_RESEARCH_SHA,
        "recipe": {
            "training_states": {"complete": 1/3, "oct_missing": 1/3, "cfp_missing": 1/3},
            "complete_selection": "analytical_integrated_mean_logits",
            "single_missing_evaluation": "one_author_forward",
            "complete_pca_reference": "analytic_author_stochastic_second_moment",
            "complete_selection_probabilities": [0.5, 0.5],
        },
        "look": {
            "arms": list(ARMS), "patterns": list(PATTERNS), "search": "positive_forward_tree",
            "factor": 16, "rank": 32, "penalty_policy": "prefix_train_pca_gcv",
            "sites": [
                "joint_input", "joint_stem", "joint_stage1", "joint_stage2", "joint_stage3",
                "joint_stage4", "joint_features", "joint_participant_feature", "embraced_feature",
            ],
        },
        "bootstrap": {"iterations": 10000, "seed": 3416, "metrics": list(REGISTERED_METRICS)},
        "devices": [0], "lock_root": base["lock_root"],
        "gpu_budget_bytes": base["gpu_budget_bytes"], "gpu_reserve_bytes": 0,
        "ram_budget_bytes": base["ram_budget_bytes"], "workspace_bytes": base["workspace_bytes"],
        "host_free_fraction_min": 0.15, "disk_reserve_bytes": 10 * 1024**3,
        "storage": {
            "policy": "backup_physical_with_data_logical_symlink_v1",
            "logical_output": str(logical_output),
            "physical_output": str(physical_output),
            "physical_root": str(backup_root),
            "reserve_bytes": 10 * 1024**3,
        },
        "output": str(logical_output),
    }
    atomic_write_json(spec, logical_output / "spec.json")
    return spec


def validate(spec):
    from look.studies.modern_embracenet import SCHEMA as MODERN_SCHEMA, validate as validate_modern
    if spec.get("schema") == MODERN_SCHEMA:
        return validate_modern(spec)
    from look.runtime.device_budget import validate as validate_gpu_policy
    validate_gpu_policy(spec)
    if spec.get("schema") != SCHEMA or spec.get("task_id") != TASK_ID or spec.get("test_access") is not False:
        raise ValueError("Undeclared EmbraceNet single-seed study")
    if spec.get("seed") != 3416 or spec.get("architecture") != "resnet18" or spec.get("embracement_size") != 256:
        raise ValueError("EmbraceNet single-seed model identity changed")
    if spec.get("packet_sha256") != PACKET_SHA or spec.get("authorization_sha256") != AUTHORIZATION_SHA:
        raise ValueError("Authorization identity changed")
    if spec.get("trial_decision_sha256") != DECISION_SHA or spec.get("phd_research_sha256") != PHD_RESEARCH_SHA:
        raise ValueError("Approved recipe/standard identity changed")
    look = spec.get("look", {})
    if look != {
        "arms": list(ARMS), "patterns": list(PATTERNS), "search": "positive_forward_tree",
        "factor": 16, "rank": 32, "penalty_policy": "prefix_train_pca_gcv",
        "sites": [
            "joint_input", "joint_stem", "joint_stage1", "joint_stage2", "joint_stage3",
            "joint_stage4", "joint_features", "joint_participant_feature", "embraced_feature",
        ],
    }:
        raise ValueError("LOOK tree contract changed")
    storage = spec.get("storage", {})
    logical_output = Path(spec["output"])
    physical_output = Path(storage.get("physical_output", ""))
    backup_root = Path("/backup/mengh/LOOK").resolve()
    if (
        storage.get("policy") != "backup_physical_with_data_logical_symlink_v1"
        or storage.get("logical_output") != str(logical_output)
        or storage.get("physical_root") != str(backup_root)
        or storage.get("reserve_bytes") != spec.get("disk_reserve_bytes")
        or not logical_output.is_symlink()
        or logical_output.resolve() != physical_output.resolve()
        or not physical_output.resolve().is_relative_to(backup_root)
    ):
        raise ValueError("EmbraceNet backup/logical-link storage policy changed")
    from look.training.observed_host import validate_config
    validate_config(spec["training"])
    if file_sha256(Path(spec["data_root"]) / "accepted.json") != spec["data_audit_sha256"]:
        raise ValueError("Data identity changed")
    if file_sha256(spec["initialization"]["path"]) != spec["initialization"]["sha256"]:
        raise ValueError("ImageNet initialization changed")
    for row in spec["source_pins"]:
        if file_sha256(row["path"]) != row["sha256"]:
            raise ValueError("Pinned source or installed dependency changed")
    if file_sha256(Path(spec["source_root"]) / "workspace/PHD_RESEARCH_STANDARD.md") != PHD_RESEARCH_SHA:
        raise ValueError("PHD science standard changed")
    commit = subprocess.check_output(["git", "-C", spec["source_root"], "rev-parse", "HEAD"], text=True).strip()
    if commit != spec["source_commit"]:
        raise ValueError("Deployed LOOK source commit changed")


def make_graph(spec, device):
    from look.studies.modern_embracenet import SCHEMA as MODERN_SCHEMA, make_graph as make_modern
    if spec.get("schema") == MODERN_SCHEMA:
        return make_modern(spec, device)
    from mhd_framework.models import create_model
    torch.hub.set_dir(str(Path(spec["initialization"]["path"]).parent.parent))
    config = dict(name="resnet18", spatial_dims=2, in_channels=3, num_classes=2, views=1, granularity="block")
    cfp = ObservedParticipantModel(create_model(config, weights="IMAGENET1K_V1"))
    oct_parent = ObservedParticipantModel(create_model(config, weights="IMAGENET1K_V1"))
    graph = build_embracenet_host(
        cfp, oct_parent, device=device, embracement_size=spec["embracement_size"],
        sampling_seed=spec["seed"], classifier_dropout=0.0,
    )
    graph.to(device)
    return graph


def make_dataset(spec, role, *, augment=False):
    from look.studies.modern_embracenet import SCHEMA as MODERN_SCHEMA, dataset
    if spec.get("schema") == MODERN_SCHEMA:
        return dataset(spec, role, augment)
    return ArrayPair(spec["data_root"], role, augment=augment, seed=spec["seed"])


def _loader(dataset, spec, check=None):
    loader = DataLoader(
        dataset, batch_size=spec["training"]["microbatch"], shuffle=False,
        num_workers=0, collate_fn=collate_observed,
        generator=torch.Generator().manual_seed(spec["seed"]),
    )
    return CheckedLoader(loader, check) if check is not None else loader


def _freeze(graph):
    graph.eval()
    for parameter in graph.parameters():
        parameter.requires_grad_(False)
    return graph


def _load_selected(spec, root, device):
    receipt = read(root / "host/accepted.json")
    if receipt.get("state") != "accepted" or receipt.get("identity") != stable_hash(spec):
        raise ValueError("Frozen EmbraceNet A is not accepted")
    for name, digest in receipt["files"].items():
        if file_sha256(root / "host" / name) != digest:
            raise ValueError("Frozen EmbraceNet A evidence changed")
    graph = make_graph(spec, device)
    state = read_selected(root / "host/best.pt", identity=stable_hash(spec),
        node_ids=[(n.id,n.name) for n in sorted(graph.nodes,key=lambda n:n.id)])
    ids = [(node.id, node.name) for node in sorted(graph.nodes, key=lambda node: node.id)]
    if state["identity"] != stable_hash(spec) or state["node_ids"] != ids:
        raise ValueError("Frozen EmbraceNet best checkpoint identity changed")
    graph.load_state_dict(state["model"], strict=True)
    from look.models.embracenet import restore_embracenet_sampling
    restore_embracenet_sampling(graph, state["sampling_state"])
    return _freeze(graph), receipt


def _tree_equal(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
            raise ValueError("Resume state differs at " + path)
        return
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or not np.array_equal(a, b):
            raise ValueError("Resume array differs at " + path)
        return
    if isinstance(a, dict):
        if set(a) != set(b):
            raise ValueError("Resume dict keys differ at " + path)
        for key in a:
            if path.endswith(".progress") and key == "seconds":
                continue
            _tree_equal(a[key], b[key], path + "." + str(key))
        return
    if isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError("Resume sequence differs at " + path)
        for index, (x, y) in enumerate(zip(a, b)):
            _tree_equal(x, y, path + f"[{index}]")
        return
    if a != b:
        raise ValueError("Resume scalar differs at " + path)


class ProbeSubset(Dataset):
    def __init__(self, dataset, size):
        self.dataset = dataset
        self.indices = list(range(min(size, len(dataset))))
        self.split = "train"; self.augment = False
        self.participant_ids = [dataset.participant_ids[index] for index in self.indices]
        self.counts = [dataset.counts[index] for index in self.indices]
    def __len__(self): return len(self.indices)
    def __getitem__(self, index): return self.dataset[self.indices[index]]


def _resource_guard(spec, root, stop_requested=lambda: False):
    import psutil
    memory = psutil.virtual_memory()
    if memory.available < spec["host_free_fraction_min"] * memory.total:
        raise MemoryError("Host free-memory reserve breached")
    if shutil.disk_usage(Path(root).resolve()).free < spec["disk_reserve_bytes"]:
        raise OSError("Artifact disk reserve breached")
    if spec.get("lease_safety_seconds") is not None:
        end = float(os.environ["MHD_EXECUTION_LEASE_END"])
        if not np.isfinite(end) or end <= 0:
            raise ValueError("Valid allocation end time required")
        if time.time() >= end - spec["lease_safety_seconds"]:
            return True
    return bool(stop_requested())


def _scan_dataset(dataset, check=lambda:False):
    for index in range(len(dataset)):
        if index % 64 == 0 and check(): raise StagePaused()
        dataset[index]
    return {"participants": len(dataset), "verified_files": len(dataset.verified)}


def _profile_checkpoint(path):
    state=torch.load(Path(path)/"last.pt",map_location="cpu",weights_only=False)
    return state


def _profile_fresh_graph(spec,device,model_state,sampling_state,rng_state):
    graph=make_graph(spec,device)
    graph.load_state_dict(model_state,strict=True)
    restore_embracenet_sampling(graph,sampling_state)
    restore_rng(rng_state)
    return graph


def stage_profile(spec, root, check):
    target = root / "profile"
    if (target / "accepted.json").exists():
        receipt = read(target / "accepted.json")
        if receipt.get("identity") != stable_hash(spec):
            raise ValueError("Profile identity changed")
        return receipt
    import psutil
    target.mkdir(parents=True,exist_ok=True)
    train = make_dataset(spec, "train", augment=True)
    fit = make_dataset(spec, "train")
    dev = make_dataset(spec, "development")
    scan = {"train": _scan_dataset(fit,check), "development": _scan_dataset(dev,check)}
    torch.cuda.reset_peak_memory_stats()
    from look.runtime.device_budget import configure
    configure(spec)
    identity = stable_hash(spec)

    template=make_graph(spec,torch.device("cuda:0"))
    initial_model=cpu_tree(template.state_dict())
    initial_sampling=capture_embracenet_sampling(template)
    initial_rng=capture_rng()
    del template;torch.cuda.empty_cache()

    def fresh_graph():
        return _profile_fresh_graph(
            spec,torch.device("cuda:0"),initial_model,initial_sampling,initial_rng
        )

    def run_target(name,target_updates,*,require_uninterrupted=False):
        folder=target/name
        # An interrupted "continuous" lane cannot certify uninterrupted equivalence.
        if require_uninterrupted and folder.exists() and not (folder/"target_complete.json").exists():
            incident=target/"incidents"/f"{name}_interrupted_{time.time_ns()}"
            incident.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(folder),str(incident))
        if (folder/"last.pt").exists():
            graph=make_graph(spec,torch.device("cuda:0"))
        else:
            graph=fresh_graph()
        result=train_embracenet_host(
            graph,train,dev,spec["training"],spec["seed"],folder,identity,
            torch.device("cuda:0"),check,preflight_target_updates=target_updates,
        )
        if check():
            raise StagePaused()
        if result.get("state")!="paused":
            raise ValueError(f"Profile lane {name} did not stop at a safe preflight boundary")
        checkpoint=_profile_checkpoint(folder)
        if checkpoint["progress"]["updates"]!=target_updates:
            raise ValueError(f"Profile lane {name} reached the wrong optimizer update")
        dev_result=evaluate_complete_analytical(
            graph,_loader(dev,spec),torch.device("cuda:0"),check
        )
        if check():
            raise StagePaused()
        atomic_write_json(
            {"identity":identity,"target_updates":target_updates,"state":"complete","test_access":False},
            folder/"target_complete.json",
        )
        del graph;torch.cuda.empty_cache()
        return checkpoint,dev_result["logits"]

    continuous_state,continuous_logits=run_target("continuous_two_updates",2,require_uninterrupted=True)
    base_folder=target/"resume_base"
    if not (base_folder/"last.pt").exists():
        base_graph=fresh_graph()
    else:
        base_graph=make_graph(spec,torch.device("cuda:0"))
    base_result=train_embracenet_host(
        base_graph,train,dev,spec["training"],spec["seed"],base_folder,identity,
        torch.device("cuda:0"),check,preflight_target_updates=1,
    )
    if check(): raise StagePaused()
    if base_result.get("state")!="paused" or _profile_checkpoint(base_folder)["progress"]["updates"]!=1:
        raise ValueError("Interrupted profile boundary must be exactly optimizer update 1")
    del base_graph;torch.cuda.empty_cache()

    resumed_states=[];resumed_logits=[]
    for name in ("resume_a","resume_b"):
        branch=target/name
        if not branch.exists():
            shutil.copytree(base_folder,branch)
        state,logits=run_target(name,2)
        resumed_states.append(state);resumed_logits.append(logits)

    _tree_equal(resumed_states[0],resumed_states[1])
    _tree_equal(continuous_state,resumed_states[0])
    if not np.array_equal(resumed_logits[0],resumed_logits[1]):
        raise ValueError("Repeated resume development logits differ")
    if not np.array_equal(continuous_logits,resumed_logits[0]):
        raise ValueError("Uninterrupted versus resumed development logits differ")

    peak = int(torch.cuda.max_memory_reserved())
    if peak > spec["gpu_budget_bytes"] or check():
        if check(): raise StagePaused()
        raise MemoryError("Profile resource budget failed")
    profile_bytes=sum(q.stat().st_size for q in target.rglob("*") if q.is_file())
    disk_free=int(shutil.disk_usage(Path(root).resolve()).free)
    receipt = {
        "schema": "look_embracenet_resource_profile_v2", "state": "accepted",
        "identity": identity, "gpu": torch.cuda.get_device_name(0),
        "gpu_peak_reserved_bytes": peak, "gpu_budget_bytes": spec["gpu_budget_bytes"],
        "gpu_reserve_bytes": spec["gpu_reserve_bytes"],
        "host_available_bytes": int(psutil.virtual_memory().available),
        "host_total_bytes": int(psutil.virtual_memory().total),
        "data_scan": scan,
        "repeated_resume_next_update_exact": True,
        "uninterrupted_vs_resumed_next_update_exact": True,
        "resume_development_logits_exact": True,
        "global_optimizer_scheduler_private_rng_mask_cursor_exact": True,
        "profile_bytes":profile_bytes,"disk_free_bytes":disk_free,
        "storage_policy":spec["storage"],"test_access": False,
    }
    atomic_write_json(receipt, target / "accepted.json")
    return receipt

def stage_host(spec, root, check):
    if spec.get("schema")=="look_modern_embracenet_20260926_v1":
        recovery=read(root/"modern_resume/accepted.json")
        if recovery.get("state")!="accepted" or recovery.get("identity")!=stable_hash(spec):
            raise ValueError("Modern host requires fresh-process recovery qualification")
    target = root / "host"
    if (target / "accepted.json").exists():
        return read(target / "accepted.json")
    train = make_dataset(spec, "train", augment=True)
    dev = make_dataset(spec, "development")
    graph = make_graph(spec, torch.device("cuda:0"))
    receipt = train_embracenet_host(
        graph, train, dev, spec["training"], spec["seed"], target, stable_hash(spec),
        torch.device("cuda:0"), check,
    )
    if receipt.get("state") == "paused":
        raise StagePaused()
    if receipt.get("state") != "accepted":
        raise RuntimeError("Formal EmbraceNet A training did not reach accepted plateau")
    return receipt


def _pca_bank(spec, root, graph, fit, check):
    return prepare_embracenet_pca_bank(
        graph, _loader(fit, spec, check), correction_sites(graph),
        spec["look"]["factor"], spec["look"]["rank"], torch.device("cuda:0"),
        root / "pca", {"case": stable_hash(spec), "host_best_sha256": file_sha256(root / "host/best.pt")},
        check,
    )


def stage_pca(spec, root, check):
    target = root / "pca"
    if (target / "accepted.json").exists():
        return read(target / "accepted.json")
    graph, host = _load_selected(spec, root, torch.device("cuda:0"))
    fit = make_dataset(spec, "train")
    bank = _pca_bank(spec, root, graph, fit, check)
    entries = []
    for (node, factor), basis in bank.items():
        bank_id = basis.source_id.split("/", 1)[0]
        path = target / bank_id / f"{correction_sites(graph).index(node):02d}_{node}_x{factor}.pt"
        entries.append({"node": node, "factor": factor, "path": str(path), "sha256": file_sha256(path), "source_id": basis.source_id})
    receipt = {
        "schema": "look_embracenet_pca_bank_v1", "state": "accepted",
        "identity": stable_hash(spec), "host_best_sha256": host["files"]["best.pt"],
        "complete_reference": "author_stochastic_analytic_second_moment_at_embraced_feature",
        "entries": entries, "test_access": False,
    }
    atomic_write_json(receipt, target / "accepted.json")
    return receipt


def _moments_equal(first,second):
    _tree_equal(vars(first),vars(second))


def stage_fit_profile(spec, root, check):
    target = root / "fit_profile"
    if (target / "accepted.json").exists():
        return read(target / "accepted.json")
    target.mkdir(parents=True,exist_ok=True)
    graph, _ = _load_selected(spec, root, torch.device("cuda:0"))
    fit = make_dataset(spec, "train")
    dev = make_dataset(spec, "development")
    bank = _pca_bank(spec, root, graph, fit, check)
    sites = correction_sites(graph)
    bases = {site: next(basis for (name, _), basis in bank.items() if name == site) for site in sites}
    max_site=max(sites,key=lambda site:bases[site].std.numel())
    max_dimension=int(bases[max_site].std.numel())
    rank=spec["look"]["rank"]
    if max_dimension<rank:
        raise ValueError("Representative maximal fitting site cannot support rank32")
    subset = ProbeSubset(fit, 64)
    raw_loader=_loader(subset,spec)

    class InterruptOnce:
        def __init__(self):
            self.calls=0;self.fired=False
        def __call__(self):
            if check(): return True
            self.calls+=1
            if not self.fired and self.calls>=2:
                self.fired=True
                return True
            return False

    identity={"case":stable_hash(spec),"purpose":"resource_profile","site":max_site,"rank":rank}
    torch.cuda.reset_peak_memory_stats()
    interrupted=InterruptOnce()
    partial=EmbraceNetFamilyStatistics(
        graph,raw_loader,{max_site:bases[max_site]},torch.device("cuda:0"),
        target/"resume_moments",identity,spec["workspace_bytes"],rank,
        interrupted,projected_ranks=(rank,),
    )
    try:
        partial.statistics("oct_missing",(),[max_site])
        raise ValueError("Fitting interruption probe did not pause")
    except SelectionPaused:
        if check(): raise StagePaused()
    partial_files=list((target/"resume_moments").glob("*.pt"))
    if len(partial_files)!=1:
        raise ValueError("Fitting interruption did not persist exactly one resumable statistics file")
    partial_payload=torch.load(partial_files[0],map_location="cpu",weights_only=False)["payload"]
    if partial_payload.get("complete") is not False or not partial_payload.get("batch_stamps"):
        raise ValueError("Fitting interruption did not persist a nonempty incomplete cursor")

    resumed=EmbraceNetFamilyStatistics(
        graph,raw_loader,{max_site:bases[max_site]},torch.device("cuda:0"),
        target/"resume_moments",identity,spec["workspace_bytes"],rank,
        check,projected_ranks=(rank,),
    )
    resumed_stats=resumed.statistics("oct_missing",(),[max_site])
    uninterrupted=EmbraceNetFamilyStatistics(
        graph,raw_loader,{max_site:bases[max_site]},torch.device("cuda:0"),
        target/"continuous_moments",identity,spec["workspace_bytes"],rank,
        check,projected_ranks=(rank,),
    )
    continuous_stats=uninterrupted.statistics("oct_missing",(),[max_site])
    _moments_equal(resumed_stats[max_site],continuous_stats[max_site])
    _moments_equal(resumed.projected[max_site][rank],uninterrupted.projected[max_site][rank])

    basis=bases[max_site]
    stats=resumed_stats[max_site]
    q_basis=basis.components[:rank].to(dtype=torch.float64,device="cpu")
    projected=resumed.projected[max_site][rank]
    ridge=_gcv_lambda(
        q_basis@stats.cxx@q_basis.T,q_basis@stats.cxy@q_basis.T,
        float(projected.syy),stats.count,rank,
    )
    dev_loader=_loader(dev,spec)
    baseline=evaluate_single_missing(
        graph,dev_loader,torch.device("cuda:0"),"oct_missing",should_pause=check
    )
    solver_rows=[]
    for arm in ARMS:
        mapping=fit_map(stats,rank,ridge,arm=arm,basis=basis.components[:rank])
        template=LOOKArtifact(
            max_site,"oct_missing","availability_zero_mask",basis.factor,rank,
            basis.feature_shape,basis.downsample_shape,basis.mean,basis.std,basis.pca_mean,
            basis.components[:rank],torch.zeros(rank,rank),torch.zeros(rank),ridge,0.0,0.0,
            float(basis.explained_variance_ratio[:rank].sum()),basis.fit_seconds,
            basis.peak_rss_bytes,basis.source_id,basis.member_names,basis.member_shapes,
            basis.protocol,basis.split_rule,basis.spatial_method,
        )
        artifact=FamilyArtifact(template,mapping)
        artifact_path=target/f"{arm}_{max_site}_rank{rank}.pt"
        atomic_save(artifact_path,artifact.record())
        loaded=FamilyArtifact.from_record(torch.load(artifact_path,map_location="cpu",weights_only=False))
        corrected=evaluate_single_missing(
            graph,dev_loader,torch.device("cuda:0"),"oct_missing",[artifact],should_pause=check
        )
        replay=evaluate_single_missing(
            graph,dev_loader,torch.device("cuda:0"),"oct_missing",[loaded],should_pause=check
        )
        if not np.array_equal(corrected["logits"],replay["logits"]):
            raise ValueError("Fit-profile artifact save/load replay changed")
        delta=float(np.max(np.abs(corrected["logits"]-baseline["logits"])))
        if not math.isfinite(delta) or delta<=0:
            raise ValueError("Fit-profile solver did not produce an effective nonzero full-dev writeback")
        solver_rows.append({
            "arm":arm,"site":max_site,"dimension":max_dimension,"rank":rank,
            "ridge_lambda":float(ridge),"artifact":str(artifact_path),
            "artifact_sha256":file_sha256(artifact_path),
            "full_dev_max_abs_logit_change":delta,
            "full_dev_metrics":corrected["metrics"],"save_load_replay_exact":True,
        })

    peak = int(torch.cuda.max_memory_reserved())
    if peak > spec["gpu_budget_bytes"]:
        raise MemoryError("LOOK fit profile GPU budget breached")
    if check(): raise StagePaused()
    profile_bytes=sum(q.stat().st_size for q in target.rglob("*") if q.is_file())
    receipt = {
        "schema": "look_embracenet_fit_profile_v2", "state": "accepted",
        "identity": stable_hash(spec), "probe_train_participants": len(subset),
        "all_nine_sites": sites, "representative_max_site":max_site,
        "representative_dimension":max_dimension,"rank":rank,
        "gpu_peak_reserved_bytes": peak,
        "fit_resume_vs_uninterrupted_exact":True,
        "projected_resume_vs_uninterrupted_exact":True,
        "solver_rows":solver_rows,
        "full_development_baseline_metrics":baseline["metrics"],
        "profile_bytes":profile_bytes,
        "disk_free_bytes":int(shutil.disk_usage(Path(root).resolve()).free),
        "storage_policy":spec["storage"],"test_access": False,
    }
    atomic_write_json(receipt, target / "accepted.json")
    return receipt

def stage_arm(spec, root, arm, check):
    target = root / arm
    if (target / "accepted.json").exists():
        return read(target / "accepted.json")
    graph, host = _load_selected(spec, root, torch.device("cuda:0"))
    fit = make_dataset(spec, "train")
    dev = make_dataset(spec, "development")
    train_loader = _loader(fit, spec, check)
    dev_loader = _loader(dev, spec, check)
    pca = _pca_bank(spec, root, graph, fit, check)
    frozen = cpu_tree(graph.state_dict())
    records = []; trees = {}
    for pattern in PATTERNS:
        correction = target / "corrections" / pattern
        bank, selection = fit_embracenet_family_trajectory(
            graph, train_loader, dev_loader, arm=arm, pattern=pattern,
            sites=correction_sites(graph), factor=spec["look"]["factor"],
            candidates=[{"rank": spec["look"]["rank"], "ridge_lambda": None}],
            pca_bank=pca, identity=stable_hash(spec), output=correction,
            device=torch.device("cuda:0"), workspace_bytes=spec["workspace_bytes"],
            should_pause=check,
            penalty_policy=spec["look"]["penalty_policy"],
        )
        trees[pattern] = {
            "selection_sha256": file_sha256(correction / "selection.json"),
            "bank_sha256": file_sha256(correction / "bank.pt"),
            "selected_path": selection["selected_path"],
            "site_attempts": selection["site_attempts"],
            "candidate_evaluations": selection["candidate_evaluations"],
            "prefix_count": selection["prefix_count"],
        }
        result = evaluate_single_missing(graph, dev_loader, torch.device("cuda:0"), pattern, bank)
        replay = evaluate_single_missing(graph, dev_loader, torch.device("cuda:0"), pattern, load_embracenet_bank(correction))
        if not np.array_equal(result["logits"], replay["logits"]):
            raise ValueError("Formal EmbraceNet LOOK tree replay changed")
        baseline = evaluate_single_missing(graph, dev_loader, torch.device("cuda:0"), pattern)
        for method, values in ((arm, result), ("host", baseline)):
            path = target / "development" / f"{method}_{pattern}.npz"
            save_prediction_bundle(values, path)
            records.append({
                "method": method, "scenario": pattern, "path": str(path),
                "sha256": file_sha256(path), "metrics": values["metrics"],
            })
    from look.training.mechanism_training import state_equal
    state_equal(graph, frozen)
    receipt = {
        "schema": "look_embracenet_formal_arm_v1", "state": "accepted",
        "identity": stable_hash(spec), "arm": arm, "host_best_sha256": host["files"]["best.pt"],
        "records": records, "trees": trees, "frozen_host_exact": True,
        "test_access": False,
    }
    atomic_write_json(receipt, target / "accepted.json")
    return receipt


def _f1_draws(labels, prediction, counts):
    labels = np.asarray(labels); prediction = np.asarray(prediction)
    indicators = np.column_stack([
        (labels == 0) & (prediction == 0), (labels == 0) & (prediction == 1),
        (labels == 1) & (prediction == 0), (labels == 1) & (prediction == 1),
    ]).astype(np.float64)
    confusion = counts @ indicators
    tn, fp, fn, tp = confusion.T
    f0 = np.divide(2*tn, 2*tn+fn+fp, out=np.zeros_like(tn), where=(2*tn+fn+fp)>0)
    f1 = np.divide(2*tp, 2*tp+fn+fp, out=np.zeros_like(tp), where=(2*tp+fn+fp)>0)
    return (f0 + f1) / 2


def _auc_draws(labels, scores, counts, chunk=256):
    labels = np.asarray(labels)
    pos = np.where(labels == 1)[0]; neg = np.where(labels == 0)[0]
    kernel = ((scores[pos, None] > scores[neg][None, :]).astype(np.float64)
              + 0.5*(scores[pos, None] == scores[neg][None, :]))
    out = np.empty(len(counts), dtype=np.float64)
    for start in range(0, len(counts), chunk):
        c = counts[start:start+chunk].astype(np.float64, copy=False)
        cp, cn = c[:, pos], c[:, neg]
        numerator = np.einsum("bi,ij,bj->b", cp, kernel, cn, optimize=True)
        denominator = cp.sum(1) * cn.sum(1)
        out[start:start+len(c)] = np.divide(numerator, denominator, out=np.full(len(c), np.nan), where=denominator>0)
    return out


def _model_metric_draws(labels, logits_by_key, counts):
    result = {metric: {} for metric in REGISTERED_METRICS}
    one_hot = np.eye(2)[labels]
    for key, logits in logits_by_key.items():
        logits = np.asarray(logits, dtype=np.float64)
        probs = probabilities_from_logits(logits)
        result["macro_f1"][key] = _f1_draws(labels, logits.argmax(1), counts)
        result["macro_auroc_ovr"][key] = _auc_draws(labels, logits[:,1]-logits[:,0], counts)
        nll = logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]
        brier = np.square(probs - one_hot).sum(axis=1)
        weights = counts.astype(np.float64, copy=False)
        result["negative_log_likelihood"][key] = np.einsum("bi,i->b", weights, nll, optimize=True) / len(labels)
        result["multiclass_brier"][key] = np.einsum("bi,i->b", weights, brier, optimize=True) / len(labels)
    return result


def _simultaneous_contrasts(labels, logits_by_key, definitions, iterations, seed):
    n = len(labels)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1/n), size=iterations)
    draws = _model_metric_draws(labels, logits_by_key, counts)
    output = {}
    for metric in REGISTERED_METRICS:
        direction = 1.0 if metric in ("macro_f1", "macro_auroc_ovr") else -1.0
        matrix = []
        points = []
        for definition in definitions:
            method, reference = definition["method_key"], definition["reference_key"]
            raw = draws[metric][method] - draws[metric][reference]
            matrix.append(direction * raw)
            m = logit_metrics(labels, logits_by_key[method])[metric]
            r = logit_metrics(labels, logits_by_key[reference])[metric]
            points.append(direction * (m-r))
        matrix = np.stack(matrix, axis=1)
        standard = np.nanstd(matrix, axis=0, ddof=1)
        centered = matrix - np.nanmean(matrix, axis=0)
        valid = standard > 0
        maximum = np.nanmax(np.abs(centered[:,valid] / standard[valid]), axis=1) if valid.any() else np.zeros(iterations)
        critical = float(np.nanquantile(maximum, .95))
        rows=[]; pvalues=[]
        for index, point in enumerate(points):
            values = matrix[:, index]; values=values[np.isfinite(values)]
            ordinary = np.quantile(values, [.025,.975]).tolist()
            simultaneous = [float(point-critical*standard[index]), float(point+critical*standard[index])] if valid[index] else None
            p = min(1.0, 2*min(float(np.mean(values <= 0)), float(np.mean(values >= 0))))
            pvalues.append(p)
            rows.append({
                **definitions[index], "favorable_improvement": float(point),
                "ordinary_95": ordinary, "simultaneous_95": simultaneous,
                "bootstrap_sd": float(standard[index]), "p_value": p,
                "direction": "positive_favors_LOOK",
            })
        adjusted = holm_adjust(pvalues)
        for row, value in zip(rows, adjusted):
            row["holm_p_value"] = float(value)
        output[metric] = {
            "critical": critical, "iterations": iterations, "seed": seed,
            "family": "four_preregistered_A_plus_LOOK_minus_same_A_contrasts",
            "rows": rows,
        }
    return output


def stage_report(spec, root):
    delivery = root / "delivery"
    if (delivery / "accepted.json").exists():
        return read(delivery / "accepted.json")
    host = read(root / "host/accepted.json")
    arm_receipts = {arm: read(root / arm / "accepted.json") for arm in ARMS}
    if any(value["host_best_sha256"] != host["files"]["best.pt"] for value in arm_receipts.values()):
        raise ValueError("Formal LOOK arms do not share the frozen A")
    arrays = {}; rows=[]; costs={}
    for arm, receipt in arm_receipts.items():
        costs[arm] = receipt["trees"]
        for record in receipt["records"]:
            path=Path(record["path"])
            if file_sha256(path) != record["sha256"]:
                raise ValueError("Formal prediction evidence changed")
            key=(record["method"], record["scenario"])
            with np.load(path,allow_pickle=False) as loaded:
                array={name:loaded[name].copy() for name in loaded.files}
            if key in arrays:
                if any(not np.array_equal(arrays[key][name], array[name]) for name in ("participant_ids","labels","logits")):
                    raise ValueError("Shared frozen-A baseline differs across LOOK arms")
            else:
                arrays[key]=array; rows.append({
                    "method":record["method"],"scenario":record["scenario"],"metrics":record["metrics"],
                    "path":str(path),"sha256":record["sha256"],
                })
    reference=next(iter(arrays.values()))
    for array in arrays.values():
        if not np.array_equal(reference["participant_ids"],array["participant_ids"]) or not np.array_equal(reference["labels"],array["labels"]):
            raise ValueError("Unpaired EmbraceNet development predictions")
    definitions=[]
    for pattern in PATTERNS:
        for arm in ARMS:
            definitions.append({
                "arm": arm, "pattern": pattern,
                "method_key": (arm,pattern), "reference_key": ("host",pattern),
            })
    logits_by_key={key:value["logits"].astype(np.float64) for key,value in arrays.items()}
    stats=_simultaneous_contrasts(
        reference["labels"].astype(np.int64), logits_by_key, definitions,
        spec["bootstrap"]["iterations"], spec["bootstrap"]["seed"],
    )
    delivery.mkdir(parents=True,exist_ok=True)
    atomic_write_json({
        "schema":"look_embracenet_single_seed_results_v1","state":"self_checked_pending_independent_review",
        "identity":stable_hash(spec),"test_access":False,"host_complete_metrics":host["metrics"],
        "development_results":rows,"paired_statistics":stats,"tree_costs":costs,
        "selection_bias":f"same {len(reference['labels'])}-person development set selects tree/checkpoint and estimates effects; intervals are conditional/exploratory",
        "uncertainty_limit":"single seed; no train-seed variation represented",
    },delivery/"results.json")
    lines=[
        "# EmbraceNet + LOOK 单种子累计候选报告","",
        f"状态：self_checked_pending_independent_review。{spec.get('cohort', {}).get('train', 1264)} train / {len(reference['labels'])} development；test封存。",
        "问题：同一个冻结EmbraceNet A在整模态缺失时，加LOOK是否改善？负结果和分支损伤均保留。","",
        f"冻结A：seed3416；best epoch {host['best_epoch']}；complete-dev解析平均logits Macro-F1 {100*host['metrics']['macro_f1']:.3f}%。",
        "训练缺失：complete / missing OCT / missing CFP 各1/3；双眼同步缺失。完整态分类选优用解析E[logits]；single-missing每状态一次作者forward。",
        "LOOK：9个实际节点；positive-forward tree；rank32；空间节点x16；prefix train-PCA GCV；PCA完整参考在embraced_feature保留作者随机条件方差。","",
        "|方法|状态|Macro-F1|AUROC|NLL|Brier|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        m=row["metrics"]
        lines.append(f"|{row['method']}|{row['scenario']}|{100*m['macro_f1']:.3f}%|{100*m['macro_auroc_ovr']:.3f}%|{m['negative_log_likelihood']:.5f}|{m['multiclass_brier']:.5f}|")
    lines += ["","四个预注册主对比均为同一冻结A下A+LOOK对A；下表“改善”统一为正值有利于LOOK（NLL/Brier已反向）。"]
    for metric in REGISTERED_METRICS:
        lines += ["",f"### {metric}"]
        for row in stats[metric]["rows"]:
            lo,hi=row["ordinary_95"];sim=row["simultaneous_95"]
            sim_text="未定义（零方差恒等对比）" if sim is None else f"[{sim[0]:.5f}, {sim[1]:.5f}]"
            lines.append(f"- {row['arm']} / {row['pattern']}: 改善 {row['favorable_improvement']:.5f}; ordinary95 [{lo:.5f}, {hi:.5f}]; 同族同时95 {sim_text}; Holm p={row['holm_p_value']:.4g}.")
    lines += ["","开发集同时承担checkpoint/tree选择与效果估计，存在选择偏差；配对bootstrap不能消除该偏差。当前只有一个训练seed，不代表训练随机性稳定性。",
              "本包不读test，不因阴性结果补调参/补seed；是否做后续重复由独立审查后的同范围科研决策另行留档。"]
    (delivery/"README.zh-CN.md").write_text("\n".join(lines)+"\n")
    receipt={
        "schema":"look_embracenet_delivery_v1","state":"self_checked_pending_independent_review",
        "identity":stable_hash(spec),"test_access":False,"scientific_acceptance":False,
        "files":{"results.json":file_sha256(delivery/"results.json"),"README.zh-CN.md":file_sha256(delivery/"README.zh-CN.md")},
        "host_best_sha256":host["files"]["best.pt"],
    }
    atomic_write_json(receipt,delivery/"accepted.json")
    return receipt


def verify_case(run, spec):
    root=Path(run);validate(spec);identity=stable_hash(spec)
    for path in (root/"profile/accepted.json",root/"host/accepted.json",root/"pca/accepted.json",root/"fit_profile/accepted.json",
                 root/"pca_free_mean/accepted.json",root/"residual_rrr/accepted.json",root/"delivery/accepted.json"):
        record=read(path)
        if record.get("identity")!=identity and path.name=="accepted.json":
            raise ValueError("EmbraceNet delivery receipt identity changed")
    if spec.get("schema")=="look_modern_embracenet_20260926_v1":
        recovery=read(root/"modern_resume/accepted.json")
        if recovery.get("state")!="accepted" or recovery.get("identity")!=identity or recovery.get("test_access") is not False:
            raise ValueError("Modern fresh-process acceptance missing or changed")
        for relative,key in (("modern_resume/run/last.pt","checkpoint_sha256"),("profile/continuous_two_updates/last.pt","reference_sha256")):
            if file_sha256(root/relative)!=recovery.get(key):
                raise ValueError("Modern recovery checkpoint changed")
    delivery=read(root/"delivery/accepted.json")
    if delivery.get("state")!="self_checked_pending_independent_review" or delivery.get("test_access") is not False:
        raise ValueError("EmbraceNet delivery is not a review candidate")
    for name,digest in delivery["files"].items():
        if file_sha256(root/"delivery"/name)!=digest:
            raise ValueError("EmbraceNet delivery evidence changed")
    return delivery


def _stage_receipt(root, stage):
    paths={
        "profile":root/"profile/accepted.json","host":root/"host/accepted.json","pca":root/"pca/accepted.json",
        "fit_profile":root/"fit_profile/accepted.json","pca_free_mean":root/"pca_free_mean/accepted.json",
        "residual_rrr":root/"residual_rrr/accepted.json","report":root/"delivery/accepted.json", "modern_resume":root/"modern_resume/accepted.json",
    }
    return paths[stage]


def pipeline(spec_path):
    spec=read(spec_path);validate(spec);root=Path(spec["output"]);root.mkdir(parents=True,exist_ok=True)
    with (root/"pipeline.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_write_json({"state":"running","identity":stable_hash(spec),"time":time.time(),"test_access":False},root/"status.json")
        try:
            for stage in (("profile","modern_resume","host","pca","fit_profile","pca_free_mean","residual_rrr")
                          if spec.get("schema")=="look_modern_embracenet_20260926_v1"
                          else ("profile","host","pca","fit_profile","pca_free_mean","residual_rrr")):
                receipt_path=_stage_receipt(root,stage)
                if receipt_path.exists():
                    record=read(receipt_path)
                    if record.get("identity")!=stable_hash(spec) or record.get("state")!="accepted":
                        raise ValueError("Existing stage receipt changed: "+stage)
                    continue
                env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=":4096:8")
                log=(root/f"{stage}.log").open("a")
                process=subprocess.run(
                    [sys.executable,"-m","look.studies.embracenet_delivery","--spec",str(spec_path),"--stage",stage],
                    env=env,stdout=log,stderr=subprocess.STDOUT,
                )
                log.close()
                if process.returncode == 75:
                    atomic_write_json({
                        "state":"paused","identity":stable_hash(spec),"stage":stage,
                        "time":time.time(),"test_access":False,
                    },root/"status.json")
                    return
                if process.returncode:
                    raise RuntimeError(f"EmbraceNet stage {stage} failed ({process.returncode})")
            stage_report(spec,root)
            verify_case(root,spec)
            atomic_write_json({"state":"completed","identity":stable_hash(spec),"time":time.time(),"test_access":False},root/"status.json")
        except Exception as error:
            import traceback
            atomic_write_json({"state":"needs_review","identity":stable_hash(spec),"error":repr(error),"traceback":traceback.format_exc(),"time":time.time(),"test_access":False},root/"status.json")
            raise


def _visible_gpu_uuid(props):
    # CUDA ordinal0 is local to the Slurm visibility mask, not physical GPU0.
    value=getattr(props,"uuid",None)
    if not value:
        raise ValueError("Actual visible CUDA device UUID required")
    from mhd_models.scheduling.standalone import nvml_uuid
    value=nvml_uuid(value)
    if os.environ.get("SLURM_JOB_ID") and torch.cuda.device_count()!=1:
        raise ValueError("This method requires exactly one allocated visible GPU")
    return str(value)


def work(spec, stage):
    validate(spec);root=Path(spec["output"]);root.mkdir(parents=True,exist_ok=True)
    stop=False
    def request(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request);signal.signal(signal.SIGUSR1,request)
    def check():
        return _resource_guard(spec,root,lambda:stop)

    props=torch.cuda.get_device_properties(0)
    locks=Path(spec["lock_root"]);locks.mkdir(parents=True,exist_ok=True)
    uuid=_visible_gpu_uuid(props)
    try:
        with (locks/(uuid+".lock")).open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from look.runtime.device_budget import configure
            configure(spec)
            torch.set_num_threads(2);torch.manual_seed(spec["seed"]);np.random.seed(spec["seed"]);random.seed(spec["seed"])
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            if check(): raise StagePaused()
            if stage=="profile": stage_profile(spec,root,check)
            elif stage=="modern_resume":
                from look.studies.modern_embracenet import SCHEMA as MODERN_SCHEMA, fresh_process_resume
                if spec.get("schema")!=MODERN_SCHEMA: raise ValueError("Undeclared modern recovery stage")
                fresh_process_resume(spec,root,check)
            elif stage=="host": stage_host(spec,root,check)
            elif stage=="pca": stage_pca(spec,root,check)
            elif stage=="fit_profile": stage_fit_profile(spec,root,check)
            elif stage in ARMS: stage_arm(spec,root,stage,check)
            else: raise ValueError("Unknown GPU stage")
            if check(): raise StagePaused()
            atomic_write_json({
                "stage":stage,"state":"completed","identity":stable_hash(spec),
                "gpu":props.name,"time":time.time(),"test_access":False,
            },root/(stage+"_status.json"))
            return "completed"
    except (StagePaused,SelectionPaused,LoaderPaused,EvaluationPaused):
        atomic_write_json({
            "stage":stage,"state":"paused","identity":stable_hash(spec),
            "gpu":props.name,"time":time.time(),"test_access":False,
            "resume":"same_stage_same_identity",
        },root/(stage+"_status.json"))
        return "paused"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--spec");parser.add_argument("--stage",choices=["init","pipeline","profile","modern_resume","host","pca","fit_profile",*ARMS,"report"])
    parser.add_argument("--init-from");parser.add_argument("--source-root");parser.add_argument("--output");parser.add_argument("--physical-output")
    args=parser.parse_args()
    if args.stage=="init":
        if not (args.init_from and args.source_root and args.output and args.physical_output):
            raise ValueError("init requires --init-from --source-root --output --physical-output")
        initialize_spec(args.init_from,args.source_root,args.output,args.physical_output);return
    if not args.spec:raise ValueError("--spec required")
    spec=read(args.spec)
    if args.stage=="pipeline":pipeline(args.spec)
    elif args.stage=="report":stage_report(spec,Path(spec["output"]))
    else:
        if work(spec,args.stage)=="paused":
            raise SystemExit(75)


if __name__=="__main__":
    main()
