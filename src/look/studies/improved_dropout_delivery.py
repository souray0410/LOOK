"""Current V5 execution contract for the completed CFP/OCT IMD study.

Historical V4 artifacts are converted by a separate offline tool. This module
only accepts current V5 checkpoints and a pinned current execution revision.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import torch

from look.data.array_pair import ArrayPair
from look.models.observed_participant import ObservedParticipantModel
from look.models.improved_modality_dropout import build_improved_dropout_host
from look.runtime.device_budget import validate as validate_gpu_budget
from look.runtime.host_checkpoint import read_selected
from look.runtime.state import file_sha256, stable_hash
from look.training.improved_modality_dropout_host import DEFAULTS, validate_config


SCHEMA = "look_improved_modality_dropout_real_admission_v1"
TASK_ID = "ws02-overnight-teacher-20260919-v1"
RUN_ID = "improved_dropout_20260920_v1"
MODEL = dict(name="resnet18", spatial_dims=2, in_channels=3,
             num_classes=2, views=1, granularity="block")


def read(path):
    return json.loads(Path(path).read_text())


def validate(spec):
    from mhd_framework import __api_version__
    from mhd_framework.models.artifacts import runtime_source_sha256

    if __api_version__ != "V5" or spec.get("framework_api") != "V5":
        raise ValueError("Current V5 IMD execution only")
    if (spec.get("schema"), spec.get("task_id"), spec.get("run_id")) != (SCHEMA, TASK_ID, RUN_ID):
        raise ValueError("Unregistered IMD study")
    if spec.get("test_access") is not False or spec.get("seed") != 3416:
        raise ValueError("IMD train/development study identity changed")
    if spec.get("scope") != "real train/development admission/profile/resume before full A":
        raise ValueError("IMD approved scope changed")
    validate_config(spec["training"])
    if spec["training"] != DEFAULTS or spec.get("model_config") != MODEL:
        raise ValueError("IMD model or optimization recipe changed")
    validate_gpu_budget(spec)
    if file_sha256(Path(spec["data_root"]) / "accepted.json") != spec["data_audit_sha256"]:
        raise ValueError("IMD data identity changed")
    init = spec["initialization"]
    if init.get("kind") != "public_imagenet_fresh_host" or file_sha256(init["path"]) != init["sha256"]:
        raise ValueError("IMD public initialization changed")
    if spec["framework_source_sha256"] != runtime_source_sha256():
        raise ValueError("MHD V5 runtime source changed")
    source_root = Path(spec["source_root"])
    commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if commit != spec["source_commit"]:
        raise ValueError("IMD source checkout changed")
    pins = spec["source_pins"]
    if not pins or len({row["path"] for row in pins}) != len(pins):
        raise ValueError("IMD source pins absent or duplicated")
    for row in pins:
        if file_sha256(row["path"]) != row["sha256"]:
            raise ValueError("IMD source or dependency pin changed")
    logical = Path(spec["output"])
    physical = Path(spec["physical_output"])
    physical_root = Path(spec["physical_root"])
    if (not logical.is_symlink() or logical.resolve() != physical.resolve()
            or not physical.resolve().is_relative_to(physical_root.resolve())):
        raise ValueError("IMD candidate storage mapping changed")


def make_graph(spec, device):
    from mhd_framework.models import create_model

    validate(spec)
    torch.hub.set_dir(str(Path(spec["initialization"]["path"]).parent.parent))
    parents = [ObservedParticipantModel(create_model(MODEL, weights="IMAGENET1K_V1")) for _ in range(2)]
    graph = build_improved_dropout_host(*parents, device=str(device),
                                        hidden_dropout=.1, classifier_dropout=.1)
    graph.to(device)
    return graph


def load_selected(spec, root, device):
    root = Path(root)
    receipt = read(root / "host/accepted.json")
    if (receipt.get("state") != "accepted" or receipt.get("identity") != stable_hash(spec)
            or receipt.get("test_access") is not False):
        raise ValueError("Current IMD host receipt not accepted")
    for name, digest in receipt["files"].items():
        if file_sha256(root / "host" / name) != digest:
            raise ValueError("Current IMD host asset changed")
    graph = make_graph(spec, device)
    ids = [(node.id, node.name) for node in sorted(graph.nodes, key=lambda node: node.id)]
    state = read_selected(root / "host/best.pt", identity=stable_hash(spec), node_ids=ids)
    graph.load_state_dict(state["model"], strict=True)
    graph.eval()
    for parameter in graph.parameters():
        parameter.requires_grad_(False)
    return graph, receipt


def datasets(spec):
    validate(spec)
    train = ArrayPair(spec["data_root"], "train", augment=True, seed=spec["seed"])
    development = ArrayPair(spec["data_root"], "development")
    if len(train) != 1264 or len(development) != 296 or set(train.participant_ids) & set(development.participant_ids):
        raise ValueError("IMD train/development cohort changed")
    return train, development
