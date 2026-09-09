from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(project_root: Path, extra_paths: Iterable[Path] = ()) -> Dict[str, str]:
    paths = [project_root / "framework.lock.json", project_root / "requirements.txt"]
    paths.extend(sorted((project_root / "src/look").rglob("*.py")))
    paths.extend(sorted((project_root / "scripts").glob("*")))
    paths.extend(sorted((project_root / "third_party/MHD_Framework/src/mhd_framework").glob("*.py")))
    paths.extend(sorted((project_root / "configs").glob("requirements*.txt")))
    paths.extend(project_root / name for name in ("README.md", "project.json"))
    paths.append(project_root / "pyproject.toml")
    paths.extend(extra_paths)
    return {
        str(path.resolve().relative_to(project_root.resolve())): sha256(path)
        for path in paths
        if path.is_file() and project_root.resolve() in path.resolve().parents
    }


def implementation_sha256(project_root: Path) -> str:
    paths = sorted((project_root / "src/look").rglob("*.py"))
    paths.extend(sorted((project_root / "third_party/MHD_Framework/src/mhd_framework").glob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def backbone_implementation_sha256(project_root: Path) -> str:
    """Hash only code that can change complete-modality backbone training."""
    relative_paths = (
        "src/look/data/dataset.py",
        "src/look/training/distributed.py",
        "src/look/training/worker.py",
        "src/look/models/graph.py",
        "src/look/evaluation/metrics.py",
        "src/look/runtime/monitoring.py",
        "src/look/training/trainer.py",
    )
    paths = [project_root / relative for relative in relative_paths]
    paths.extend(sorted((project_root / "third_party/MHD_Framework/src/mhd_framework").glob("*.py")))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Backbone implementation files are missing: {missing}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_manifest(
    config: Dict[str, Any], labels_csv: Path, project_root: Path
) -> Dict[str, Any]:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    data_contract_paths = [
        labels_csv,
        labels_csv.parent / "class_mapping.json",
        labels_csv.parent.parent / "cohort_flow.json",
        labels_csv.parent.parent / "cohort_manifest.json",
        labels_csv.parent.parent / "verification.json",
    ]
    return {
        "config": config,
        "labels_sha256": sha256(labels_csv),
        "data_contract_sha256": {
            path.name: sha256(path) for path in data_contract_paths if path.is_file()
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE", "1"),
        "cuda_devices": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        ),
        "git_commit": git_commit,
        "source_sha256": source_hashes(project_root),
    }


def write_json_atomic(payload: Dict[str, Any], path: Path) -> None:
    from look.runtime.state import atomic_write_json
    atomic_write_json(payload, path)
