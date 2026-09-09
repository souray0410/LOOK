#!/usr/bin/env python3
"""Remove only the stopped all-position three-seed Step 38 attempt."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.state import atomic_write_json, file_sha256, utc_now


TARGET_STUDY = "7eb81db4949c"
TARGET_PROTOCOL = "unified_fusion_joint_look_macro_f1_v2"
SESSION = "look-unified-20260904-191807"
PROFILE_TOKEN = "__clf-unified-macro-f1__gan-not_applicable__look-unified_backbone__"
EXPECTED_CASES = {
    *(f"oct_only:{seed}" for seed in (3407, 3408, 3409)),
    *(f"cfp_only:{seed}" for seed in (3407, 3408, 3409)),
    *(f"input:{seed}" for seed in (3407, 3408, 3409)),
    "stem:3407",
}


def _case_from_name(name: str) -> str:
    match = re.search(r"resnet50_(?:oct_cfp_fusion_)?(oct_only|cfp_only|input|stem)__[^_].*__seed(3407|3408|3409)__", name)
    if not match:
        raise RuntimeError(f"Unexpected target artifact name: {name}")
    return f"{match.group(1)}:{match.group(2)}"


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _assert_safe(path: Path, runtime_root: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(runtime_root.resolve())
    if path.is_symlink() or any(item.is_symlink() for item in path.rglob("*")):
        raise RuntimeError(f"Symlink in cleanup target: {path}")
    relative = resolved.relative_to(runtime_root.resolve())
    if any(part.lower() in {"dataset", "test", "tests", "smoke", "pca", "generators"} for part in relative.parts):
        raise RuntimeError(f"Protected path in cleanup target: {path}")


def _assert_stopped(runtime_root: Path) -> None:
    if subprocess.run(["tmux", "has-session", "-t", SESSION], capture_output=True).returncode == 0:
        raise RuntimeError(f"Session {SESSION} is still alive")
    process = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True)
    writers = []
    for line in process.stdout.splitlines():
        if str(runtime_root) in line and any(token in line for token in ("38_run_unified_study.py", "look_core.distributed_worker")):
            writers.append(line.strip())
    if writers:
        raise RuntimeError("Live study writers: " + " | ".join(writers))


def _atomic_registry(registry: Path, experiment_ids: set[str]) -> int:
    payload = json.loads(registry.read_text()) if registry.exists() else {"experiments": []}
    rows = payload.get("experiments")
    if not isinstance(rows, list):
        raise RuntimeError("Unexpected experiment registry schema")
    kept = [row for row in rows if row.get("experiment_id") not in experiment_ids]
    removed = len(rows) - len(kept)
    unexpected = {row.get("experiment_id") for row in rows if PROFILE_TOKEN in str(row.get("experiment_id"))} - experiment_ids
    if unexpected:
        raise RuntimeError(f"Non-target matching registry rows: {sorted(unexpected)}")
    atomic_write_json({**payload, "experiments": kept}, registry)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    paths = resolve_runtime_arguments(args)
    runtime_root = paths.runs_root.parent.resolve()
    _assert_stopped(runtime_root)

    study = paths.runs_root / "unified_study" / TARGET_STUDY
    summary_path = study / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("protocol") != TARGET_PROTOCOL or summary.get("status") != "failed":
        raise RuntimeError("Target study identity/status mismatch")
    if summary.get("test_access") is not False:
        raise RuntimeError("Target study unexpectedly accessed test")

    experiment_dirs = sorted(path for path in (paths.runs_root / "experiments").glob(f"*{PROFILE_TOKEN}*") if path.is_dir())
    backbone_dirs = sorted(path for path in (paths.runs_root / "backbones").iterdir() if path.is_dir())
    if {_case_from_name(path.name) for path in experiment_dirs} != EXPECTED_CASES or len(experiment_dirs) != len(EXPECTED_CASES):
        raise RuntimeError("Experiment cleanup set differs from the ten stopped-study cases")
    if {_case_from_name(path.name) for path in backbone_dirs} != EXPECTED_CASES or len(backbone_dirs) != len(EXPECTED_CASES):
        raise RuntimeError("Backbone cleanup set differs from the ten stopped-study cases")
    experiment_ids = {path.name for path in experiment_dirs}
    backbone_ids = {path.name for path in backbone_dirs}

    sweep_dirs = []
    for plan in sorted((paths.runs_root / "sweeps").glob("validation__*/study_plan.json")):
        ids = set(json.loads(plan.read_text()).get("experiment_ids", []))
        if ids and ids <= experiment_ids:
            sweep_dirs.append(plan.parent)
        elif ids & experiment_ids:
            raise RuntimeError(f"Mixed target/non-target sweep: {plan.parent}")
    if len(sweep_dirs) != len(EXPECTED_CASES):
        raise RuntimeError("Expected exactly ten stopped-study sweeps")

    state_dirs = sorted(path for path in paths.cache_root.joinpath("pipeline_state").glob(f"experiment__*{PROFILE_TOKEN}*__validation") if path.is_dir())
    if len(state_dirs) != len(EXPECTED_CASES):
        raise RuntimeError("Expected exactly ten stopped-study experiment states")
    partials = sorted(path for path in paths.cache_root.joinpath("partial").glob("classifier__*.json") if any(bid in path.name for bid in backbone_ids))
    logs = sorted((paths.runs_root / "logs").glob(f"{SESSION}_*.log"))
    unified_state = paths.cache_root / "pipeline_state" / "unified_study"
    targets = [study, *experiment_dirs, *backbone_dirs, *sweep_dirs, *state_dirs, *partials, *logs]
    if unified_state.exists():
        targets.append(unified_state)
    unique = []
    seen = set()
    for path in targets:
        key = str(path.resolve())
        if key not in seen:
            _assert_safe(path, runtime_root)
            seen.add(key); unique.append(path)

    audit_path = paths.runs_root / "maintenance" / "superseded_unified_study_cleanup.json"
    audit = {
        "status": "planned",
        "created_at_utc": utc_now(),
        "execute": args.execute,
        "reason": "Restore the prespecified seed-3407 fusion screening followed by selected-position replication; the stopped scheduler incorrectly ran all positions at three seeds.",
        "target_study": TARGET_STUDY,
        "target_protocol": TARGET_PROTOCOL,
        "source_identity": {
            "implementation_sha256": summary.get("implementation_sha256"),
            "spec_sha256": summary.get("spec_sha256"),
            "labels_sha256": summary.get("labels_sha256"),
        },
        "aggregate_metrics": {key: {"macro_f1": row["macro_f1"], "best_epoch": row["best_epoch"]} for key, row in summary.get("backbones", {}).items()},
        "targets": [{"path": str(path), "bytes": _size(path)} for path in unique],
        "registry": str(paths.runs_root / "experiment_registry.json"),
        "protected": ["dataset", "test", "tests", "smoke", "pca", "generators"],
    }
    atomic_write_json(audit, audit_path)
    if not args.execute:
        print(json.dumps({"status": "planned", "audit": str(audit_path), "targets": len(unique)}, indent=2))
        return

    registry_removed = _atomic_registry(paths.runs_root / "experiment_registry.json", experiment_ids)
    for path in sorted(unique, key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    remaining = [str(path) for path in unique if path.exists()]
    if remaining:
        raise RuntimeError(f"Cleanup incomplete: {remaining}")
    audit.update(status="complete", completed_at_utc=utc_now(), registry_rows_removed=registry_removed)
    atomic_write_json(audit, audit_path)
    print(json.dumps({"status": "complete", "audit": str(audit_path), "targets_removed": len(unique), "registry_rows_removed": registry_removed}, indent=2))


if __name__ == "__main__":
    main()
