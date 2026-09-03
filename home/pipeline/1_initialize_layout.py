#!/usr/bin/env python3
"""Initialize the portable LOOK runtime layout and source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).chmod(0o644)
        Path(temporary_name).replace(destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def source_manifest(project_root: Path, release_id: str) -> Path:
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    files = sorted(
        path
        for path in project_root.rglob("*")
        if path.is_file()
        and path.name not in {"file_manifest.json", ".DS_Store"}
        and not excluded.intersection(path.parts)
        and not any(part.endswith(".egg-info") for part in path.parts)
    )
    destination = project_root / "file_manifest.json"
    atomic_json(
        {
            "schema_version": 2,
            "release_id": release_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": [
                {
                    "path": str(path.relative_to(project_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in files
            ],
        },
        destination,
    )
    return destination


def release_manifest(project_root: Path, data_root: Path, source_manifest_path: Path, release_id: str) -> Path | None:
    bundle_root = project_root.parent
    if project_root.name != "home" or not (bundle_root / "data").is_dir():
        return None
    skeleton = sorted(
        path
        for path in (bundle_root / "data").rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    )
    release_documents = [
        path for path in (
            bundle_root / "README.md",
            bundle_root / "GENERAL_PROJECT_STANDARD.md",
        )
        if path.is_file()
    ]
    destination = bundle_root / "release_manifest.json"
    atomic_json(
        {
            "schema_version": 2,
            "project": "LOOK",
            "release_id": release_id,
            "home_tree": "home",
            "data_tree": "data",
            "deployment_data_root": str(data_root),
            "source_manifest": {
                "path": "home/file_manifest.json",
                "sha256": sha256(source_manifest_path),
            },
            "release_documents": [
                {
                    "path": str(path.relative_to(bundle_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in release_documents
            ],
            "data_skeleton": [
                {
                    "path": str(path.relative_to(bundle_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in skeleton
            ],
        },
        destination,
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    project_root = (args.project_root or Path(__file__).resolve().parents[1]).resolve()
    config = json.loads((project_root / "project.json").read_text(encoding="utf-8"))
    manifest = source_manifest(project_root, config["release_id"])
    if args.source_only:
        print(manifest)
        return 0
    data_root = Path(
        args.data_root
        or os.environ.get("LOOK_DATA_ROOT")
        or config["deployment_defaults"]["data_root"]
    ).expanduser().resolve()
    directories = config["directories"]
    dataset_root = Path(
        args.dataset_root
        or os.environ.get("LOOK_DATASET_ROOT")
        or config["deployment_defaults"].get("dataset_root")
        or data_root / directories["dataset"]
    ).expanduser().resolve()
    cache_root = Path(
        args.cache_root or os.environ.get("LOOK_CACHE_ROOT") or data_root / directories["cache"]
    ).expanduser().resolve()
    runs_root = Path(
        args.runs_root or os.environ.get("LOOK_RUNS_ROOT") or data_root / directories["runs"]
    ).expanduser().resolve()
    runtime = {
        "dataset": dataset_root,
        "pipeline_state": cache_root / "pipeline_state",
        "partial": cache_root / "partial",
        "quarantine": cache_root / "quarantine",
        "backbones": runs_root / "backbones",
        "experiments": runs_root / "experiments",
        "baseline_selection": runs_root / "baseline_selection",
        "generators": runs_root / "generators",
        "freezes": runs_root / "freezes",
        "sweeps": runs_root / "sweeps",
        "smoke": runs_root / "smoke",
        "task_scout": runs_root / "task_scout",
        "logs": runs_root / "logs",
    }
    print(json.dumps({"dry_run": args.dry_run, "runtime": {k: str(v) for k, v in runtime.items()}}, indent=2))
    if not args.dry_run:
        for path in runtime.values():
            path.mkdir(parents=True, exist_ok=True)
    generated_release = release_manifest(project_root, data_root, manifest, config["release_id"])
    if generated_release:
        print(generated_release)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
