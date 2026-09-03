#!/usr/bin/env python3
"""Adopt an existing dataset/cache into the current runtime by same-filesystem rename."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.state import atomic_write_json, file_sha256, utc_now


def tree_stats(root: Path) -> dict[str, int]:
    files = 0
    bytes_total = 0
    for directory, _, names in os.walk(root):
        base = Path(directory)
        for name in names:
            path = base / name
            if path.is_file():
                files += 1
                bytes_total += path.stat().st_size
    return {"files": files, "bytes": bytes_total}


def same_device(source: Path, destination_parent: Path) -> bool:
    return source.stat().st_dev == destination_parent.stat().st_dev


def adopt(source: Path, destination: Path, *, execute: bool) -> dict[str, object]:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        return {"status": "already_current", "path": str(destination)}
    if source.exists() and destination.exists():
        raise FileExistsError(f"Both source and destination exist: {source}, {destination}")
    if not source.exists() and destination.exists():
        return {
            "status": "already_adopted",
            "path": str(destination),
            "stats": tree_stats(destination),
        }
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not same_device(source, destination.parent):
        raise RuntimeError("Adoption requires a same-filesystem rename; copying is refused")
    before = tree_stats(source)
    label = source / "reference_labels.csv"
    label_hash = file_sha256(label) if label.is_file() else None
    if not execute:
        return {
            "status": "dry_run",
            "source": str(source),
            "destination": str(destination),
            "stats": before,
            "reference_labels_sha256": label_hash,
        }
    os.rename(source, destination)
    after = tree_stats(destination)
    if before != after:
        raise RuntimeError(f"Post-rename tree statistics differ: {before} != {after}")
    moved_label = destination / "reference_labels.csv"
    if label_hash is not None and file_sha256(moved_label) != label_hash:
        raise RuntimeError("reference_labels.csv changed during adoption")
    return {
        "status": "adopted",
        "source": str(source),
        "destination": str(destination),
        "stats": after,
        "reference_labels_sha256": label_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-preprocess-cache", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    paths = resolve_runtime_arguments(args)
    plan = {
        "dataset": adopt(
            args.source_dataset, paths.dataset_root, execute=args.execute
        ),
        "preprocess_cache": adopt(
            args.source_preprocess_cache,
            paths.preprocess_cache_root,
            execute=args.execute,
        ),
        "created_at_utc": utc_now(),
    }
    if args.execute:
        destination = paths.cache_root / "pipeline_state" / "data_tree_adoption.json"
        atomic_write_json(plan, destination)
        plan["manifest_path"] = str(destination)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
