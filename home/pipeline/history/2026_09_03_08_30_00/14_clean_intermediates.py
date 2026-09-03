#!/usr/bin/env python3
"""Step 14: Remove reproducible intermediates after final dataset verification."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from look_core.cli import add_runtime_arguments, add_source_arguments, resolve_runtime_arguments, resolve_source_arguments


ROOT: Path
KEEP = {
    "21015",
    "21016",
    "21017",
    "21018",
    "paired_eye_manifest.csv",
    "phenotypes",
    "cohorts",
}
REMOVABLE = {
    "labels",
    "pairing_reports",
    ".ophthalmology_export.pid",
    ".ophthalmology_verify.pid",
    "ophthalmology_export.log",
    "ophthalmology_export_20260830_160245.csv",
    "ophthalmology_export_20260830_160404.csv",
    "ophthalmology_export_20260830_160533.csv",
    "ophthalmology_verification.log",
    "ophthalmology_verification.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    add_source_arguments(parser)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--execute", action="store_true")
    return resolve_source_arguments(parser.parse_args())


def is_read_only(path: Path) -> bool:
    result = subprocess.run(
        ["findmnt", "-no", "OPTIONS", "--target", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return "ro" in result.stdout.strip().split(",")


def main() -> int:
    global ROOT
    args = parse_args()
    paths = resolve_runtime_arguments(args)
    ROOT = (args.root or paths.dataset_root).resolve()
    verification = paths.cohort_root / "verification.json"
    if not verification.is_file():
        raise SystemExit("Run Step 13 before cleanup")
    result = json.loads(verification.read_text())
    if result.get("status") != "PASS":
        raise SystemExit("Refusing cleanup because verification status is not PASS")
    if not all(is_read_only(path) for path in (args.source_root, args.label_root / "ukb670300.csv")):
        raise SystemExit("Refusing cleanup because a source mount is not read-only")
    actual = {path.name for path in ROOT.iterdir()}
    unexpected = actual - KEEP - REMOVABLE
    if unexpected:
        raise SystemExit(f"Unexpected top-level paths; refusing cleanup: {unexpected}")
    targets = sorted(actual & REMOVABLE)
    cohorts_root = ROOT / "cohorts"
    cohort_names = {path.name for path in cohorts_root.iterdir() if path.is_dir()}
    if cohort_names != {paths.cohort_root.name}:
        raise SystemExit(f"Unexpected cohort directories; refusing cleanup: {cohort_names}")
    print(json.dumps({"execute": args.execute, "remove": targets}, indent=2))
    if not args.execute:
        print("Dry run only. Re-run with --execute after reviewing the allowlist.")
        return 0
    for name in targets:
        path = ROOT / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    remaining = {path.name for path in ROOT.iterdir()}
    if remaining != KEEP:
        raise RuntimeError(f"Unexpected final dataset-root contents: {remaining}")
    remaining_cohorts = {path.name for path in cohorts_root.iterdir() if path.is_dir()}
    if remaining_cohorts != {paths.cohort_root.name}:
        raise RuntimeError(f"Unexpected final cohort directories: {remaining_cohorts}")
    print(json.dumps({"status": "PASS", "remaining": sorted(remaining)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
