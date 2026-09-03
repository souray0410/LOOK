#!/usr/bin/env python3
"""Step 10: build the bilateral visit manifest and extract selected UKB fields."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.defaults import DeploymentDefaults
from look_core.phenotype import build_paired_visit_manifest, extract_selected_table
from look_core.reproducibility import write_json_atomic


def require_read_only(path: Path) -> None:
    result = subprocess.run(
        ["findmnt", "-no", "OPTIONS", "--target", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    if "ro" not in result.stdout.strip().split(","):
        raise SystemExit(f"Refusing to read a source volume not mounted read-only: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--source", action="append", type=Path, help="UKB phenotype CSV; repeat for cross-check exports.")
    parser.add_argument("--paired-csv", type=Path, help="Strict paired eye manifest from Step 9.")
    parser.add_argument("--chunksize", type=int, default=1000)
    args = parser.parse_args()
    paths = resolve_runtime_arguments(args)
    defaults = DeploymentDefaults.load(paths.project_root)
    if not args.source:
        args.source = [defaults.ukb_label_root / "ukb670300.csv"]
    args.paired_csv = args.paired_csv or paths.dataset_root / "pairing_reports/strict_paired_fundus_oct.csv"
    args.paths = paths
    return args


def main() -> None:
    args = parse_args()
    paths = args.paths
    paired_source = args.paired_csv.resolve()
    if not paired_source.is_file():
        raise SystemExit(f"Missing paired-eye source: {paired_source}")

    sources = [source.resolve() for source in args.source]
    for source in sources:
        if not source.is_file():
            raise SystemExit(f"Missing phenotype table: {source}")
        require_read_only(source)

    phenotype_root = paths.dataset_root / "phenotypes"
    paired_destination = paths.dataset_root / "paired_eye_manifest.csv"
    paired_report = build_paired_visit_manifest(paired_source, paired_destination)
    participants = set(pd.read_csv(paired_destination, dtype=str)["participant_id"])

    tables = []
    for source in sources:
        destination = phenotype_root / "matched_fields" / f"{source.stem}_matched.csv"
        report = extract_selected_table(source, destination, participants, chunksize=args.chunksize)
        if report["manual_retinal_309xx_present"]:
            raise RuntimeError(f"Unexpected expert retinal grading fields found in {source}")
        if report["missing_participants"]:
            raise RuntimeError(
                f"{source} is missing {report['missing_participants']} paired participants"
            )
        tables.append(report)

    manifest = {
        "schema_version": 1,
        "paired_visit_manifest": paired_report,
        "phenotype_tables": tables,
        "reference_standard": "record_derived_clinical_phenotype",
        "expert_retinal_grading_30905_30943_available": False,
        "source_volumes_read_only": True,
    }
    manifest_path = phenotype_root / "extraction_manifest.json"
    write_json_atomic(manifest, manifest_path)
    print(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
