#!/usr/bin/env python3
"""Step 9: Re-audit and verify the final strictly paired dataset."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.defaults import DeploymentDefaults
from look_core.state import atomic_write_text


SCRIPT_DIR = Path(__file__).resolve().parent
AUDIT_SCRIPT = SCRIPT_DIR / "7_audit_ophthalmology_pairs.py"
EXPECTED = {"left": 86_957, "right": 88_198}


def source_is_read_only(source_root: Path) -> bool:
    result = subprocess.run(
        ["findmnt", "-no", "OPTIONS", "--target", str(source_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    return "ro" in result.stdout.strip().split(",")


def verify_paired_csv(path: Path, root: Path) -> tuple[int, int]:
    rows = 0
    missing_paths = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            for column in ("fundus_paths", "oct_paths"):
                for relative in row[column].split(";"):
                    if not (root / relative).is_file():
                        missing_paths += 1
    return rows, missing_paths


def csv_data_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    add_runtime_arguments(parser)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    defaults = DeploymentDefaults.load(args.project_root)
    args.source_root = args.source_root or defaults.ukb_source_root
    root = (args.root or resolve_runtime_arguments(args).dataset_root).resolve()
    report_dir = root / "pairing_reports"
    subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), "--root", str(root), "--report-dir", str(report_dir)],
        check=True,
    )
    summary_path = report_dir / "pairing_summary.json"
    paired_path = report_dir / "strict_paired_fundus_oct.csv"
    unpaired_path = report_dir / "strict_unpaired_fundus_oct.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paired_rows, missing_paths = verify_paired_csv(paired_path, root)
    unpaired_rows = csv_data_rows(unpaired_path)
    source_ro = source_is_read_only(args.source_root)

    problems: list[str] = []
    expected_total = sum(EXPECTED.values())
    if paired_rows != expected_total:
        problems.append(f"paired rows {paired_rows} != {expected_total}")
    if missing_paths:
        problems.append(f"paired CSV references {missing_paths} missing files")
    if unpaired_rows:
        problems.append(f"unpaired CSV contains {unpaired_rows} rows")
    if not source_ro:
        problems.append("source volume is not read-only")

    for eye, expected in EXPECTED.items():
        data = summary["eyes"][eye]
        values = {
            data["fundus_measurements"],
            data["oct_measurements"],
            data["strict_paired_measurements"],
        }
        if values != {expected}:
            problems.append(f"{eye} counts do not all equal {expected}")
        if data["strict_fundus_only"] or data["strict_oct_only"]:
            problems.append(f"{eye} still contains unpaired measurements")

    report_path = report_dir / "final_paired_verification.txt"
    lines = [
        f"verified_at={time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"status={'FAIL' if problems else 'PASS'}",
        f"left_pairs={summary['eyes']['left']['strict_paired_measurements']}",
        f"right_pairs={summary['eyes']['right']['strict_paired_measurements']}",
        f"paired_csv_rows={paired_rows}",
        f"unpaired_csv_rows={unpaired_rows}",
        f"missing_referenced_files={missing_paths}",
        f"source_read_only={str(source_ro).lower()}",
    ]
    lines.extend(f"problem={problem}" for problem in problems)
    report = "\n".join(lines) + "\n"
    atomic_write_text(report, report_path)
    print(report, end="")
    print(f"Report: {report_path}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
