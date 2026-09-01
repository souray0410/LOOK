#!/usr/bin/env python3
"""Step 11: Build a non-destructive eye-level label candidate manifest."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import time
from pathlib import Path

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments


CLASS_FIELDS = {
    "1": ("diabetes_related_eye_disease", "5890"),
    "2": ("glaucoma", "6119"),
    "3": ("vision_loss_trauma", "5419"),
    "4": ("cataract", "5441"),
    "5": ("macular_degeneration", "5912"),
    "6": ("other_serious_eye_condition", "5934"),
}
SIDE_CODES = {
    "right": {"1", "3"},
    "left": {"2", "3"},
}
VALID_SIDE_CODES = {"1", "2", "3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--phenotype", type=Path)
    parser.add_argument("--crosscheck", type=Path)
    parser.add_argument("--paired-csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()
    root = resolve_runtime_arguments(args).dataset_root
    args.phenotype = args.phenotype or root / "labels/matched_raw/ukb670300_matched.csv"
    args.crosscheck = args.crosscheck or root / "labels/matched_raw/ukb679947_matched.csv"
    args.paired_csv = args.paired_csv or root / "pairing_reports/strict_paired_fundus_oct.csv"
    args.output = args.output or root / "labels/curation/ukb_eye_label_candidates.csv"
    return args


def required_columns() -> list[str]:
    columns: list[str] = []
    for instance in ("0", "1"):
        columns.extend(f"6148-{instance}.{array}" for array in range(5))
        columns.extend(
            f"{field}-{instance}.0" for _, field in CLASS_FIELDS.values()
        )
    return columns


def read_selected(
    path: Path,
    needed: list[str],
    progress_every: int,
) -> tuple[dict[str, tuple[str, ...]], int]:
    selected: dict[str, tuple[str, ...]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {column: position for position, column in enumerate(header)}
        missing = [column for column in needed if column not in index]
        if not header or header[0] != "eid" or missing:
            raise ValueError(f"Invalid phenotype schema in {path}; missing={missing}")
        positions = [index[column] for column in needed]
        for number, row in enumerate(reader, start=1):
            selected[row[0]] = tuple(row[position].strip() for position in positions)
            if number % progress_every == 0:
                print(f"[{path.name}] rows={number:,}", flush=True)
    return selected, len(header)


def classify_eye(
    values: tuple[str, ...],
    needed_index: dict[str, int],
    instance: str,
    eye: str,
) -> dict[str, str]:
    response = {
        values[needed_index[f"6148-{instance}.{array}"]]
        for array in range(5)
        if values[needed_index[f"6148-{instance}.{array}"]]
    }
    positive_codes = sorted(response & set(CLASS_FIELDS), key=int)
    eye_labels: list[str] = []
    contralateral: list[str] = []
    unresolved: list[str] = []
    inconsistent: list[str] = []

    for code, (label, side_field) in CLASS_FIELDS.items():
        side = values[needed_index[f"{side_field}-{instance}.0"]]
        if code in response:
            if side not in VALID_SIDE_CODES:
                unresolved.append(label)
            elif side in SIDE_CODES[eye]:
                eye_labels.append(label)
            else:
                contralateral.append(label)
        elif side:
            inconsistent.append(f"{label}:side={side}_without_6148")

    if response == {"-7"} and not inconsistent:
        status, label = "normal_candidate", "normal"
    elif response & {"-1", "-3"}:
        status, label = "unknown_response", ""
    elif inconsistent:
        status, label = "inconsistent_fields", ""
    elif unresolved:
        status, label = "unresolved_side", ""
    elif len(eye_labels) > 1:
        status, label = "multilabel", ""
    elif len(eye_labels) == 1:
        status, label = "single_disease", eye_labels[0]
    elif positive_codes and len(contralateral) == len(positive_codes):
        status, label = "contralateral_only", ""
    elif not response:
        status, label = "missing_response", ""
    else:
        status, label = "unresolved_response", ""

    return {
        "candidate_status": status,
        "candidate_label": label,
        "raw_6148_codes": ";".join(sorted(response, key=lambda x: int(x))),
        "eye_positive_labels": ";".join(eye_labels),
        "contralateral_labels": ";".join(contralateral),
        "unresolved_labels": ";".join(unresolved),
        "field_inconsistencies": ";".join(inconsistent),
    }


def main() -> int:
    args = parse_args()
    phenotype = args.phenotype.resolve()
    crosscheck = args.crosscheck.resolve()
    paired_csv = args.paired_csv.resolve()
    output = args.output.resolve()
    temporary = output.with_name(output.name + ".partial")
    report_path = output.with_suffix(".audit.json")
    for path in (phenotype, crosscheck, paired_csv):
        if not path.is_file():
            raise SystemExit(f"Missing input: {path}")

    needed = required_columns()
    needed_index = {column: index for index, column in enumerate(needed)}
    started = time.monotonic()
    primary, primary_columns = read_selected(
        phenotype, needed, args.progress_every
    )
    secondary, secondary_columns = read_selected(
        crosscheck, needed, args.progress_every
    )
    if set(primary) != set(secondary):
        raise ValueError("The two matched phenotype tables contain different EIDs")
    crosscheck_mismatches = sum(
        primary[eid] != secondary[eid] for eid in primary
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    status_counts: collections.Counter[str] = collections.Counter()
    label_counts: collections.Counter[str] = collections.Counter()
    eye_visit_counts: collections.Counter[tuple[str, str, str]] = (
        collections.Counter()
    )
    output_rows = 0
    fieldnames: list[str] = []
    try:
        with (
            paired_csv.open(newline="", encoding="utf-8") as input_handle,
            temporary.open("x", newline="", encoding="utf-8") as output_handle,
        ):
            reader = csv.DictReader(input_handle)
            fieldnames = list(reader.fieldnames or []) + [
                "candidate_status",
                "candidate_label",
                "raw_6148_codes",
                "eye_positive_labels",
                "contralateral_labels",
                "unresolved_labels",
                "field_inconsistencies",
            ]
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                eid = row["participant_id"]
                if eid not in primary:
                    raise ValueError(f"Paired participant missing from phenotype: {eid}")
                result = classify_eye(
                    primary[eid], needed_index, row["instance"], row["eye"]
                )
                row.update(result)
                writer.writerow(row)
                output_rows += 1
                status_counts[result["candidate_status"]] += 1
                if result["candidate_label"]:
                    label_counts[result["candidate_label"]] += 1
                eye_visit_counts[
                    (eid, row["instance"], row["eye"])
                ] += 1
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "purpose": "Non-destructive candidate audit; not a final ground-truth cohort.",
        "label_provenance": (
            "UK Biobank touchscreen field 6148 and conditional eye-side fields "
            "5890, 6119, 5419, 5441, 5912, and 5934."
        ),
        "phenotype": str(phenotype),
        "crosscheck": str(crosscheck),
        "paired_csv": str(paired_csv),
        "output": str(output),
        "phenotype_rows": len(primary),
        "primary_columns": primary_columns,
        "crosscheck_columns": secondary_columns,
        "crosscheck_value_mismatched_eids": crosscheck_mismatches,
        "output_rows": output_rows,
        "unique_eye_visit_keys": len(eye_visit_counts),
        "duplicate_array_rows": sum(value - 1 for value in eye_visit_counts.values()),
        "status_counts": dict(sorted(status_counts.items())),
        "candidate_label_counts": dict(sorted(label_counts.items())),
        "output_bytes": output.stat().st_size,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "destructive_changes": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Candidate manifest: {output}")
    print(f"Audit report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
