#!/usr/bin/env python3
"""Step 12: Build and execute the approved five-class UKB eye cohort."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from look_core.cli import add_runtime_arguments, add_source_arguments, resolve_runtime_arguments, resolve_source_arguments


ROOT: Path
CANDIDATES: Path
REFERENCE_LABELS: Path
CLASS_MAPPING: Path
COHORT_FLOW: Path
EXCLUSIONS: Path
IMAGE_FIELDS = ("21015", "21016", "21017", "21018")
CLASS_TO_ID = {
    "normal": 0,
    "diabetes_related_eye_disease": 1,
    "glaucoma": 2,
    "cataract": 3,
    "macular_degeneration": 4,
}
SPLIT_SEED = "ukb-eye-5class-v1"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    add_source_arguments(parser)
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write final artifacts and delete excluded output images.",
    )
    return resolve_source_arguments(parser.parse_args())


def require_read_only(path: Path) -> None:
    result = subprocess.run(
        ["findmnt", "-no", "OPTIONS", "--target", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    if "ro" not in result.stdout.strip().split(","):
        raise SystemExit(f"Source is not mounted read-only: {path}")


def safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe image path: {value!r}")
    if path.parts[0] not in IMAGE_FIELDS or path.suffix.lower() != ".png":
        raise ValueError(f"Unexpected image path: {value!r}")
    return path


def participant_split(eid: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SEED}:{eid}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 7_000:
        return "train"
    if bucket < 8_500:
        return "validation"
    return "test"


def exclusion_reason(row: dict[str, str]) -> str | None:
    if row["array"] != "0":
        return "duplicate_array_not_zero"
    if row["candidate_status"] == "multilabel":
        return "multiple_diseases_in_same_eye"
    if row["candidate_status"] == "contralateral_only":
        return "disease_reported_only_for_contralateral_eye"
    if row["candidate_status"] == "unknown_response":
        return "unknown_or_prefer_not_to_answer"
    if row["candidate_status"] == "missing_response":
        return "missing_eye_disease_response"
    if row["candidate_status"] not in {"normal_candidate", "single_disease"}:
        return f"unapproved_status:{row['candidate_status']}"
    if row["candidate_label"] == "other_serious_eye_condition":
        return "heterogeneous_other_serious_eye_condition"
    if row["candidate_label"] == "vision_loss_trauma":
        return "vision_loss_trauma_outside_target_diseases"
    if row["candidate_label"] not in CLASS_TO_ID:
        return f"unapproved_label:{row['candidate_label']}"
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    global ROOT, CANDIDATES, REFERENCE_LABELS, CLASS_MAPPING, COHORT_FLOW, EXCLUSIONS
    args = parse_args()
    ROOT = (args.root or resolve_runtime_arguments(args).dataset_root).resolve()
    CANDIDATES = ROOT / "labels/curation/ukb_eye_label_candidates.csv"
    REFERENCE_LABELS = ROOT / "reference_labels.csv"
    CLASS_MAPPING = ROOT / "class_mapping.json"
    COHORT_FLOW = ROOT / "cohort_flow.json"
    EXCLUSIONS = ROOT / "cohort_exclusions.csv.gz"
    if not CANDIDATES.is_file():
        raise SystemExit(f"Missing Step 11 candidate manifest: {CANDIDATES}")
    for source in (args.source_root, args.label_root / "ukb670300.csv"):
        require_read_only(source)

    eligible: list[dict[str, str | int]] = []
    excluded: list[dict[str, str]] = []
    keep_paths: set[Path] = set()
    input_status: collections.Counter[str] = collections.Counter()
    exclusion_counts: collections.Counter[str] = collections.Counter()
    class_counts: collections.Counter[str] = collections.Counter()
    split_class_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    split_participants: dict[str, set[str]] = collections.defaultdict(set)
    sample_keys: set[tuple[str, str, str, str]] = set()

    with CANDIDATES.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            input_status[row["candidate_status"]] += 1
            reason = exclusion_reason(row)
            if reason:
                exclusion_counts[reason] += 1
                excluded.append(
                    {
                        "participant_id": row["participant_id"],
                        "instance": row["instance"],
                        "array": row["array"],
                        "eye": row["eye"],
                        "candidate_status": row["candidate_status"],
                        "candidate_label": row["candidate_label"],
                        "exclusion_reason": reason,
                        "fundus_path": row["fundus_paths"],
                        "oct_path": row["oct_paths"],
                    }
                )
                continue

            key = (
                row["participant_id"], row["instance"], row["array"], row["eye"]
            )
            if key in sample_keys:
                raise ValueError(f"Duplicate eligible sample key: {key}")
            sample_keys.add(key)
            fundus = safe_relative_path(row["fundus_paths"])
            oct_image = safe_relative_path(row["oct_paths"])
            if fundus in keep_paths or oct_image in keep_paths:
                raise ValueError(f"Image path reused by multiple samples: {key}")
            keep_paths.update((fundus, oct_image))
            split = participant_split(row["participant_id"])
            label = row["candidate_label"]
            class_counts[label] += 1
            split_class_counts[(split, label)] += 1
            split_participants[split].add(row["participant_id"])
            eligible.append(
                {
                    "participant_id": row["participant_id"],
                    "instance": row["instance"],
                    "array": row["array"],
                    "eye": row["eye"],
                    "fundus_path": str(fundus),
                    "oct_path": str(oct_image),
                    "label_id": CLASS_TO_ID[label],
                    "label_name": label,
                    "split": split,
                    "reference_source": "ukb_doctor_informed_self_report",
                    "reference_fields": "6148;5890;6119;5441;5912",
                }
            )

    if len(eligible) != 162_202:
        raise ValueError(f"Expected 162,202 eligible pairs, got {len(eligible):,}")
    if len(keep_paths) != 2 * len(eligible):
        raise ValueError("Eligible paths are not one-to-one")

    all_paths: set[Path] = set()
    for field in IMAGE_FIELDS:
        all_paths.update(path.relative_to(ROOT) for path in (ROOT / field).rglob("*.png"))
    missing_keep = keep_paths - all_paths
    delete_paths = all_paths - keep_paths
    if missing_keep:
        raise ValueError(f"Missing {len(missing_keep):,} eligible images")
    if len(all_paths) != 350_310 or len(delete_paths) != 25_906:
        raise ValueError(
            f"Unexpected image inventory: all={len(all_paths):,}, "
            f"delete={len(delete_paths):,}"
        )

    preview = {
        "execute": args.execute,
        "input_rows": sum(input_status.values()),
        "eligible_pairs": len(eligible),
        "eligible_images": len(keep_paths),
        "images_to_delete": len(delete_paths),
        "class_counts": dict(sorted(class_counts.items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
    }
    print(json.dumps(preview, indent=2))
    if not args.execute:
        print("Dry run only. Re-run with --execute after reviewing these counts.")
        return 0
    for output in (REFERENCE_LABELS, CLASS_MAPPING, COHORT_FLOW, EXCLUSIONS):
        if output.exists():
            raise SystemExit(f"Refusing to overwrite existing final artifact: {output}")

    labels_partial = REFERENCE_LABELS.with_suffix(".csv.partial")
    exclusions_partial = EXCLUSIONS.with_suffix(".csv.gz.partial")
    labels_partial.unlink(missing_ok=True)
    exclusions_partial.unlink(missing_ok=True)
    fieldnames = list(eligible[0])
    with labels_partial.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eligible)
    with gzip.open(exclusions_partial, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(excluded[0]))
        writer.writeheader()
        writer.writerows(excluded)
    os.replace(labels_partial, REFERENCE_LABELS)
    os.replace(exclusions_partial, EXCLUSIONS)

    CLASS_MAPPING.write_text(
        json.dumps(
            {
                "dataset": "UKB paired fundus-OCT five-class cohort",
                "classes": [
                    {"id": class_id, "name": name}
                    for name, class_id in CLASS_TO_ID.items()
                ],
                "reference_standard_type": (
                    "Doctor-informed participant self-report with eye laterality; "
                    "not expert image grading."
                ),
                "split_method": (
                    "Participant-level deterministic SHA-256 hash, 70/15/15"
                ),
                "split_seed": SPLIT_SEED,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    deletion_failures: list[str] = []
    for number, relative in enumerate(sorted(delete_paths), start=1):
        try:
            (ROOT / relative).unlink()
        except OSError as error:
            deletion_failures.append(f"{relative}: {error}")
        if number % 5_000 == 0:
            print(f"deleted={number:,}/{len(delete_paths):,}", flush=True)
    if deletion_failures:
        raise RuntimeError(
            f"Failed to delete {len(deletion_failures)} images; "
            f"first={deletion_failures[0]}"
        )
    for field in IMAGE_FIELDS:
        directories = sorted(
            (path for path in (ROOT / field).rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            try:
                directory.rmdir()
            except OSError:
                pass

    cohort_flow = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "approved_task": "five-class paired fundus-OCT classification",
        "classes": CLASS_TO_ID,
        "input_candidate_rows": sum(input_status.values()),
        "input_status_counts": dict(sorted(input_status.items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "final_pairs": len(eligible),
        "final_images": len(keep_paths),
        "class_counts": dict(sorted(class_counts.items())),
        "split_class_counts": {
            split: {
                label: split_class_counts[(split, label)] for label in CLASS_TO_ID
            }
            for split in ("train", "validation", "test")
        },
        "split_unique_participants": {
            split: len(split_participants[split])
            for split in ("train", "validation", "test")
        },
        "deleted_output_images": len(delete_paths),
        "source_mounts_read_only": True,
        "reference_labels_sha256": sha256(REFERENCE_LABELS),
        "reference_standard_limitation": (
            "Fields are doctor-informed participant self-report, not masked expert "
            "image adjudication. They must not be described as image-grading ground truth."
        ),
    }
    COHORT_FLOW.write_text(
        json.dumps(cohort_flow, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(cohort_flow, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
