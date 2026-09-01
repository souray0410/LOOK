#!/usr/bin/env python3
"""Step 13: Verify the final five-class UKB eye dataset end to end."""

from __future__ import annotations

import collections
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from look_core.cli import add_runtime_arguments, add_source_arguments, resolve_runtime_arguments, resolve_source_arguments
from look_core.state import atomic_write_json


ROOT: Path
LABELS: Path
MAPPING: Path
FLOW: Path
OUTPUT: Path
DATA_MANIFEST: Path
IMAGE_FIELDS = ("21015", "21016", "21017", "21018")
EXPECTED = {
    "normal": 150_378,
    "diabetes_related_eye_disease": 1_326,
    "glaucoma": 2_225,
    "cataract": 6_958,
    "macular_degeneration": 1_315,
}
def is_read_only(path: Path) -> bool:
    result = subprocess.run(
        ["findmnt", "-no", "OPTIONS", "--target", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return "ro" in result.stdout.strip().split(",")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    global ROOT, LABELS, MAPPING, FLOW, OUTPUT, DATA_MANIFEST
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    add_source_arguments(parser)
    parser.add_argument("--root", type=Path)
    args = resolve_source_arguments(parser.parse_args())
    ROOT = (args.root or resolve_runtime_arguments(args).dataset_root).resolve()
    LABELS = ROOT / "reference_labels.csv"
    MAPPING = ROOT / "class_mapping.json"
    FLOW = ROOT / "cohort_flow.json"
    OUTPUT = ROOT / "verification.json"
    DATA_MANIFEST = ROOT.parent / "data_manifest.json"
    errors: list[str] = []
    for path in (LABELS, MAPPING, FLOW):
        if not path.is_file():
            errors.append(f"missing artifact: {path}")
    if errors:
        raise SystemExit("\n".join(errors))

    mapping = json.loads(MAPPING.read_text())
    valid = {item["name"]: item["id"] for item in mapping["classes"]}
    rows = 0
    class_counts: collections.Counter[str] = collections.Counter()
    split_counts: collections.Counter[str] = collections.Counter()
    split_class: collections.Counter[tuple[str, str]] = collections.Counter()
    participant_splits: dict[str, set[str]] = collections.defaultdict(set)
    sample_keys: set[tuple[str, str, str, str]] = set()
    referenced: set[Path] = set()
    with LABELS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            label = row["label_name"]
            if label not in valid or int(row["label_id"]) != valid[label]:
                errors.append(f"invalid label at row {rows + 1}")
            if row["array"] != "0":
                errors.append(f"nonzero array at row {rows + 1}")
            key = (
                row["participant_id"], row["instance"], row["array"], row["eye"]
            )
            if key in sample_keys:
                errors.append(f"duplicate sample key: {key}")
            sample_keys.add(key)
            split = row["split"]
            if split not in {"train", "validation", "test"}:
                errors.append(f"invalid split at row {rows + 1}")
            participant_splits[row["participant_id"]].add(split)
            class_counts[label] += 1
            split_counts[split] += 1
            split_class[(split, label)] += 1
            for column in ("fundus_path", "oct_path"):
                relative = Path(row[column])
                if relative.is_absolute() or ".." in relative.parts:
                    errors.append(f"unsafe path at row {rows + 1}: {relative}")
                if relative in referenced:
                    errors.append(f"reused image path: {relative}")
                referenced.add(relative)
                if not (ROOT / relative).is_file():
                    errors.append(f"missing image: {relative}")

    leaked = sorted(eid for eid, splits in participant_splits.items() if len(splits) > 1)
    if leaked:
        errors.append(f"participant split leakage: {len(leaked)}")
    if rows != 162_202:
        errors.append(f"expected 162202 rows, got {rows}")
    if dict(class_counts) != EXPECTED:
        errors.append(f"class counts differ: {dict(class_counts)}")
    if len(referenced) != 324_404:
        errors.append(f"expected 324404 referenced images, got {len(referenced)}")

    disk_images: set[Path] = set()
    field_counts: dict[str, int] = {}
    for field in IMAGE_FIELDS:
        paths = {path.relative_to(ROOT) for path in (ROOT / field).rglob("*.png")}
        field_counts[field] = len(paths)
        disk_images.update(paths)
    missing = referenced - disk_images
    extraneous = disk_images - referenced
    if missing:
        errors.append(f"missing referenced disk images: {len(missing)}")
    if extraneous:
        errors.append(f"extraneous disk images: {len(extraneous)}")
    source_read_only = all(
        is_read_only(path) for path in (args.source_root, args.label_root / "ukb670300.csv")
    )
    if not source_read_only:
        errors.append("one or more source mounts are not read-only")

    result = {
        "verified_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": "PASS" if not errors else "FAIL",
        "pairs": rows,
        "images": len(disk_images),
        "field_counts": field_counts,
        "class_counts": dict(sorted(class_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_class_counts": {
            split: {label: split_class[(split, label)] for label in valid}
            for split in ("train", "validation", "test")
        },
        "unique_participants": len(participant_splits),
        "participant_split_leakage": len(leaked),
        "missing_images": len(missing),
        "extraneous_images": len(extraneous),
        "source_mounts_read_only": source_read_only,
        "reference_labels_sha256": sha256(LABELS),
        "errors": errors[:100],
    }
    atomic_write_json(result, OUTPUT)
    if not errors:
        critical = (LABELS, MAPPING, FLOW, ROOT / "cohort_exclusions.csv.gz", OUTPUT)
        sample_paths = sorted(
            disk_images,
            key=lambda path: hashlib.sha256(str(path).encode("utf-8")).digest(),
        )[:32]
        dataset_bytes = int(
            subprocess.check_output(["du", "-sb", str(ROOT)], text=True).split()[0]
        )
        atomic_write_json(
            {
                "schema_version": 2,
                "project": "LOOK",
                "status": "PASS",
                "dataset_root": str(ROOT),
                "verified_at": result["verified_at"],
                "inventory": {
                    "bytes": dataset_bytes,
                    "files": len(disk_images) + len(critical),
                    "png_files": len(disk_images),
                    "pairs": rows,
                    "participants": len(participant_splits),
                    "participant_split_leakage": len(leaked),
                    "missing_images": len(missing),
                    "extraneous_images": len(extraneous),
                    "critical_sha256": {path.name: sha256(path) for path in critical},
                    "sample_image_sha256": {
                        str(path): sha256(ROOT / path) for path in sample_paths
                    },
                },
            },
            DATA_MANIFEST,
        )
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
