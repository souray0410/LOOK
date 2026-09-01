#!/usr/bin/env python3
"""Step 10: Extract all phenotype columns for participants with paired eye images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.defaults import DeploymentDefaults


MANUAL_RETINAL_FIELDS = {
    "30905", "30906", "30907", "30908", "30910", "30911", "30912",
    "30913", "30914", "30915", "30916", "30917", "30918", "30919",
    "30920", "30921", "30922", "30923", "30924", "30925", "30926",
    "30927", "30928", "30929", "30930", "30931", "30932", "30933",
    "30934", "30935", "30936", "30937", "30938", "30939", "30940",
    "30941", "30942", "30943",
}
POTENTIAL_LABEL_FIELDS = {
    "6148",   # Self-reported eye problems/disorders
    "20002",  # Self-reported non-cancer illness
    "40006",  # Cancer type
    "41202",  # Main ICD-10 diagnoses
    "41270",  # All ICD-10 diagnoses
    "42040",  # GP clinical event records
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        help="Phenotype CSV; may be supplied multiple times.",
    )
    parser.add_argument("--paired-csv", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--progress-every", type=int, default=50_000)
    args = parser.parse_args()
    defaults = DeploymentDefaults.load(args.project_root)
    if not args.source:
        args.source = [defaults.ukb_label_root / "ukb670300.csv", defaults.ukb_label_root / "ukb679947.csv"]
    dataset_root = resolve_runtime_arguments(args).dataset_root
    args.paired_csv = args.paired_csv or dataset_root / "pairing_reports/strict_paired_fundus_oct.csv"
    args.output_root = args.output_root or dataset_root / "labels"
    return args


def require_read_only(path: Path) -> None:
    result = subprocess.run(
        ["findmnt", "-no", "OPTIONS", "--target", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    if "ro" not in result.stdout.strip().split(","):
        raise SystemExit(f"Refusing to read a source volume not mounted ro: {path}")


def paired_participants(path: Path) -> set[str]:
    participants: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            participants.add(row["participant_id"])
    if not participants:
        raise ValueError("Paired image manifest contains no participants")
    return participants


def header_fields(header: bytes) -> list[str]:
    text = header.decode("utf-8-sig")
    return next(csv.reader([text]))


def base_field(column: str) -> str:
    return column.split("-", 1)[0]


def extract_table(
    source: Path,
    destination: Path,
    participants: set[bytes],
    metadata_dir: Path,
    progress_every: int,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    source_rows = 0
    matched_rows = 0
    matched_ids: set[bytes] = set()
    malformed_rows = 0
    started = time.monotonic()

    with source.open("rb") as input_handle:
        header = input_handle.readline()
        columns = header_fields(header)
        if not columns or columns[0] != "eid":
            raise ValueError(f"First column is not eid in {source}")
        column_index = metadata_dir / f"{source.stem}.columns.txt"
        column_index.write_text("\n".join(columns) + "\n", encoding="utf-8")
        field_ids = {base_field(column) for column in columns[1:]}

        try:
            with temporary.open("xb") as output_handle:
                output_handle.write(header)
                digest.update(header)
                for line in input_handle:
                    source_rows += 1
                    first = line.split(b",", 1)[0].strip().strip(b'"')
                    if not first.isdigit():
                        malformed_rows += 1
                        continue
                    if first in participants:
                        output_handle.write(line)
                        digest.update(line)
                        matched_rows += 1
                        matched_ids.add(first)
                    if source_rows % progress_every == 0:
                        elapsed = max(time.monotonic() - started, 0.001)
                        print(
                            f"[{source.name}] rows={source_rows:,}, "
                            f"matched={matched_rows:,}, "
                            f"rate={source_rows / elapsed:.1f} rows/s",
                            flush=True,
                        )
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    return {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "destination": str(destination),
        "destination_bytes": destination.stat().st_size,
        "sha256": digest.hexdigest(),
        "columns": len(columns),
        "unique_field_ids": len(field_ids),
        "source_rows": source_rows,
        "matched_rows": matched_rows,
        "matched_unique_eids": len(matched_ids),
        "missing_paired_eids": len(participants - matched_ids),
        "malformed_rows": malformed_rows,
        "manual_retinal_fields_present": sorted(MANUAL_RETINAL_FIELDS & field_ids),
        "potential_label_fields_present": sorted(POTENTIAL_LABEL_FIELDS & field_ids),
    }


def main() -> int:
    args = parse_args()
    sources = tuple(args.source or DEFAULT_SOURCES)
    paired_csv = args.paired_csv.resolve()
    output_root = args.output_root.resolve()
    matched_dir = output_root / "matched_raw"
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    require_read_only(sources[0].resolve())

    participant_text = paired_participants(paired_csv)
    participant_bytes = {value.encode("ascii") for value in participant_text}
    eid_path = output_root / "paired_image_eids.csv"
    with eid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["eid"])
        writer.writerows([eid] for eid in sorted(participant_text, key=int))

    results = []
    for source in sources:
        source = source.resolve()
        if not source.is_file():
            raise SystemExit(f"Missing source phenotype table: {source}")
        destination = matched_dir / f"{source.stem}_matched.csv"
        results.append(
            extract_table(
                source,
                destination,
                participant_bytes,
                metadata_dir,
                args.progress_every,
            )
        )

    source_docs = sources[0].parent.parent / "docs/ukb670300.html"
    if source_docs.is_file():
        shutil.copy2(source_docs, metadata_dir / source_docs.name)
    field_dictionary = Path(__file__).resolve().parents[1] / "tool/research/data_dictionary/UKB_RETINAL_GRADING_FIELDS.csv"
    if field_dictionary.is_file():
        shutil.copy2(field_dictionary, metadata_dir / field_dictionary.name)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "paired_manifest": str(paired_csv),
        "paired_unique_eids": len(participant_text),
        "note": (
            "All available phenotype columns were retained. Fields 30905-30943 "
            "are absent from both source exports and were not fabricated."
        ),
        "tables": results,
    }
    manifest_path = output_root / "phenotype_match_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
