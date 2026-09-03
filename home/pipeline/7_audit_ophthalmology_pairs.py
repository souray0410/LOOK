#!/usr/bin/env python3
"""Step 7: Audit fundus/OCT pairing without modifying exported images."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments


NAME_RE = re.compile(
    r"^(?P<participant>\d+)_(?P<field>2101[5-8])_"
    r"(?P<instance>\d+)_(?P<array>\d+)$"
)


@dataclass(frozen=True, order=True)
class Key:
    participant: str
    instance: int
    array: int


FIELD_INFO = {
    "left": {"fundus": "21015", "oct": "21017"},
    "right": {"fundus": "21016", "oct": "21018"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    dataset_root = resolve_runtime_arguments(args).dataset_root
    args.root = args.root or dataset_root
    args.report_dir = args.report_dir or dataset_root / "pairing_reports"
    return args


def parse_key(stem: str, expected_field: str) -> Key:
    match = NAME_RE.match(stem)
    if not match or match.group("field") != expected_field:
        raise ValueError(f"Unexpected dataset name: {stem}")
    return Key(
        participant=match.group("participant"),
        instance=int(match.group("instance")),
        array=int(match.group("array")),
    )


def scan_fundus(root: Path, field_id: str) -> dict[Key, list[Path]]:
    records: dict[Key, list[Path]] = defaultdict(list)
    for path in (root / field_id).rglob("*.png"):
        records[parse_key(path.stem, field_id)].append(path)
    return records


def scan_oct(root: Path, field_id: str) -> dict[Key, list[Path]]:
    records: dict[Key, list[Path]] = defaultdict(list)
    for path in (root / field_id).rglob("*.png"):
        records[parse_key(path.parent.name, field_id)].append(path)
    return records


def visit_key(key: Key) -> tuple[str, int]:
    return key.participant, key.instance


def relative_paths(paths: list[Path], root: Path) -> str:
    return ";".join(str(path.relative_to(root)) for path in sorted(paths))


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "pairing_key": "participant + instance + array + eye",
        "root": str(root),
        "eyes": {},
    }

    paired_path = report_dir / "strict_paired_fundus_oct.csv"
    unpaired_path = report_dir / "strict_unpaired_fundus_oct.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as paired_handle, \
        unpaired_path.open("w", newline="", encoding="utf-8") as unpaired_handle:
        paired_writer = csv.DictWriter(
            paired_handle,
            fieldnames=[
                "participant_id",
                "instance",
                "array",
                "eye",
                "fundus_paths",
                "oct_paths",
            ],
        )
        unpaired_writer = csv.DictWriter(
            unpaired_handle,
            fieldnames=[
                "participant_id",
                "instance",
                "array",
                "eye",
                "available_modality",
                "paths",
            ],
        )
        paired_writer.writeheader()
        unpaired_writer.writeheader()

        for eye, fields in FIELD_INFO.items():
            fundus = scan_fundus(root, fields["fundus"])
            oct_data = scan_oct(root, fields["oct"])
            fundus_keys = set(fundus)
            oct_keys = set(oct_data)
            strict_pairs = fundus_keys & oct_keys
            fundus_only = fundus_keys - oct_keys
            oct_only = oct_keys - fundus_keys
            fundus_visits = {visit_key(key) for key in fundus_keys}
            oct_visits = {visit_key(key) for key in oct_keys}
            visit_pairs = fundus_visits & oct_visits

            for key in sorted(strict_pairs):
                paired_writer.writerow(
                    {
                        "participant_id": key.participant,
                        "instance": key.instance,
                        "array": key.array,
                        "eye": eye,
                        "fundus_paths": relative_paths(fundus[key], root),
                        "oct_paths": relative_paths(oct_data[key], root),
                    }
                )
            for key in sorted(fundus_only):
                unpaired_writer.writerow(
                    {
                        "participant_id": key.participant,
                        "instance": key.instance,
                        "array": key.array,
                        "eye": eye,
                        "available_modality": "fundus_only",
                        "paths": relative_paths(fundus[key], root),
                    }
                )
            for key in sorted(oct_only):
                unpaired_writer.writerow(
                    {
                        "participant_id": key.participant,
                        "instance": key.instance,
                        "array": key.array,
                        "eye": eye,
                        "available_modality": "oct_only",
                        "paths": relative_paths(oct_data[key], root),
                    }
                )

            summary["eyes"][eye] = {
                "fundus_measurements": len(fundus_keys),
                "oct_measurements": len(oct_keys),
                "strict_paired_measurements": len(strict_pairs),
                "strict_fundus_only": len(fundus_only),
                "strict_oct_only": len(oct_only),
                "participant_instance_fundus": len(fundus_visits),
                "participant_instance_oct": len(oct_visits),
                "participant_instance_pairs": len(visit_pairs),
                "duplicate_fundus_keys": sum(
                    1 for paths in fundus.values() if len(paths) > 1
                ),
                "duplicate_oct_keys": sum(
                    1 for paths in oct_data.values() if len(paths) > 1
                ),
            }

    summary_path = report_dir / "pairing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Paired CSV:   {paired_path}")
    print(f"Unpaired CSV: {unpaired_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
