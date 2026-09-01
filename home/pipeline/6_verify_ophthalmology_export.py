#!/usr/bin/env python3
"""Step 6: Verify the complete pre-pruning export against the source."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.defaults import DeploymentDefaults
from look_core.state import atomic_write_text


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def image_member_count(archive: Path) -> int:
    with zipfile.ZipFile(archive) as bundle:
        return sum(
            1
            for info in bundle.infolist()
            if not info.is_dir()
            and PurePosixPath(info.filename).suffix.lower() in IMAGE_SUFFIXES
        )


def manifest_errors(destination: Path) -> tuple[Path | None, int]:
    manifests = sorted(
        destination.glob("ophthalmology_export_*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not manifests:
        return None, 0
    manifest = manifests[-1]
    with manifest.open(newline="", encoding="utf-8") as handle:
        errors = sum(1 for row in csv.DictReader(handle) if row["status"] == "error")
    return manifest, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    add_runtime_arguments(parser)
    parser.add_argument("--destination-root", type=Path)
    args = parser.parse_args()
    defaults = DeploymentDefaults.load(args.project_root)
    args.source_root = args.source_root or defaults.ukb_source_root
    source = args.source_root.resolve()
    destination = args.destination_root or resolve_runtime_arguments(args).dataset_root
    report_path = destination / "ophthalmology_verification.txt"
    expected = {
        "21015": image_member_count(source / "21015.zip"),
        "21016": image_member_count(source / "21016.zip"),
        "21017": sum(1 for _ in (source / "21017").rglob("*.zip")),
        "21018": sum(1 for _ in (source / "21018").rglob("*.zip")),
    }
    actual = {
        field_id: sum(1 for _ in (destination / field_id).rglob("*.png"))
        for field_id in expected
    }
    partials = sum(1 for _ in destination.rglob("*.partial"))
    manifest, errors = manifest_errors(destination)
    free_gib = shutil.disk_usage(destination).free / 1024**3
    success = expected == actual and partials == 0 and errors == 0

    lines = [
        f"verified_at={time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"status={'PASS' if success else 'FAIL'}",
    ]
    for field_id in expected:
        lines.append(
            f"field={field_id} expected={expected[field_id]} actual={actual[field_id]}"
        )
    lines.extend(
        [
            f"partial_files={partials}",
            f"manifest={manifest or 'missing'}",
            f"manifest_errors={errors}",
            f"free_space_gib={free_gib:.1f}",
        ]
    )
    report = "\n".join(lines) + "\n"
    atomic_write_text(report, report_path)
    print(report, end="")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
