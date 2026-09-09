#!/usr/bin/env python3
"""Step 8: Remove only files listed by the strict unpaired pairing audit."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments


EXPECTED_UNPAIRED_FILES = 17_317
ALLOWED_FIELDS = {"21015", "21016", "21017", "21018"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete paths in the strict unpaired manifest after validation."
    )
    add_runtime_arguments(parser)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--expected-count", type=int, default=EXPECTED_UNPAIRED_FILES
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform deletion. Without this flag, only validate and report.",
    )
    args = parser.parse_args()
    dataset_root = resolve_runtime_arguments(args).dataset_root
    args.root = args.root or dataset_root
    args.manifest = args.manifest or args.root / "pairing_reports/strict_unpaired_fundus_oct.csv"
    return args


def validated_paths(root: Path, manifest: Path) -> list[Path]:
    root = root.resolve()
    paths: list[Path] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            modality = row["available_modality"]
            if modality not in {"fundus_only", "oct_only"}:
                raise ValueError(f"Unexpected modality: {modality}")
            for relative_text in row["paths"].split(";"):
                relative = Path(relative_text)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe relative path: {relative_text}")
                if not relative.parts or relative.parts[0] not in ALLOWED_FIELDS:
                    raise ValueError(f"Path outside image fields: {relative_text}")
                path = (root / relative).resolve()
                if path.parent != root and root not in path.parents:
                    raise ValueError(f"Path escaped dataset root: {path}")
                paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError("Unpaired manifest contains duplicate paths")
    return sorted(paths)


def remove_empty_parents(path: Path, stop: Path) -> None:
    parent = path.parent
    while parent != stop and stop in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest = args.manifest.resolve()
    paths = validated_paths(root, manifest)
    existing = [path for path in paths if path.is_file()]
    missing = [path for path in paths if not path.exists()]
    total_bytes = sum(path.stat().st_size for path in existing)

    print(f"Dataset root:   {root}")
    print(f"Manifest:       {manifest}")
    print(f"Listed paths:   {len(paths):,}")
    print(f"Existing files: {len(existing):,}")
    print(f"Missing paths:  {len(missing):,}")
    print(f"Bytes to free:  {total_bytes / 1024**3:.2f} GiB")

    if len(paths) != args.expected_count:
        raise SystemExit(
            f"Refusing deletion: expected {args.expected_count:,} paths, "
            f"manifest contains {len(paths):,}"
        )
    if missing:
        raise SystemExit("Refusing deletion because one or more paths are missing")
    if not args.execute:
        print("Validation passed. Re-run with --execute to delete these files.")
        return 0

    report_dir = root / "pairing_reports"
    log_path = report_dir / f"removed_unpaired_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes"])
        writer.writeheader()
        for index, path in enumerate(existing, start=1):
            byte_count = path.stat().st_size
            relative = path.relative_to(root)
            path.unlink()
            writer.writerow({"relative_path": relative, "bytes": byte_count})
            if index % 1000 == 0:
                handle.flush()
                os.fsync(handle.fileno())
                print(f"Deleted {index:,}/{len(existing):,}", flush=True)
            remove_empty_parents(path, root)

    print(f"Deleted files:  {len(existing):,}")
    print(f"Deletion log:   {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
