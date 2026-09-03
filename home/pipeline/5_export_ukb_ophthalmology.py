#!/usr/bin/env python3
"""Step 5: Export UK Biobank fundus images and middle OCT B-scans.

The source volume is treated as read-only. Writes are atomic and the operation is
idempotent, so an interrupted export can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.defaults import DeploymentDefaults


FUNDUS_FIELDS = ("21015", "21016")
OCT_FIELDS = ("21017", "21018")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SLICE_NUMBER_RE = re.compile(r"_(\d+)(?=\.[^.]+$)")
COPY_BUFFER_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class ExportResult:
    field_id: str
    source: str
    source_member: str
    destination: str
    status: str
    byte_count: int
    sha256: str
    total_slices: int | str = ""
    selected_position: int | str = ""
    selected_slice_number: int | str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export UKB fundus PNGs and the middle slice of each OCT ZIP."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Read-only UKB source root.",
    )
    add_runtime_arguments(parser)
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=None,
        help="Output root.",
    )
    parser.add_argument(
        "--mode", choices=("all", "fundus", "oct"), default="all"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent OCT archives. Fundus archives remain sequential.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many items per field (for validation).",
    )
    parser.add_argument(
        "--progress-every", type=int, default=500, help="Progress reporting interval."
    )
    args = parser.parse_args()
    defaults = DeploymentDefaults.load(args.project_root)
    args.source_root = args.source_root or defaults.ukb_source_root
    if args.destination_root is None:
        args.destination_root = resolve_runtime_arguments(args).dataset_root
    return args


def safe_member_path(member_name: str) -> PurePosixPath:
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe ZIP member path: {member_name!r}")
    return path


def copy_and_hash(source: BinaryIO, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with temporary.open("xb") as output:
            while True:
                chunk = source.read(COPY_BUFFER_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return byte_count, digest.hexdigest()


def existing_result(
    field_id: str,
    source: Path,
    member: str,
    destination: Path,
    **metadata: int,
) -> ExportResult:
    return ExportResult(
        field_id=field_id,
        source=str(source),
        source_member=member,
        destination=str(destination),
        status="existing",
        byte_count=destination.stat().st_size,
        sha256="",
        **metadata,
    )


def export_fundus_field(
    source_root: Path, destination_root: Path, field_id: str, limit: int | None
) -> Iterable[ExportResult]:
    archive = source_root / f"{field_id}.zip"
    with zipfile.ZipFile(archive) as bundle:
        members = [
            info
            for info in bundle.infolist()
            if not info.is_dir()
            and PurePosixPath(info.filename).suffix.lower() in IMAGE_SUFFIXES
        ]
        if limit is not None:
            members = members[:limit]
        for info in members:
            try:
                relative = safe_member_path(info.filename)
                destination = destination_root.joinpath(*relative.parts)
                if destination.is_file() and destination.stat().st_size == info.file_size:
                    yield existing_result(
                        field_id, archive, info.filename, destination
                    )
                    continue
                with bundle.open(info) as source:
                    byte_count, digest = copy_and_hash(source, destination)
                yield ExportResult(
                    field_id=field_id,
                    source=str(archive),
                    source_member=info.filename,
                    destination=str(destination),
                    status="exported",
                    byte_count=byte_count,
                    sha256=digest,
                )
            except Exception as exc:  # Continue the dataset export and record failures.
                yield ExportResult(
                    field_id=field_id,
                    source=str(archive),
                    source_member=info.filename,
                    destination="",
                    status="error",
                    byte_count=0,
                    sha256="",
                    error=f"{type(exc).__name__}: {exc}",
                )


def slice_number(member_name: str) -> int:
    match = SLICE_NUMBER_RE.search(PurePosixPath(member_name).name)
    if not match:
        raise ValueError(f"Cannot parse OCT slice number: {member_name!r}")
    return int(match.group(1))


def export_oct_archive(
    source_root: Path, destination_root: Path, archive: Path, field_id: str
) -> ExportResult:
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = [
                info
                for info in bundle.infolist()
                if not info.is_dir()
                and PurePosixPath(info.filename).suffix.lower() in IMAGE_SUFFIXES
            ]
            if not members:
                raise ValueError("Archive contains no supported image slices")
            members.sort(key=lambda info: (slice_number(info.filename), info.filename))

            # For an even N, choose the upper middle: position N/2 + 1 (65 of 128).
            selected_index = len(members) // 2
            selected = members[selected_index]
            selected_number = slice_number(selected.filename)
            relative_archive = archive.relative_to(source_root)
            member_name = safe_member_path(selected.filename).name
            destination = (
                destination_root
                / relative_archive.parent
                / archive.stem
                / member_name
            )
            metadata = {
                "total_slices": len(members),
                "selected_position": selected_index + 1,
                "selected_slice_number": selected_number,
            }
            if destination.is_file() and destination.stat().st_size == selected.file_size:
                return existing_result(
                    field_id, archive, selected.filename, destination, **metadata
                )
            with bundle.open(selected) as source:
                byte_count, digest = copy_and_hash(source, destination)
            return ExportResult(
                field_id=field_id,
                source=str(archive),
                source_member=selected.filename,
                destination=str(destination),
                status="exported",
                byte_count=byte_count,
                sha256=digest,
                **metadata,
            )
    except Exception as exc:
        return ExportResult(
            field_id=field_id,
            source=str(archive),
            source_member="",
            destination="",
            status="error",
            byte_count=0,
            sha256="",
            error=f"{type(exc).__name__}: {exc}",
        )


def oct_archives(source_root: Path, field_id: str, limit: int | None) -> list[Path]:
    archives = sorted((source_root / field_id).rglob("*.zip"))
    return archives if limit is None else archives[:limit]


def write_result(writer: csv.DictWriter, handle: BinaryIO, result: ExportResult) -> None:
    writer.writerow(result.__dict__)
    handle.flush()


def report_progress(label: str, completed: int, total: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 0.001)
    print(
        f"[{label}] {completed:,}/{total:,} ({completed / total:.1%}), "
        f"{completed / elapsed:.1f} items/s",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    source_root = args.source_root.resolve()
    destination_root = args.destination_root.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"Source root does not exist: {source_root}")
    mount = subprocess.run(
        ["findmnt", "-no", "OPTIONS", "--target", str(source_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    if "ro" not in mount.stdout.strip().split(","):
        raise SystemExit("Refusing export because the source volume is not read-only")
    destination_root.mkdir(parents=True, exist_ok=True)

    usage = shutil.disk_usage(destination_root)
    print(f"Source:      {source_root}")
    print(f"Destination: {destination_root}")
    print(f"Free space:  {usage.free / 1024**3:.1f} GiB")
    print("Middle rule: upper middle (position N/2 + 1 for even N)")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    manifest_path = destination_root / f"ophthalmology_export_{run_id}.csv"
    fields = list(ExportResult.__dataclass_fields__)
    counts = {"exported": 0, "existing": 0, "error": 0}

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=fields)
        writer.writeheader()

        if args.mode in ("all", "fundus"):
            for field_id in FUNDUS_FIELDS:
                started = time.monotonic()
                total = 0
                with zipfile.ZipFile(source_root / f"{field_id}.zip") as bundle:
                    total = sum(
                        1
                        for info in bundle.infolist()
                        if not info.is_dir()
                        and PurePosixPath(info.filename).suffix.lower()
                        in IMAGE_SUFFIXES
                    )
                if args.limit is not None:
                    total = min(total, args.limit)
                for completed, result in enumerate(
                    export_fundus_field(
                        source_root, destination_root, field_id, args.limit
                    ),
                    start=1,
                ):
                    write_result(writer, manifest, result)
                    counts[result.status] += 1
                    if completed % args.progress_every == 0 or completed == total:
                        report_progress(field_id, completed, total, started)

        if args.mode in ("all", "oct"):
            for field_id in OCT_FIELDS:
                archives = oct_archives(source_root, field_id, args.limit)
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(
                            export_oct_archive,
                            source_root,
                            destination_root,
                            archive,
                            field_id,
                        ): archive
                        for archive in archives
                    }
                    for completed, future in enumerate(as_completed(futures), start=1):
                        result = future.result()
                        write_result(writer, manifest, result)
                        counts[result.status] += 1
                        if (
                            completed % args.progress_every == 0
                            or completed == len(archives)
                        ):
                            report_progress(
                                field_id, completed, len(archives), started
                            )

    print(f"Manifest: {manifest_path}")
    print(
        "Summary: "
        + ", ".join(f"{key}={value:,}" for key, value in counts.items())
    )
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
