from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from look.runtime.state import file_sha256


def build_file_manifest(paths: Iterable[Path]) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}
    for path in map(Path, paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest[str(path)] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return manifest


def validate_file_manifest(manifest: Mapping[str, Mapping[str, object]]) -> list[str]:
    errors: list[str] = []
    for raw_path, expected in manifest.items():
        path = Path(raw_path)
        if not path.is_file():
            errors.append(f"missing:{path}")
            continue
        if path.stat().st_size != int(expected["bytes"]):
            errors.append(f"size:{path}")
            continue
        if file_sha256(path) != expected["sha256"]:
            errors.append(f"sha256:{path}")
    return errors


def missing_or_invalid(paths: Iterable[Path], manifest: Mapping[str, Mapping[str, object]] | None = None) -> list[Path]:
    candidates = [Path(path) for path in paths]
    if manifest is None:
        return [path for path in candidates if not path.is_file()]
    invalid = {error.split(":", 1)[1] for error in validate_file_manifest(manifest)}
    return [path for path in candidates if str(path) in invalid]
