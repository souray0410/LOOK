from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, Literal

from look.runtime.state import atomic_write_json, file_sha256, stable_hash, utc_now


SelectionMode = Literal["delete_matches", "keep_matches"]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def plan_keyword_filter(
    root: Path,
    keywords: Iterable[str],
    *,
    mode: SelectionMode,
    recursive: bool = True,
) -> dict[str, object]:
    """Create a content-addressed allowlist; this function never changes files."""

    root = Path(root).resolve()
    words = tuple(dict.fromkeys(word for word in keywords if word))
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not words:
        raise ValueError("At least one non-empty keyword is required")
    if mode not in {"delete_matches", "keep_matches"}:
        raise ValueError(f"Unsupported mode: {mode}")

    files = root.rglob("*") if recursive else root.glob("*/*")
    selected: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    inspected = 0
    for path in sorted(files):
        if not path.is_file() or path.is_symlink():
            continue
        inspected += 1
        matches = [word for word in words if word in path.name]
        counts.update(matches)
        should_select = bool(matches) if mode == "delete_matches" else not matches
        if should_select:
            selected.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "matched_keywords": matches,
                }
            )
    identity = {"root": str(root), "keywords": words, "mode": mode, "recursive": recursive, "files": selected}
    return {
        **identity,
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "inspected_files": inspected,
        "selected_files": len(selected),
        "keyword_counts": dict(counts),
        "plan_id": stable_hash(identity),
    }


def write_filter_plan(plan: dict[str, object], destination: Path) -> None:
    atomic_write_json(plan, Path(destination))


def execute_filter_plan(plan_path: Path, quarantine_root: Path, *, execute: bool = False) -> dict[str, object]:
    """Validate every allowlisted file and move it to quarantine when explicitly enabled."""

    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    root = Path(plan["root"]).resolve()
    quarantine_root = Path(quarantine_root).resolve()
    entries = list(plan["files"])
    identity = {
        "root": str(root),
        "keywords": tuple(plan["keywords"]),
        "mode": plan["mode"],
        "recursive": plan["recursive"],
        "files": entries,
    }
    if stable_hash(identity) != plan["plan_id"]:
        raise ValueError("Filter plan identity hash does not match")

    verified: list[tuple[Path, dict[str, object]]] = []
    for entry in entries:
        path = root / str(entry["relative_path"])
        if not _inside(path, root) or path.is_symlink() or not path.is_file():
            raise ValueError(f"Unsafe or missing allowlisted path: {path}")
        if path.stat().st_size != int(entry["bytes"]) or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Allowlisted file changed after planning: {path}")
        verified.append((path, entry))
    if not execute:
        return {"status": "dry_run", "verified_files": len(verified), "plan_id": plan["plan_id"]}

    moved: list[str] = []
    destination_root = quarantine_root / str(plan["plan_id"])
    for source, entry in verified:
        destination = destination_root / str(entry["relative_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.move(str(source), destination)
        moved.append(str(entry["relative_path"]))
    result = {"status": "completed", "plan_id": plan["plan_id"], "moved_files": moved, "completed_at_utc": utc_now()}
    atomic_write_json(result, destination_root / "execution.json")
    return result
