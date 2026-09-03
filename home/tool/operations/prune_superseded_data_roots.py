#!/usr/bin/env python3
"""Delete explicitly named superseded runtime roots while preserving one release."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--keep-release", required=True)
    parser.add_argument("--delete-release", action="append", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    family = args.family_root.resolve()
    keep = (family / args.keep_release).resolve()
    if not keep.is_dir() or keep.parent != family:
        raise RuntimeError(f"Declared keep release is invalid: {keep}")
    if args.keep_release in args.delete_release:
        raise ValueError("The keep release cannot be deleted")
    targets = []
    for release in args.delete_release:
        path = (family / release).resolve()
        if path.parent != family or path == keep:
            raise RuntimeError(f"Deletion target escapes the family root: {path}")
        if path.is_symlink():
            raise RuntimeError(f"Symlink deletion is refused: {path}")
        targets.append(path)
    plan = {
        "family_root": str(family),
        "keep": str(keep),
        "delete": [str(path) for path in targets if path.exists()],
        "missing": [str(path) for path in targets if not path.exists()],
        "execute": args.execute,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    for path in targets:
        if path.exists():
            shutil.rmtree(path)
    remaining = sorted(path.name for path in family.iterdir() if path.is_dir())
    if remaining != [args.keep_release]:
        raise RuntimeError(f"Unexpected runtime roots remain: {remaining}")
    print(json.dumps({"status": "PASS", "remaining": remaining}, indent=2))


if __name__ == "__main__":
    main()
