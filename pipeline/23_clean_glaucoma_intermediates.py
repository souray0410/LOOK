#!/usr/bin/env python3
"""Step 23: remove only stale partial artifacts after cohort verification."""

from __future__ import annotations

import argparse
import json

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    paths = resolve_runtime_arguments(args)
    verification = paths.cohort_root / "verification.json"
    if not verification.is_file():
        raise SystemExit("Run Step 22 before cleanup")
    report = json.loads(verification.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise SystemExit("Cohort verification is not PASS")

    roots = (paths.dataset_root, paths.cache_root / "partial")
    targets = sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*.partial")
        if path.is_file()
    )
    print(json.dumps({"execute": args.execute, "targets": [str(path) for path in targets]}, indent=2))
    if not args.execute:
        print("Dry run only. Re-run with --execute after reviewing the list.")
        return
    for path in targets:
        path.unlink()
    print(json.dumps({"status": "PASS", "removed": len(targets)}, indent=2))


if __name__ == "__main__":
    main()
