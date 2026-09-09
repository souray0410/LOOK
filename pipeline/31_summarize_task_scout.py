#!/usr/bin/env python3
"""Summarize validation-only UKB task-scout evidence without selecting a winner."""

from __future__ import annotations

import argparse
import json

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.task_scout import summarize_task_scout


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--scout-id")
    args = parser.parse_args()
    paths = resolve_runtime_arguments(args)
    task_root = paths.runs_root / "task_scout"
    if args.scout_id:
        scout_root = task_root / f"scout__{args.scout_id}"
    else:
        candidates = sorted(
            task_root.glob("scout__*/progress.json"),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(f"No task scout found under {task_root}")
        scout_root = candidates[-1].parent
    report = summarize_task_scout(
        scout_root,
        paths.dataset_root / "cohorts" / "task_scout",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed": report["completed_configurations"],
                "planned": report["planned_configurations"],
                "json": str(scout_root / "task_usability_summary.json"),
                "markdown": str(scout_root / "task_usability_summary.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
