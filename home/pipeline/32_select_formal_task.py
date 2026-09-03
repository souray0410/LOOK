#!/usr/bin/env python3
"""Select one validated task-scout profile for formal baseline qualification."""

from __future__ import annotations

import argparse
import json

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.task_selection import select_formal_task


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--profile")
    parser.add_argument("--scout-id")
    parser.add_argument("--reviewer-note", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    paths = resolve_runtime_arguments(args)
    project = json.loads((paths.project_root / "project.json").read_text())
    profile = args.profile or project["selected_task_profile"]
    result = select_formal_task(
        paths,
        profile_id=profile,
        scout_id=args.scout_id,
        reviewer_note=args.reviewer_note,
        execute=args.execute,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
