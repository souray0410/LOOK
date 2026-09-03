#!/usr/bin/env python3
"""Freeze a reviewed baseline candidate before the formal LOOK study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from look_core.study_grid import freeze_baseline_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reviewer-note", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "status": "dry_run",
            "candidate": str(args.candidate.resolve()),
            "action": "freeze reviewed baseline for formal LOOK execution",
        }, indent=2))
        return
    result = freeze_baseline_candidate(
        args.candidate,
        reviewer_note=args.reviewer_note,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
