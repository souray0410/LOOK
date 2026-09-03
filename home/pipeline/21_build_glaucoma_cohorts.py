#!/usr/bin/env python3
"""Step 21: build the primary glaucoma cohort and validation-only task bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.cohort import DEFAULT_SAMPLING_SEED, derive_glaucoma_cohorts
from look_core.task_scout import TASK_SCOUT_SEED, build_task_cohort_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--candidate-csv", type=Path)
    parser.add_argument("--sampling-seed", default=DEFAULT_SAMPLING_SEED)
    parser.add_argument("--task-scout-seed", default=TASK_SCOUT_SEED)
    args = parser.parse_args()
    paths = resolve_runtime_arguments(args)
    source = (
        args.candidate_csv
        or paths.dataset_root / "phenotypes" / "record_phenotype_candidates.csv"
    )
    glaucoma = derive_glaucoma_cohorts(
        source,
        paths.cohort_root,
        sampling_seed=args.sampling_seed,
    )
    task_bank = build_task_cohort_bank(
        source,
        paths.dataset_root / "cohorts" / "task_scout",
        seed=args.task_scout_seed,
    )
    print(json.dumps({"glaucoma": glaucoma, "task_bank": task_bank}, indent=2))


if __name__ == "__main__":
    main()
