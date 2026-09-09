#!/usr/bin/env python3
"""Step 12: build balanced, natural, and incident participant cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.cohort import DEFAULT_SAMPLING_SEED, derive_four_class_cohorts


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--candidate-csv", type=Path)
    parser.add_argument("--sampling-seed", default=DEFAULT_SAMPLING_SEED)
    args = parser.parse_args()
    paths = resolve_runtime_arguments(args)
    source = args.candidate_csv or paths.dataset_root / "phenotypes/record_phenotype_candidates.csv"
    report = derive_four_class_cohorts(source, paths.cohort_root, sampling_seed=args.sampling_seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
