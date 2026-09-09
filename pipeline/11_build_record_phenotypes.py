#!/usr/bin/env python3
"""Step 11: construct temporally defined record-derived phenotype candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.phenotype import build_candidate_table


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--phenotype", type=Path, help="Selected matched phenotype CSV from Step 10.")
    parser.add_argument("--paired-visits", type=Path, help="Earliest complete bilateral visit manifest.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunksize", type=int, default=500)
    args = parser.parse_args()
    paths = resolve_runtime_arguments(args)
    phenotype = args.phenotype or paths.dataset_root / "phenotypes/matched_fields/ukb670300_matched.csv"
    paired_visits = args.paired_visits or paths.dataset_root / "paired_eye_manifest.csv"
    output = args.output or paths.dataset_root / "phenotypes/record_phenotype_candidates.csv"
    for source in (phenotype, paired_visits):
        if not source.is_file():
            raise SystemExit(f"Missing Step 11 input: {source}")
    report = build_candidate_table(phenotype, paired_visits, output, chunksize=args.chunksize)
    print(json.dumps(report, indent=2))
    print(f"Candidates: {output}")


if __name__ == "__main__":
    main()
