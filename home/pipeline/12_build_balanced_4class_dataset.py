#!/usr/bin/env python3
"""Build balanced and natural four-class weak-reference cohort manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.cohort import DEFAULT_SAMPLING_SEED, derive_four_class_cohorts


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--source-labels-csv", type=Path)
    parser.add_argument("--sampling-seed", default=DEFAULT_SAMPLING_SEED)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    paths = resolve_runtime_arguments(args)
    source = (args.source_labels_csv or paths.image_root / "reference_labels.csv").resolve()
    preview = {
        "source_labels_csv": str(source),
        "cohort_root": str(paths.cohort_root),
        "sampling_seed": args.sampling_seed,
        "execute": args.execute,
    }
    print(json.dumps(preview, indent=2))
    if not args.execute:
        print("Dry run only; pass --execute to create cohort manifests.")
        return
    print(json.dumps(derive_four_class_cohorts(source, paths.cohort_root, sampling_seed=args.sampling_seed), indent=2))


if __name__ == "__main__":
    main()
