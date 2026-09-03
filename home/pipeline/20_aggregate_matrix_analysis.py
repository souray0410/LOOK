#!/usr/bin/env python3
"""Aggregate selected LOOK matrix diagnostics from completed experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    from look_core.cli import add_runtime_arguments, resolve_runtime_arguments

    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    from look_core.matrix_analysis import aggregate_matrix_analyses

    paths = resolve_runtime_arguments(args)
    output = args.output or paths.runs_root / "look_matrix_analysis_all_experiments.csv"
    count = aggregate_matrix_analyses(paths.runs_root, output)
    print(f"Aggregated {count} selected LOOK matrices into {output}")


if __name__ == "__main__":
    main()
