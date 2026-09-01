#!/usr/bin/env python3
"""Run the shared LOOK study grid sequentially on one visible GPU."""

from __future__ import annotations

import argparse
import json
import os

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--phase", choices=("validation", "test"), default="validation")
    parser.add_argument("--confirmation", default="")
    parser.add_argument(
        "--fusion-positions", nargs="+",
        default=["input", "stem", "layer1", "layer2", "layer3", "layer4", "feature"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    parser.add_argument(
        "--filling-strategies", nargs="+", choices=("normalized_mean", "paired_cgan"),
        default=["normalized_mean", "paired_cgan"],
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--skip-full-path-audit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch

    from look_core.reproducibility import enforce_single_gpu
    from look_core.study_grid import StudyGrid, run_study_grid

    enforce_single_gpu()
    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal study execution")
    paths = resolve_runtime_arguments(args)
    grid = StudyGrid(
        fusion_positions=args.fusion_positions,
        seeds=args.seeds,
        filling_strategies=args.filling_strategies,
    )
    result = run_study_grid(
        grid,
        paths,
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        execute=not args.dry_run,
        phase=args.phase,
        confirmation=args.confirmation,
        check_all_image_paths=not args.skip_full_path_audit,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
