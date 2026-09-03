#!/usr/bin/env python3
"""Run, freeze, or test the shared LOOK study grid on a selected GPU list."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.distributed import parse_gpu_devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--gpus", default="0,1", help="Physical GPU list, e.g. 0 or 0,1.")
    parser.add_argument(
        "--mode",
        choices=("full-study", "baseline-selection"),
        default="full-study",
    )
    parser.add_argument("--phase", choices=("validation", "freeze", "test"), default="validation")
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--baseline-selection-manifest", type=Path)
    parser.add_argument(
        "--fusion-positions", nargs="+",
        default=None,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
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
    gpu_devices = parse_gpu_devices(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))

    import torch

    from look_core.study_grid import (
        StudyGrid,
        freeze_study_grid,
        run_baseline_selection,
        run_study_grid,
        study_grid_from_baseline_selection,
    )

    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal study execution")
    paths = resolve_runtime_arguments(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.mode == "baseline-selection":
        if args.phase != "validation":
            raise ValueError("baseline-selection is validation-only")
        result = run_baseline_selection(
            paths,
            device,
            execute=not args.dry_run,
            check_all_image_paths=not args.skip_full_path_audit,
            gpu_devices=gpu_devices,
        )
        print(json.dumps(result, indent=2))
        return
    else:
        baseline_manifest = args.baseline_selection_manifest
        if baseline_manifest is None and not args.dry_run:
            candidates = sorted(
                (paths.runs_root / "baseline_selection").glob(
                    "baseline_selection__*.json"
                )
            )
            if len(candidates) != 1:
                raise ValueError(
                    "Formal full-study execution requires exactly one frozen baseline "
                    "selection manifest or --baseline-selection-manifest"
                )
            baseline_manifest = candidates[0]
        if baseline_manifest is not None:
            grid = study_grid_from_baseline_selection(
                baseline_manifest,
                filling_strategies=args.filling_strategies,
            )
            if args.fusion_positions is not None and args.fusion_positions != grid.fusion_positions:
                raise ValueError("--fusion-positions conflicts with the frozen baseline")
            if args.seeds is not None and args.seeds != grid.seeds:
                raise ValueError("--seeds conflicts with the frozen baseline")
        else:
            grid = StudyGrid(
                fusion_positions=args.fusion_positions or [
                    "input", "stem", "layer1", "layer2", "layer3", "layer4", "feature"
                ],
                seeds=args.seeds or [3407, 3408, 3409],
                filling_strategies=args.filling_strategies,
            )
    if args.phase == "freeze":
        if args.dry_run:
            raise ValueError("freeze does not support dry-run; complete validation first")
        result = freeze_study_grid(grid, paths, device, gpu_devices=gpu_devices)
    else:
        result = run_study_grid(
            grid, paths, device, execute=not args.dry_run, phase=args.phase,
            frozen_manifest=args.frozen_manifest,
            check_all_image_paths=(
                not args.skip_full_path_audit and args.mode == "full-study"
            ),
            bootstrap_iterations=args.bootstrap_iterations,
            gpu_devices=gpu_devices,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
