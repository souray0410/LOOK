#!/usr/bin/env python3
"""Run short validation-only baselines across prespecified UKB task candidates."""

from __future__ import annotations

import argparse
import json
import os

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.distributed import parse_gpu_devices


DEFAULT_PROFILES = [
    "glaucoma_high_confidence",
    "glaucoma_all_evidence",
    "any_target_eye_disease_high_confidence",
    "any_target_eye_disease",
    "glaucoma_objective_only",
    "diabetic_eye_disease_all_evidence",
    "macular_degeneration_all_evidence",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--fusion-positions", nargs="+", default=["feature"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    gpu_devices = parse_gpu_devices(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))
    paths = resolve_runtime_arguments(args)
    verification = (
        paths.dataset_root / "cohorts" / "task_scout" / "task_bank_verification.json"
    )
    if args.execute:
        if not verification.is_file():
            raise RuntimeError("Run Steps 21-22 before the task scout")
        if json.loads(verification.read_text())["status"] != "PASS":
            raise RuntimeError("The task cohort bank has not passed verification")

    from look_core.task_scout import run_task_scout

    result = run_task_scout(
        paths,
        profile_ids=args.profiles,
        fusion_positions=args.fusion_positions,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        gpu_devices=gpu_devices,
        execute=args.execute,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
