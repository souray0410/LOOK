#!/usr/bin/env python3
"""Run both frozen-classifier filling arms end to end on a tiny train subset."""

from __future__ import annotations

import argparse
import os


def run_arm(filling_strategy: str, paths, torch, gpu_devices):
    from look_core.config import ExperimentConfig, ExperimentSelection
    from look_core.pipeline import ExperimentRunner, PipelineOptions

    config = ExperimentConfig(
        data_root=paths.dataset_root,
        output_root=paths.runs_root / "smoke" / "pipeline",
        cache_root=paths.cache_root,
        epochs=1,
        patience=1,
        effective_batch_size=4,
        micro_batch_size=2,
        num_workers=0,
        warmup_epochs=0,
        gan_epochs=1,
        gan_patience=1,
        gan_batch_size=2,
        gan_num_workers=0,
        gan_base_channels=4,
        world_size=len(gpu_devices),
        missing_ratios=[0.5],
        downsample_factors=[16],
        latent_dims=[2],
        max_pca_rank=2,
        alpha_grid=[0.0, 1.0],
    )
    selection = ExperimentSelection(
        fusion_position="feature",
        seed=3407,
        filling_strategy=filling_strategy,
        run_label="integration_smoke",
    )
    options = PipelineOptions(
        fit_look=True,
        evaluate_random_missing=True,
        phase="validation",
        resume=True,
        smoke_limit=8,
        bootstrap_iterations=100,
        gpu_devices=gpu_devices,
    )
    return ExperimentRunner(config, selection, options, torch.device("cuda:0")).run()


def main() -> None:
    from look_core.cli import add_runtime_arguments, resolve_runtime_arguments

    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    from look_core.distributed import parse_gpu_devices
    gpu_devices = parse_gpu_devices(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the pipeline integration smoke test")
    paths = resolve_runtime_arguments(args)
    mean_result = run_arm("normalized_mean", paths, torch, gpu_devices)
    gan_result = run_arm("paired_cgan", paths, torch, gpu_devices)
    assert mean_result["status"] == gan_result["status"] == "complete"
    assert mean_result["checkpoint"]["backbone_id"] == gan_result["checkpoint"]["backbone_id"]
    assert mean_result["checkpoint"]["sha256"] == gan_result["checkpoint"]["sha256"]
    assert mean_result["checkpoint"]["frozen_after_loading"] is True
    assert gan_result["filling"]["joint_training"] is False
    assert mean_result["look"]["enabled"] and gan_result["look"]["enabled"]
    assert mean_result["test"] == gan_result["test"] == {}
    print("PIPELINE SMOKE PASS: shared frozen classifier, both fills, LOOK, sealed test")


if __name__ == "__main__":
    main()
