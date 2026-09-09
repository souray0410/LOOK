from __future__ import annotations

import argparse
import json
from pathlib import Path

from V4.MHD_Utils_V4 import destroy_mhd_distributed, initialize_mhd_distributed

from .config import ExperimentConfig
from .data import UKBBilateralVisitDataset, make_loader
from .graph import build_resnet50_mhd_graph
from .gan import make_gan_loaders, train_paired_cgan_direction_ddp
from .reproducibility import seed_everything
from .train import train_complete_model_ddp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--stage", choices=("classifier", "gan"), required=True)
    return parser.parse_args()


def run_classifier(payload: dict, context) -> None:
    config = ExperimentConfig.from_dict(payload["config"])
    if config.world_size != context.world_size:
        raise RuntimeError(f"Configured world_size={config.world_size}, launched={context.world_size}")
    seed = int(payload["seed"])
    seed_everything(seed)
    kwargs = {
        "labels_csv": config.labels_csv,
        "data_root": config.image_root,
        "image_size": config.image_size,
        "limit": payload.get("smoke_limit"),
        "base_seed": seed,
        "preprocess_cache_root": config.preprocess_cache_root,
    }
    train_dataset = UKBBilateralVisitDataset(split="train", augment=True, **kwargs)
    validation_dataset = UKBBilateralVisitDataset(split="validation", augment=False, **kwargs)
    train_loader = make_loader(
        train_dataset, config.per_device_micro_batch_size, config.num_workers, True,
        seed, config.sampling_strategy, context.rank, context.world_size,
    )
    validation_loader = make_loader(
        validation_dataset, config.per_device_micro_batch_size, config.num_workers, False,
        seed, config.sampling_strategy, context.rank, context.world_size,
    )
    run_dir = Path(payload["run_dir"])
    graph = build_resnet50_mhd_graph(
        payload["fusion_position"], config.num_classes, config.per_device_micro_batch_size,
        config.image_size,
        "cpu",
        pretrained=not (run_dir / "last").is_dir()
        and not (run_dir / "best").is_dir()
        and not (run_dir / "best.pt").is_file(),
        classifier_dropout=config.classifier_dropout,
        label_smoothing=config.label_smoothing,
    )
    seed_everything(seed + context.rank)
    if not (run_dir / "training_complete.json").is_file():
        train_complete_model_ddp(
            graph,
            train_loader,
            validation_loader,
            config,
            run_dir,
            context,
        )


def run_gan(payload: dict, context) -> None:
    config = ExperimentConfig.from_dict(payload["config"])
    if config.world_size != context.world_size:
        raise RuntimeError(f"Configured world_size={config.world_size}, launched={context.world_size}")
    seed = int(payload["seed"])
    direction_seed = int(payload["direction_seed"])
    seed_everything(direction_seed)
    kwargs = {
        "labels_csv": config.labels_csv,
        "data_root": config.image_root,
        "image_size": config.image_size,
        "limit": payload.get("smoke_limit"),
        "base_seed": seed,
        "preprocess_cache_root": config.preprocess_cache_root,
    }
    train_aug = UKBBilateralVisitDataset(split="train", augment=True, **kwargs)
    train_clean = UKBBilateralVisitDataset(split="train", augment=False, **kwargs)
    train_loader, validation_loader, _ = make_gan_loaders(
        train_aug, train_clean, config, seed, int(payload["direction_seed"]),
        context.rank, context.world_size,
    )
    train_paired_cgan_direction_ddp(
        payload["source"], payload["target"], train_loader, validation_loader,
        config, Path(payload["output_dir"]), context,
        rank_seed=direction_seed + context.rank,
    )


def main() -> None:
    args = parse_args()
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    context = initialize_mhd_distributed()
    try:
        if args.stage == "classifier":
            run_classifier(payload, context)
        else:
            run_gan(payload, context)
    finally:
        destroy_mhd_distributed()


if __name__ == "__main__":
    main()
