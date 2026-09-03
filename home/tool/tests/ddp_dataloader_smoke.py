"""Exercise spawned UKB DataLoader workers after CUDA/DDP initialization."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist

from MHD_Project.MHD_Utils_V4 import (
    destroy_mhd_distributed,
    initialize_mhd_distributed,
)
from look_core.data import UKBPairedEyeDataset, make_loader
from look_core.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=4096)
    args = parser.parse_args()
    paths = ProjectPaths.load(args.project_root)
    context = initialize_mhd_distributed()
    try:
        dataset = UKBPairedEyeDataset(
            paths.labels_csv,
            paths.image_root,
            "train",
            image_size=224,
            augment=True,
            limit=args.limit,
            preprocess_cache_root=paths.preprocess_cache_root,
        )
        loader = make_loader(
            dataset,
            batch_size=128,
            num_workers=args.workers,
            train=True,
            seed=3407,
            rank=context.rank,
            world_size=context.world_size,
        )
        expected = len(loader) * 128 * context.world_size
        for epoch in range(args.epochs):
            dataset.set_epoch(epoch)
            loader.sampler.set_epoch(epoch)
            local_count = torch.zeros((), dtype=torch.int64, device=context.device)
            checksum = torch.zeros((), dtype=torch.float64, device=context.device)
            for batch in loader:
                oct_tensor = batch["oct"].to(context.device, non_blocking=True)
                cfp_tensor = batch["cfp"].to(context.device, non_blocking=True)
                local_count += len(oct_tensor)
                checksum += oct_tensor.double().mean() + cfp_tensor.double().mean()
            dist.all_reduce(local_count)
            dist.all_reduce(checksum)
            if int(local_count.item()) != expected:
                raise RuntimeError(
                    f"Epoch {epoch} gathered {local_count.item()} rows; expected {expected}"
                )
            if context.is_main:
                print(
                    f"epoch={epoch + 1} rows={local_count.item()} "
                    f"checksum={checksum.item():.8f}",
                    flush=True,
                )
        if context.is_main:
            print("DDP DATALOADER SPAWN SMOKE PASS", flush=True)
    finally:
        destroy_mhd_distributed()


if __name__ == "__main__":
    main()
