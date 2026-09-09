from __future__ import annotations

from .state import durable_replace

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm.auto import tqdm

from V4.MHD_Utils_V4 import (
    MHD_DistributedContext,
    mhd_barrier,
)

from .data import DistributedEvalSampler, IMAGENET_MEAN, IMAGENET_STD
from .distributed import assert_module_state_identical
from .monitoring import TrainingMonitor
from .reproducibility import seed_everything, write_json_atomic


class DownBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, normalize: bool = True) -> None:
        super().__init__()
        layers = [nn.Conv2d(input_channels, output_channels, 4, 2, 1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(output_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.block(x), skip], dim=1)


class PairedUNetGenerator(nn.Module):
    """Pix2pix-style 2D generator adapted to 224x224 paired cross-modal images."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.down1 = DownBlock(3, c, normalize=False)
        self.down2 = DownBlock(c, c * 2)
        self.down3 = DownBlock(c * 2, c * 4)
        self.down4 = DownBlock(c * 4, c * 8)
        self.down5 = DownBlock(c * 8, c * 8)
        self.up4 = UpBlock(c * 8, c * 8, dropout=0.5)
        self.up3 = UpBlock(c * 16, c * 4, dropout=0.5)
        self.up2 = UpBlock(c * 8, c * 2)
        self.up1 = UpBlock(c * 4, c)
        self.output = nn.Sequential(nn.ConvTranspose2d(c * 2, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u4 = self.up4(d5, d4)
        u3 = self.up3(u4, d3)
        u2 = self.up2(u3, d2)
        u1 = self.up1(u2, d1)
        return self.output(u1)


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.model = nn.Sequential(
            nn.Conv2d(6, c, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(c * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(c * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(c * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 8, 1, 4, 1, 1),
        )

    def forward(self, source, target):
        return self.model(torch.cat([source, target], dim=1))


def _to_gan_range(tensor: torch.Tensor) -> torch.Tensor:
    mean = tensor.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = tensor.new_tensor(IMAGENET_STD)[None, :, None, None]
    return ((tensor * std + mean).clamp(0, 1) * 2) - 1


def _flatten_bilateral(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim != 5 or tensor.shape[1] != 2:
        raise ValueError(f"Expected [B,2,C,H,W] or [B,C,H,W], got {tuple(tensor.shape)}")
    return tensor.reshape(tensor.shape[0] * 2, *tensor.shape[2:])


def gan_internal_split_indices(frame, validation_fraction: float, seed: int):
    participants = frame["participant_id"].astype(str)
    unique_participants = sorted(participants.unique())
    if len(unique_participants) < 2:
        raise RuntimeError("At least two participants are required for the GAN internal split")
    validation_participants = set()
    scores = {}
    for participant in unique_participants:
        digest = hashlib.sha256(f"gan:{seed}:{participant}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        scores[participant] = value
        if value < validation_fraction:
            validation_participants.add(participant)
    if not validation_participants:
        validation_participants.add(min(scores, key=scores.get))
    if len(validation_participants) == len(unique_participants):
        validation_participants.remove(max(scores, key=scores.get))
    validation = [i for i, participant in enumerate(participants) if participant in validation_participants]
    training = [i for i, participant in enumerate(participants) if participant not in validation_participants]
    return training, validation


def make_gan_loaders(
    dataset_train_aug, dataset_train_clean, config, seed: int, loader_seed: int | None = None,
    rank: int = 0, world_size: int = 1,
):
    training_indices, validation_indices = gan_internal_split_indices(
        dataset_train_clean.frame, config.gan_validation_fraction, seed
    )
    training_subset = Subset(dataset_train_aug, training_indices)
    validation_subset = Subset(dataset_train_clean, validation_indices)
    training_sampler = None
    validation_sampler = None
    if world_size > 1:
        training_sampler = DistributedSampler(
            training_subset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=seed if loader_seed is None else loader_seed,
        )
        validation_sampler = DistributedEvalSampler(len(validation_subset), rank, world_size)
    generator = torch.Generator().manual_seed(seed if loader_seed is None else loader_seed)
    training = DataLoader(
        training_subset,
        batch_size=config.per_device_gan_batch_size if world_size > 1 else config.gan_batch_size,
        shuffle=training_sampler is None,
        sampler=training_sampler,
        generator=generator,
        num_workers=config.gan_num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=config.gan_num_workers > 0,
    )
    validation = DataLoader(
        validation_subset,
        batch_size=config.per_device_gan_batch_size if world_size > 1 else config.gan_batch_size,
        shuffle=False,
        sampler=validation_sampler,
        num_workers=config.gan_num_workers,
        pin_memory=True,
        persistent_workers=config.gan_num_workers > 0,
    )
    return training, validation, {
        "rule": "SHA256(gan:seed:participant_id) with deterministic non-empty fallback",
        "seed": seed,
        "validation_fraction": config.gan_validation_fraction,
        "training_rows": len(training_indices),
        "validation_rows": len(validation_indices),
        "training_participants": int(dataset_train_clean.frame.iloc[training_indices].participant_id.nunique()),
        "validation_participants": int(dataset_train_clean.frame.iloc[validation_indices].participant_id.nunique()),
    }


@torch.no_grad()
def _validation_l1(generator, loader, source_key: str, target_key: str, device):
    generator.eval()
    total, count = 0.0, 0
    for batch in loader:
        source = _to_gan_range(_flatten_bilateral(batch[source_key].to(device, non_blocking=True)))
        target = _to_gan_range(_flatten_bilateral(batch[target_key].to(device, non_blocking=True)))
        generated = generator(source)
        total += torch.nn.functional.l1_loss(generated, target, reduction="sum").item()
        count += target.numel()
    return total / max(1, count)


def train_paired_cgan_direction(
    source_key: str,
    target_key: str,
    train_loader,
    validation_loader,
    config,
    output_dir: Path,
    device: torch.device,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = PairedUNetGenerator(config.gan_base_channels).to(device)
    discriminator = PatchDiscriminator(config.gan_base_channels).to(device)
    optimizer_g = Adam(generator.parameters(), lr=config.gan_learning_rate, betas=(config.gan_beta1, 0.999))
    optimizer_d = Adam(discriminator.parameters(), lr=config.gan_learning_rate, betas=(config.gan_beta1, 0.999))
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    best_validation, stale = float("inf"), 0
    history = []
    start_epoch = 0
    last_path = output_dir / "last_training.pt"
    if last_path.is_file():
        last = torch.load(last_path, map_location=device, weights_only=False)
        generator.load_state_dict(last["generator_state_dict"])
        discriminator.load_state_dict(last["discriminator_state_dict"])
        optimizer_g.load_state_dict(last["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(last["optimizer_d_state_dict"])
        start_epoch = int(last["epoch"])
        best_validation = float(last["best_validation"])
        stale = int(last["stale"])
        history = list(last["history"])

    accumulation = config.gan_accumulation_steps
    for epoch in range(start_epoch, config.gan_epochs):
        generator.train()
        discriminator.train()
        sum_g, sum_d = 0.0, 0.0
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"cGAN {source_key}->{target_key} {epoch + 1}", leave=False)):
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            should_step = step - group_start + 1 == group_size
            source = _to_gan_range(_flatten_bilateral(batch[source_key].to(device, non_blocking=True)))
            target = _to_gan_range(_flatten_bilateral(batch[target_key].to(device, non_blocking=True)))
            generated = generator(source)

            real_logits = discriminator(source, target)
            fake_logits = discriminator(source, generated.detach())
            loss_d = 0.5 * (
                adversarial_loss(real_logits, torch.ones_like(real_logits))
                + adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
            )
            (loss_d / group_size).backward()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            fake_logits = discriminator(source, generated)
            loss_g = adversarial_loss(fake_logits, torch.ones_like(fake_logits))
            loss_g = loss_g + config.gan_lambda_l1 * l1_loss(generated, target)
            (loss_g / group_size).backward()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
            if should_step:
                optimizer_d.step()
                optimizer_g.step()
                optimizer_d.zero_grad(set_to_none=True)
                optimizer_g.zero_grad(set_to_none=True)
            sum_g += loss_g.item()
            sum_d += loss_d.item()

        validation_l1 = _validation_l1(generator, validation_loader, source_key, target_key, device)
        history.append(
            {
                "epoch": epoch + 1,
                "generator_loss": sum_g / max(1, len(train_loader)),
                "discriminator_loss": sum_d / max(1, len(train_loader)),
                "validation_l1": validation_l1,
            }
        )
        write_json_atomic(history, output_dir / "history.json")
        if validation_l1 < best_validation:
            best_validation, stale = validation_l1, 0
            temporary = output_dir / "best_generator.pt.tmp"
            torch.save(
                {
                    "source": source_key,
                    "target": target_key,
                    "base_channels": config.gan_base_channels,
                    "epoch": epoch + 1,
                    "validation_l1": validation_l1,
                    "training_config": {
                        "epochs": config.gan_epochs,
                        "patience": config.gan_patience,
                        "batch_size": config.gan_batch_size,
                        "effective_batch_size": config.gan_effective_batch_size,
                        "learning_rate": config.gan_learning_rate,
                        "beta1": config.gan_beta1,
                        "lambda_l1": config.gan_lambda_l1,
                    },
                    "generator_state_dict": generator.state_dict(),
                },
                temporary,
            )
            durable_replace(temporary, output_dir / "best_generator.pt")
        else:
            stale += 1
        temporary = output_dir / "last_training.pt.tmp"
        torch.save(
            {
                "epoch": epoch + 1,
                "best_validation": best_validation,
                "stale": stale,
                "history": history,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
            },
            temporary,
        )
        durable_replace(temporary, last_path)
        if stale >= config.gan_patience:
            break
    write_json_atomic(
        {"status": "complete", "epochs_completed": len(history), "best_validation_l1": best_validation},
        output_dir / "training_complete.json",
    )
    return load_generator(output_dir / "best_generator.pt", device), history


def load_generator(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    generator = PairedUNetGenerator(checkpoint["base_channels"]).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    return generator.eval()


@torch.no_grad()
def _validation_l1_ddp(generator, loader, source_key, target_key, context):
    generator.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=context.device)
    for batch in loader:
        source = _to_gan_range(_flatten_bilateral(batch[source_key].to(context.device, non_blocking=True)))
        target = _to_gan_range(_flatten_bilateral(batch[target_key].to(context.device, non_blocking=True)))
        generated = generator(source)
        totals[0] += torch.nn.functional.l1_loss(generated, target, reduction="sum").double()
        totals[1] += target.numel()
    if context.distributed:
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    return float((totals[0] / totals[1].clamp_min(1)).item())


def train_paired_cgan_direction_ddp(
    source_key: str,
    target_key: str,
    train_loader,
    validation_loader,
    config,
    output_dir: Path,
    context: MHD_DistributedContext,
    rank_seed: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    generator_raw = PairedUNetGenerator(config.gan_base_channels)
    discriminator_raw = PatchDiscriminator(config.gan_base_channels)
    best_validation, stale, history, start_epoch = float("inf"), 0, [], 0
    last_path = output_dir / "last_training.pt"
    last = None
    if last_path.is_file():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        generator_raw.load_state_dict(last["generator_state_dict"])
        discriminator_raw.load_state_dict(last["discriminator_state_dict"])
        start_epoch = int(last["epoch"])
        best_validation = float(last["best_validation"])
        stale = int(last["stale"])
        history = list(last["history"])
    generator_state_sha256 = assert_module_state_identical(generator_raw, context)
    discriminator_state_sha256 = assert_module_state_identical(discriminator_raw, context)
    generator_raw.to(context.device)
    discriminator_raw.to(context.device)
    optimizer_g = Adam(generator_raw.parameters(), lr=config.gan_learning_rate, betas=(config.gan_beta1, 0.999))
    optimizer_d = Adam(discriminator_raw.parameters(), lr=config.gan_learning_rate, betas=(config.gan_beta1, 0.999))
    if last is not None:
        optimizer_g.load_state_dict(last["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(last["optimizer_d_state_dict"])
    generator = DistributedDataParallel(
        generator_raw, device_ids=[context.local_rank], init_sync=False
    ) if context.distributed else generator_raw
    discriminator = DistributedDataParallel(
        discriminator_raw, device_ids=[context.local_rank], init_sync=False
    ) if context.distributed else discriminator_raw
    seed_everything(rank_seed)
    adversarial_loss, l1_loss = nn.BCEWithLogitsLoss(), nn.L1Loss()
    monitor = TrainingMonitor(output_dir) if context.is_main else None
    accumulation = config.gan_accumulation_steps
    for epoch in range(start_epoch, config.gan_epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        underlying = getattr(train_loader.dataset, "dataset", train_loader.dataset)
        if hasattr(underlying, "set_epoch"):
            underlying.set_epoch(epoch)
        generator.train(); discriminator.train()
        local = torch.zeros(3, dtype=torch.float64, device=context.device)
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        iterator = tqdm(train_loader, desc=f"cGAN {source_key}->{target_key} {epoch + 1}", leave=False, disable=not context.is_main)
        for step, batch in enumerate(iterator):
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            should_step = step - group_start + 1 == group_size
            generator_sync = generator.no_sync() if not should_step and hasattr(generator, "no_sync") else nullcontext()
            discriminator_sync = discriminator.no_sync() if not should_step and hasattr(discriminator, "no_sync") else nullcontext()
            source = _to_gan_range(_flatten_bilateral(batch[source_key].to(context.device, non_blocking=True)))
            target = _to_gan_range(_flatten_bilateral(batch[target_key].to(context.device, non_blocking=True)))
            with generator_sync:
                generated = generator(source)
                with discriminator_sync:
                    real_logits = discriminator(source, target)
                    fake_logits = discriminator(source, generated.detach())
                    loss_d = 0.5 * (
                        adversarial_loss(real_logits, torch.ones_like(real_logits))
                        + adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
                    )
                    (loss_d / group_size).backward()
                for parameter in discriminator_raw.parameters():
                    parameter.requires_grad_(False)
                fake_logits = discriminator_raw(source, generated)
                loss_g = adversarial_loss(fake_logits, torch.ones_like(fake_logits)) + config.gan_lambda_l1 * l1_loss(generated, target)
                (loss_g / group_size).backward()
                for parameter in discriminator_raw.parameters():
                    parameter.requires_grad_(True)
            if should_step:
                optimizer_d.step()
                optimizer_g.step()
                optimizer_d.zero_grad(set_to_none=True)
                optimizer_g.zero_grad(set_to_none=True)
            local += torch.tensor([loss_g.item(), loss_d.item(), 1.0], device=context.device)
        if context.distributed:
            torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.SUM)
        validation_l1 = _validation_l1_ddp(
            generator_raw, validation_loader, source_key, target_key, context
        )
        record = {
            "epoch": epoch + 1,
            "generator_loss": float((local[0] / local[2].clamp_min(1)).item()),
            "discriminator_loss": float((local[1] / local[2].clamp_min(1)).item()),
            "validation_l1": validation_l1,
        }
        history.append(record)
        if validation_l1 < best_validation:
            best_validation, stale = validation_l1, 0
            if context.is_main:
                temporary = output_dir / "best_generator.pt.tmp"
                torch.save({
                    "source": source_key, "target": target_key,
                    "base_channels": config.gan_base_channels, "epoch": epoch + 1,
                    "validation_l1": validation_l1,
                    "training_config": config.as_dict(),
                    "generator_state_dict": generator_raw.state_dict(),
                    "world_size": context.world_size,
                    "initial_state_sha256": generator_state_sha256,
                }, temporary)
                durable_replace(temporary, output_dir / "best_generator.pt")
        else:
            stale += 1
        if context.is_main:
            write_json_atomic(history, output_dir / "history.json")
            temporary = output_dir / "last_training.pt.tmp"
            torch.save({
                "epoch": epoch + 1, "best_validation": best_validation, "stale": stale,
                "history": history, "generator_state_dict": generator_raw.state_dict(),
                "discriminator_state_dict": discriminator_raw.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "world_size": context.world_size,
                "generator_initial_state_sha256": generator_state_sha256,
                "discriminator_initial_state_sha256": discriminator_state_sha256,
            }, temporary)
            durable_replace(temporary, last_path)
            monitor.append({"event": "gan_epoch", **record})
        mhd_barrier(context)
        if stale >= config.gan_patience:
            break
    if context.is_main:
        write_json_atomic({
            "status": "complete", "epochs_completed": len(history),
            "best_validation_l1": best_validation, "world_size": context.world_size,
        }, output_dir / "training_complete.json")
        monitor.finalize(event="gan_epoch")
    mhd_barrier(context)
