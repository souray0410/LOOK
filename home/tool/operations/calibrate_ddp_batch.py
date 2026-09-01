#!/usr/bin/env python3
"""Measure real two-rank training memory for one per-device classifier batch."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW

from MHD_Project.MHD_Utils_V3 import (
    MHDForwardAdapter,
    destroy_mhd_distributed,
    initialize_mhd_distributed,
    mhd_all_gather_object,
    mhd_barrier,
    unwrap_mhd_graph,
)
from look_core.graph import build_resnet50_mhd_graph, optimizer_parameter_groups
from look_core.gan import PairedUNetGenerator, PatchDiscriminator


def classifier_steps(args, context) -> None:
    graph = build_resnet50_mhd_graph(
        args.fusion_position,
        num_classes=5,
        batch_size=args.per_device_batch,
        image_size=224,
        device=context.device,
        pretrained=False,
    )
    adapter = MHDForwardAdapter(
        graph,
        input_nodes=("oct_input", "cfp_input", "label_gt"),
        output_nodes=("fusion_logits", "loss", "batch_accuracy"),
    ).to(context.device)
    model = DistributedDataParallel(
        adapter, device_ids=[context.local_rank], output_device=context.local_rank
    )
    optimizer = AdamW(
        optimizer_parameter_groups(unwrap_mhd_graph(model), 1e-4, 1e-3),
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    torch.cuda.reset_peak_memory_stats(context.device)
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        oct_tensor = torch.randn(args.per_device_batch, 3, 224, 224, device=context.device)
        cfp_tensor = torch.randn_like(oct_tensor)
        labels = torch.randint(0, 5, (args.per_device_batch,), device=context.device)
        with torch.amp.autocast("cuda", enabled=True):
            outputs = model(
                {"oct_input": oct_tensor, "cfp_input": cfp_tensor, "label_gt": labels}
            )
            loss = outputs["loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def gan_steps(args, context) -> None:
    generator_raw = PairedUNetGenerator(64).to(context.device)
    discriminator_raw = PatchDiscriminator(64).to(context.device)
    generator = DistributedDataParallel(generator_raw, device_ids=[context.local_rank])
    discriminator = DistributedDataParallel(discriminator_raw, device_ids=[context.local_rank])
    optimizer_g = Adam(generator_raw.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = Adam(discriminator_raw.parameters(), lr=2e-4, betas=(0.5, 0.999))
    adversarial_loss, l1_loss = nn.BCEWithLogitsLoss(), nn.L1Loss()
    torch.cuda.reset_peak_memory_stats(context.device)
    for _ in range(args.steps):
        source = torch.randn(args.per_device_batch, 3, 224, 224, device=context.device)
        target = torch.randn_like(source)
        generated = generator(source)
        optimizer_d.zero_grad(set_to_none=True)
        real_logits = discriminator(source, target)
        fake_logits = discriminator(source, generated.detach())
        loss_d = 0.5 * (
            adversarial_loss(real_logits, torch.ones_like(real_logits))
            + adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
        )
        loss_d.backward()
        optimizer_d.step()
        optimizer_g.zero_grad(set_to_none=True)
        for parameter in discriminator_raw.parameters():
            parameter.requires_grad_(False)
        fake_logits = discriminator_raw(source, generated)
        loss_g = adversarial_loss(fake_logits, torch.ones_like(fake_logits)) + 100.0 * l1_loss(
            generated, target
        )
        loss_g.backward()
        optimizer_g.step()
        for parameter in discriminator_raw.parameters():
            parameter.requires_grad_(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-device-batch", type=int, required=True)
    parser.add_argument("--stage", choices=("classifier", "gan"), default="classifier")
    parser.add_argument("--fusion-position", default="feature")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.per_device_batch < 1 or args.steps < 1:
        raise ValueError("Batch and steps must be positive")

    context = initialize_mhd_distributed()
    try:
        torch.cuda.empty_cache()
        if args.stage == "classifier":
            classifier_steps(args, context)
        else:
            gan_steps(args, context)
        torch.cuda.synchronize(context.device)
        gib = 1024**3
        local = {
            "rank": context.rank,
            "device": str(context.device),
            "stage": args.stage,
            "per_device_batch": args.per_device_batch,
            "allocated_gib": torch.cuda.memory_allocated(context.device) / gib,
            "reserved_gib": torch.cuda.memory_reserved(context.device) / gib,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(context.device) / gib,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(context.device) / gib,
        }
        records = mhd_all_gather_object(local, context)
        if context.is_main:
            print(json.dumps(records, indent=2))
        mhd_barrier(context)
    finally:
        destroy_mhd_distributed()


if __name__ == "__main__":
    main()
