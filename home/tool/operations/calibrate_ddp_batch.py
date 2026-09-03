#!/usr/bin/env python3
"""Measure real two-rank training memory for one per-device classifier batch."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW

from MHD_Project.MHD_Utils_V4 import (
    MHD_ParallelConfig,
    destroy_mhd_distributed,
    initialize_mhd_distributed,
    mhd_barrier,
    prepare_mhd_model,
    unwrap_mhd_graph,
)
from look_core.distributed import all_gather_object, assert_module_state_identical
from look_core.graph import build_resnet50_mhd_graph, optimizer_parameter_groups
from look_core.gan import PairedUNetGenerator, PatchDiscriminator


def classifier_steps(args, context) -> dict:
    started = time.perf_counter()
    timings = {}

    def mark(name: str) -> None:
        torch.cuda.synchronize(context.device)
        now = time.perf_counter()
        timings[name] = now - mark.previous
        mark.previous = now
        print(f"rank={context.rank} stage={name} seconds={timings[name]:.3f}", flush=True)

    mark.previous = started
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    graph = build_resnet50_mhd_graph(
        args.fusion_position,
        num_classes=2,
        batch_size=args.per_device_batch,
        image_size=224,
        device="cpu",
        pretrained=False,
    )
    mark("build_graph")
    initial_state_sha256 = assert_module_state_identical(graph, context)
    mark("verify_initial_state")
    model = prepare_mhd_model(
        graph,
        input_nodes=("oct_input", "cfp_input", "label_gt"),
        output_nodes=("fusion_logits", "loss", "batch_accuracy"),
        parallel=MHD_ParallelConfig(
            data_parallel="ddp" if context.distributed else "none",
        ),
        context=context,
        precision="fp16",
    )
    mark("prepare_model")
    raw_graph = unwrap_mhd_graph(model)
    optimizer = AdamW(
        optimizer_parameter_groups(raw_graph, 1e-4, 1e-3),
        weight_decay=1e-4,
    )
    mark("prepare_optimizer")
    torch.cuda.reset_peak_memory_stats(context.device)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        oct_tensor = torch.randn(args.per_device_batch, 2, 3, 224, 224, device=context.device)
        cfp_tensor = torch.randn_like(oct_tensor)
        labels = torch.randint(0, 4, (args.per_device_batch,), device=context.device)
        with torch.amp.autocast("cuda", enabled=True):
            outputs = model(
                {"oct_input": oct_tensor, "cfp_input": cfp_tensor, "label_gt": labels}
            )
            loss = outputs["loss"]
        mark(f"step_{step + 1}_forward")
        raw_graph.backward(levels=raw_graph.backward_levels)
        mark(f"step_{step + 1}_backward")
        optimizer.step()
        mark(f"step_{step + 1}_optimizer")
    parameters = tuple(raw_graph.parameters())
    parameter_sum = torch.zeros((), dtype=torch.float32, device=context.device)
    parameter_squared_sum = torch.zeros_like(parameter_sum)
    sampled_values = 0
    for parameter in parameters:
        flat = parameter.detach().reshape(-1).float()
        stride = max(1, flat.numel() // 1024)
        values = flat[::stride][:1024]
        sampled_values += values.numel()
        parameter_sum.add_(values.sum())
        parameter_squared_sum.add_(values.square().sum())
    mark("parameter_fingerprint")
    logits_gradient = raw_graph.get_node_by_name(
        "fusion_logits"
    ).gradient_message.current_state
    return {
        "parameter_sum": float(parameter_sum.item()),
        "parameter_squared_sum": float(parameter_squared_sum.item()),
        "parameter_fingerprint_values": sampled_values,
        "timings_seconds": timings,
        "fusion_logits_gradient_l2": float(logits_gradient.detach().float().norm().item()),
        "backward_api": "MHD_Graph.backward",
        "initial_state_sha256": initial_state_sha256,
    }


def gan_steps(args, context) -> dict:
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    generator_raw = PairedUNetGenerator(64)
    discriminator_raw = PatchDiscriminator(64)
    generator_state_sha256 = assert_module_state_identical(generator_raw, context)
    discriminator_state_sha256 = assert_module_state_identical(discriminator_raw, context)
    generator_raw.to(context.device)
    discriminator_raw.to(context.device)
    generator = DistributedDataParallel(
        generator_raw, device_ids=[context.local_rank], init_sync=False
    )
    discriminator = DistributedDataParallel(
        discriminator_raw, device_ids=[context.local_rank], init_sync=False
    )
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
    return {
        "backward_api": "torch.autograd (independent filling baseline)",
        "generator_initial_state_sha256": generator_state_sha256,
        "discriminator_initial_state_sha256": discriminator_state_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-device-batch", type=int, required=True)
    parser.add_argument("--stage", choices=("classifier", "gan"), default="classifier")
    parser.add_argument("--fusion-position", default="feature")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.per_device_batch < 1 or args.steps < 1:
        raise ValueError("Batch and steps must be positive")

    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "0")
    context = initialize_mhd_distributed()
    try:
        torch.cuda.empty_cache()
        if args.stage == "classifier":
            details = classifier_steps(args, context)
        else:
            details = gan_steps(args, context)
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
            **details,
        }
        records = all_gather_object(local, context)
        if context.is_main:
            print(json.dumps(records, indent=2))
        mhd_barrier(context)
    finally:
        destroy_mhd_distributed()


if __name__ == "__main__":
    main()
