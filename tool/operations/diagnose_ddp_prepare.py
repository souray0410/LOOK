#!/usr/bin/env python3
"""Time the individual V4 DDP preparation layers on a real LOOK graph."""

from __future__ import annotations

import argparse
import time

import torch
from torch.nn.parallel import DistributedDataParallel

from V4.MHD_Utils_V4 import (
    MHD_ParallelConfig,
    _MHD_ForwardAdapter,
    _build_device_mesh,
    destroy_mhd_distributed,
    initialize_mhd_distributed,
)
from look_core.graph import build_resnet50_mhd_graph


def timed(context, name, action):
    torch.cuda.synchronize(context.device)
    started = time.perf_counter()
    result = action()
    torch.cuda.synchronize(context.device)
    print(
        f"rank={context.rank} stage={name} seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-position", default="feature")
    parser.add_argument("--broadcast-buffers", action="store_true")
    parser.add_argument("--skip-init-sync", action="store_true")
    args = parser.parse_args()
    context = initialize_mhd_distributed()
    try:
        torch.manual_seed(3407)
        graph = timed(
            context,
            "build_graph",
            lambda: build_resnet50_mhd_graph(
                args.fusion_position,
                batch_size=1,
                pretrained=False,
                device=context.device,
            ),
        )
        adapter = timed(
            context,
            "build_adapter",
            lambda: _MHD_ForwardAdapter(
                graph,
                ("oct_input", "cfp_input", "label_gt"),
                ("fusion_logits", "loss", "batch_accuracy"),
            ),
        )
        mesh = timed(
            context,
            "build_device_mesh",
            lambda: _build_device_mesh(
                context,
                MHD_ParallelConfig(data_parallel="ddp"),
            )[0],
        )
        timed(
            context,
            "ddp_constructor",
            lambda: DistributedDataParallel(
                adapter,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                process_group=mesh["dp"].get_group(),
                broadcast_buffers=args.broadcast_buffers,
                init_sync=not args.skip_init_sync,
            ),
        )
    finally:
        destroy_mhd_distributed()


if __name__ == "__main__":
    main()
