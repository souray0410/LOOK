from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from MHD_Project.MHD_Utils_V4 import (
    MHD_Monitor,
    MHD_ParallelConfig,
    MHD_Trainer,
    destroy_mhd_distributed,
    initialize_mhd_distributed,
    mhd_barrier,
)
from test_mhd_trainer_criteria import _graph, validation_macro_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = initialize_mhd_distributed()
    try:
        graph = _graph()
        optimizer = torch.optim.SGD(graph.parameters(), lr=0.1)
        monitor = MHD_Monitor(["batch_accuracy"])
        trainer = MHD_Trainer(
            graph,
            optimizer,
            monitor,
            forward_levels=[0, 1],
            backward_levels=[3, 2],
            criteria=validation_macro_f1,
            criteria_mode="max",
            save_dir=str(args.output),
            input_nodes=["input", "target"],
            input_mapping={"input": "features", "target": "labels"},
            output_nodes=["logits", "loss", "batch_accuracy", "target"],
            parallel=MHD_ParallelConfig(data_parallel="ddp"),
            distributed_context=context,
        )
        target = torch.tensor([0, 0]) if context.rank == 0 else torch.tensor([1, 1])
        metrics = trainer.eval_epoch([
            {
                "features": torch.tensor([[8.0, 0.0], [8.0, 0.0]]),
                "labels": target,
                "participant_id": [f"rank-{context.rank}-a", f"rank-{context.rank}-b"],
            }
        ], epoch=0)
        expected = torch.tensor(1 / 3).item()
        if metrics["validation_macro_f1"] != expected:
            raise RuntimeError(f"Expected {expected}, got {metrics}")
        trainer.save_last_checkpoint(1)
        mhd_barrier(context)
        if context.is_main:
            (args.output / "smoke.json").write_text(
                json.dumps({
                    "status": "complete",
                    "world_size": context.world_size,
                    "validation_macro_f1": metrics["validation_macro_f1"],
                }, indent=2),
                encoding="utf-8",
            )
    finally:
        destroy_mhd_distributed()


if __name__ == "__main__":
    main()
