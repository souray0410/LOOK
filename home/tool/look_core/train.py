from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from MHD_Project.MHD_Utils_V4 import (
    MHD_DistributedContext,
    MHD_Monitor,
    MHD_ParallelConfig,
    MHD_Trainer,
    mhd_barrier,
)

from .distributed import assert_module_state_identical

from .graph import (
    classification_loss_metadata,
    optimizer_parameter_groups,
    reset_and_forward,
)
from .metrics import classification_metrics, validation_macro_f1
from .monitoring import TrainingMonitor
from .reproducibility import (
    backbone_implementation_sha256,
    sha256,
    write_json_atomic,
)


def _schedule(epoch: int, epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def predict(graph, loader, device: torch.device) -> Dict[str, np.ndarray]:
    graph.eval()
    labels, probabilities, participants = [], [], []
    for batch in tqdm(loader, desc="Predict", leave=False):
        oct_tensor = batch["oct"].to(device, non_blocking=True)
        cfp_tensor = batch["cfp"].to(device, non_blocking=True)
        logits = reset_and_forward(graph, oct_tensor, cfp_tensor)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(batch["label"].numpy())
        participants.extend(batch["participant_id"])
    return {
        "labels": np.concatenate(labels),
        "probabilities": np.concatenate(probabilities),
        "participant_ids": np.asarray(participants),
    }


def train_complete_model_ddp(
    graph,
    train_loader,
    validation_loader,
    config,
    run_dir: Path,
    context: MHD_DistributedContext,
):
    stage_epochs = config.epochs
    stage_patience = config.patience
    backward_levels = graph.backward_levels
    run_dir.mkdir(parents=True, exist_ok=True)
    backbone_hash = backbone_implementation_sha256(Path(__file__).resolve().parents[2])
    initial_state_sha256 = assert_module_state_identical(graph, context)
    optimizer = AdamW(
        optimizer_parameter_groups(graph, config.pretrained_lr, config.new_layer_lr),
        weight_decay=config.weight_decay,
    )
    stage_warmup = config.warmup_epochs
    scheduler = LambdaLR(
        optimizer, lambda epoch: _schedule(epoch, stage_epochs, stage_warmup)
    )
    accumulation = config.effective_batch_size // config.global_micro_batch_size
    graph_monitor = MHD_Monitor(
        config.monitor_nodes,
        config.monitor_edges,
        node_states=(
            "feature_message.current_state",
            "gradient_message.current_state",
        ),
    )
    input_mapping = {"label_gt": "label"}
    if graph.get_node_by_name("oct_input") is not None:
        input_mapping["oct_input"] = "oct"
    if graph.get_node_by_name("cfp_input") is not None:
        input_mapping["cfp_input"] = "cfp"
    trainer = MHD_Trainer(
        graph,
        optimizer,
        graph_monitor,
        forward_levels=graph.forward_levels,
        backward_levels=backward_levels,
        criteria=validation_macro_f1,
        criteria_mode="max",
        save_dir=str(run_dir),
        lr_scheduler=scheduler,
        input_nodes=tuple(input_mapping),
        input_mapping=input_mapping,
        output_nodes=("fusion_logits", "loss", "batch_accuracy", "label_gt"),
        parallel=MHD_ParallelConfig(
            data_parallel="ddp" if context.distributed else "none",
        ),
        distributed_context=context,
        precision="fp16" if config.amp else "fp32",
        grad_accum_steps=accumulation,
        monitor_interval_steps=config.monitor_interval_steps,
    )
    completion_path = run_dir / "training_complete.json"
    if completion_path.is_file():
        trainer.load_checkpoint(load_best=True)
        return graph

    start_epoch = 0
    if (run_dir / "last").is_dir():
        start_epoch = trainer.load_checkpoint(load_last=True)
    history_path = run_dir / "history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.is_file()
        else []
    )
    history = history[:start_epoch]
    monitor = TrainingMonitor(run_dir) if context.is_main else None
    for epoch in range(start_epoch, stage_epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        train_metrics = trainer.train_epoch(train_loader, epoch)
        eval_metrics = trainer.eval_epoch(validation_loader, epoch)
        complete_logits = trainer.last_eval_tensors["fusion_logits"]
        complete_labels = trainer.last_eval_tensors["label_gt"]
        if len(complete_labels) != len(validation_loader.dataset):
            raise RuntimeError(
                f"Distributed validation returned {len(complete_labels)} rows; "
                f"expected {len(validation_loader.dataset)}"
            )
        probabilities = torch.softmax(complete_logits.float(), dim=1).numpy()
        labels = complete_labels.long().numpy()
        metrics = classification_metrics(labels, probabilities)
        metrics["cross_entropy"] = float(eval_metrics[trainer.loss_node_name])
        graph_score = float(eval_metrics[trainer.criteria_name])
        score = float(metrics[config.primary_metric])
        if not np.isclose(graph_score, score, atol=1e-7, rtol=1e-6):
            raise RuntimeError(
                f"PyTorch criterion {graph_score} does not match full validation "
                f"{config.primary_metric} {score}"
            )
        record = {
            "epoch": epoch + 1,
            "train_loss": float(train_metrics[trainer.loss_node_name]),
            "train_batch_accuracy": float(train_metrics["batch_accuracy"]),
            "validation": metrics,
            "criteria": trainer.criteria_name,
            "criteria_value": graph_score,
            "learning_rates": [group["lr"] for group in trainer.optimizer.param_groups],
            "graph_monitor": trainer.last_monitor_metrics,
        }
        history.append(record)
        trainer.save_last_checkpoint(epoch + 1)
        if context.is_main:
            write_json_atomic(history, run_dir / "history.json")
            monitor.append({"event": "epoch", **record})
        mhd_barrier(context)
        if epoch + 1 - int(trainer.history["best_epoch"]) >= stage_patience:
            break
    epochs_completed = len(history)
    best_epoch = int(trainer.history["best_epoch"])
    best_score = float(trainer.history["best_eval_value"])
    trainer.load_checkpoint(load_best=True)
    if context.is_main:
        temporary = run_dir / "best.pt.tmp"
        torch.save({
            "architecture_id": graph.architecture_id,
            "epoch": best_epoch,
            "primary_metric": config.primary_metric,
            "criteria": trainer.criteria_name,
            "criteria_mode": trainer.criteria_mode,
            "criteria_scope": "full_validation_distributed",
            "score": best_score,
            "graph_state_dict": graph.state_dict(),
            "config": config.as_dict(),
            "backbone_training": "complete_modalities_only",
            "training_strategy": "end_to_end_finetuning",
            "training_stage": "complete_modalities",
            "framework_version": "MHD V4",
            "backward_api": "MHD_Graph.backward",
            "initial_state_sha256": initial_state_sha256,
            "labels_sha256": sha256(config.labels_csv),
            "backbone_implementation_sha256": backbone_hash,
            "world_size": context.world_size,
            "classification_loss": classification_loss_metadata(graph),
        }, temporary)
        temporary.replace(run_dir / "best.pt")
        write_json_atomic({
            "status": "complete",
            "epochs_completed": epochs_completed,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "criteria": trainer.criteria_name,
            "criteria_mode": trainer.criteria_mode,
            "criteria_scope": "full_validation_distributed",
            "canonical_checkpoint": str(run_dir / "best"),
            "portable_checkpoint": str(run_dir / "best.pt"),
            "checkpoint_sha256": sha256(run_dir / "best.pt"),
            "labels_sha256": sha256(config.labels_csv),
            "backbone_implementation_sha256": backbone_hash,
            "world_size": context.world_size,
            "framework_version": "MHD V4",
            "backward_api": "MHD_Graph.backward",
            "initial_state_sha256": initial_state_sha256,
            "classification_loss": classification_loss_metadata(graph),
            "training_strategy": "end_to_end_finetuning",
            "training_stage": "complete_modalities",
            "backward_levels": list(backward_levels),
        }, run_dir / "training_complete.json")
        monitor.finalize()
    mhd_barrier(context)
    return graph
