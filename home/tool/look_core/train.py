from __future__ import annotations

import json
import math
from contextlib import nullcontext
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
    mhd_barrier,
    prepare_mhd_model,
    unwrap_mhd_graph,
)

from .distributed import all_gather_object, assert_module_state_identical

from .graph import (
    classification_loss_metadata,
    optimizer_parameter_groups,
    reset_and_forward,
    set_crt_train_mode,
)
from .metrics import classification_metrics
from .monitoring import TrainingMonitor
from .reproducibility import sha256, write_json_atomic


def _schedule(epoch: int, epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def backward_mhd_loss(
    graph,
    loss: torch.Tensor,
    scaler,
    backward_levels,
    divisor: int = 1,
) -> None:
    """Route an AMP-compatible scalar loss seed through the V4 hypergraph."""
    if scaler.is_enabled():
        scaler.scale(loss)
        scale = float(scaler.get_scale())
    else:
        scale = 1.0
    graph._backward(
        levels=list(backward_levels),
        retain_graph=False,
        loss_scale=scale / float(divisor),
    )
    if scale != 1.0:
        for node in graph.nodes:
            initial = node.gradient_message.initial_state
            current = node.gradient_message.current_state
            node.gradient_message.current_state = initial + (current - initial) / scale


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


def train_complete_model(
    graph,
    train_loader,
    validation_loader,
    config,
    run_dir: Path,
    device: torch.device,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    graph.to(device)
    parameter_groups = optimizer_parameter_groups(
        graph, config.pretrained_lr, config.new_layer_lr
    )
    optimizer = AdamW(parameter_groups, weight_decay=config.weight_decay)
    scheduler = LambdaLR(
        optimizer, lambda epoch: _schedule(epoch, config.epochs, config.warmup_epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    accumulation = config.effective_batch_size // config.global_micro_batch_size
    best_score, stale_epochs = -float("inf"), 0
    history = []
    start_epoch = 0
    last_path = run_dir / "last.pt"
    if last_path.is_file():
        last = torch.load(last_path, map_location=device, weights_only=False)
        graph.load_state_dict(last["graph_state_dict"])
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scheduler.load_state_dict(last["scheduler_state_dict"])
        scaler.load_state_dict(last["scaler_state_dict"])
        start_epoch = int(last["epoch"])
        best_score = float(last["best_score"])
        stale_epochs = int(last["stale_epochs"])
        history = list(last["history"])

    for epoch in range(start_epoch, config.epochs):
        graph.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        optimizer_steps = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)):
            oct_tensor = batch["oct"].to(device, non_blocking=True)
            cfp_tensor = batch["cfp"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
                outputs = reset_and_forward(graph, oct_tensor, cfp_tensor, labels)
                loss = outputs["loss"]
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            backward_mhd_loss(graph, loss, scaler, graph.backward_levels, group_size)
            if step - group_start + 1 == group_size:
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                optimizer_steps += int(float(scaler.get_scale()) >= scale_before)
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item()

        validation = predict(graph, validation_loader, device)
        metrics = classification_metrics(validation["labels"], validation["probabilities"])
        score = float(metrics[config.primary_metric])
        record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, len(train_loader)),
            "validation": metrics,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "optimizer_steps": optimizer_steps,
        }
        history.append(record)
        write_json_atomic(history, run_dir / "history.json")

        if score > best_score:
            best_score, stale_epochs = score, 0
            temporary = run_dir / "best.pt.tmp"
            torch.save(
                {
                    "architecture_id": graph.architecture_id,
                    "epoch": epoch + 1,
                    "primary_metric": config.primary_metric,
                    "score": score,
                    "graph_state_dict": graph.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.as_dict(),
                    "backbone_training": "complete_modalities_only",
                    "framework_version": "MHD V4",
                    "backward_api": "MHD_Graph.backward",
                    "labels_sha256": sha256(config.labels_csv),
                    "classification_loss": classification_loss_metadata(graph),
                },
                temporary,
            )
            temporary.replace(run_dir / "best.pt")
        else:
            stale_epochs += 1
        if optimizer_steps:
            scheduler.step()
        temporary = run_dir / "last.pt.tmp"
        torch.save(
            {
                "epoch": epoch + 1,
                "graph_state_dict": graph.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_score": best_score,
                "stale_epochs": stale_epochs,
                "history": history,
                "framework_version": "MHD V4",
                "backward_api": "MHD_Graph.backward",
                "classification_loss": classification_loss_metadata(graph),
            },
            temporary,
        )
        temporary.replace(last_path)
        if stale_epochs >= config.patience:
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    graph.load_state_dict(checkpoint["graph_state_dict"])
    write_json_atomic(
        {
            "status": "complete",
            "epochs_completed": len(history),
            "best_epoch": checkpoint["epoch"],
            "best_score": checkpoint["score"],
            "framework_version": "MHD V4",
            "backward_api": "MHD_Graph.backward",
            "classification_loss": classification_loss_metadata(graph),
        },
        run_dir / "training_complete.json",
    )
    return graph, history


@torch.no_grad()
def predict_distributed(model, loader, context: MHD_DistributedContext) -> Dict[str, np.ndarray]:
    model.eval()
    local = {"labels": [], "probabilities": [], "participant_ids": [], "loss_sum": 0.0, "count": 0}
    for batch in tqdm(loader, desc="Distributed validation", leave=False, disable=not context.is_main):
        labels = batch["label"].to(context.device, non_blocking=True)
        outputs = model({
            "oct_input": batch["oct"].to(context.device, non_blocking=True),
            "cfp_input": batch["cfp"].to(context.device, non_blocking=True),
            "label_gt": labels,
        })
        probabilities = torch.softmax(outputs["fusion_logits"], dim=1)
        local["labels"].extend(labels.cpu().tolist())
        local["probabilities"].extend(probabilities.cpu().tolist())
        local["participant_ids"].extend(map(str, batch["participant_id"]))
        local["loss_sum"] += float(outputs["loss"].item()) * len(labels)
        local["count"] += len(labels)
    pieces = all_gather_object(local, context)
    labels = np.asarray([item for piece in pieces for item in piece["labels"]], dtype=np.int64)
    probabilities = np.asarray([item for piece in pieces for item in piece["probabilities"]], dtype=np.float64)
    participants = np.asarray([item for piece in pieces for item in piece["participant_ids"]])
    expected = len(loader.dataset)
    if len(labels) != expected:
        raise RuntimeError(f"Distributed validation returned {len(labels)} rows; expected {expected}")
    return {
        "labels": labels,
        "probabilities": probabilities,
        "participant_ids": participants,
        "cross_entropy": sum(piece["loss_sum"] for piece in pieces) / max(1, sum(piece["count"] for piece in pieces)),
    }


def train_complete_model_ddp(
    graph,
    train_loader,
    validation_loader,
    config,
    run_dir: Path,
    context: MHD_DistributedContext,
    *,
    training_stage: str = "representation",
    stage1_checkpoint_sha256: str | None = None,
):
    if training_stage not in {"representation", "crt"}:
        raise ValueError(f"Unknown classifier training stage: {training_stage}")
    stage_epochs = config.epochs if training_stage == "representation" else config.crt_epochs
    stage_patience = config.patience if training_stage == "representation" else config.crt_patience
    backward_levels = (
        graph.backward_levels
        if training_stage == "representation"
        else graph.crt_backward_levels
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"
    last = None
    if last_path.is_file():
        last = torch.load(last_path, map_location=graph.device, weights_only=False)
        graph.load_state_dict(last["graph_state_dict"])
    initial_state_sha256 = assert_module_state_identical(graph, context)
    model = prepare_mhd_model(
        graph,
        input_nodes=("oct_input", "cfp_input", "label_gt"),
        output_nodes=("fusion_logits", "loss", "batch_accuracy"),
        levels=graph.forward_levels,
        backward_levels=backward_levels,
        parallel=MHD_ParallelConfig(
            data_parallel="ddp" if context.distributed else "none",
        ),
        context=context,
        precision="fp16" if config.amp else "fp32",
    )
    raw_graph = unwrap_mhd_graph(model)
    if training_stage == "representation":
        optimizer = AdamW(
            optimizer_parameter_groups(raw_graph, config.pretrained_lr, config.new_layer_lr),
            weight_decay=config.weight_decay,
        )
        stage_warmup = config.warmup_epochs
    else:
        classifier_parameters = [
            parameter for parameter in raw_graph.parameters() if parameter.requires_grad
        ]
        if not classifier_parameters:
            raise RuntimeError("cRT has no trainable classifier parameters")
        optimizer = AdamW(
            classifier_parameters,
            lr=config.crt_learning_rate,
            weight_decay=config.crt_weight_decay,
        )
        stage_warmup = 0
    scheduler = LambdaLR(
        optimizer, lambda epoch: _schedule(epoch, stage_epochs, stage_warmup)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and context.device.type == "cuda")
    accumulation = config.effective_batch_size // config.global_micro_batch_size
    start_epoch, best_score, stale_epochs, history = 0, -float("inf"), 0, []
    if last is not None:
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scheduler.load_state_dict(last["scheduler_state_dict"])
        scaler.load_state_dict(last["scaler_state_dict"])
        start_epoch = int(last["epoch"])
        best_score = float(last["best_score"])
        stale_epochs = int(last["stale_epochs"])
        history = list(last["history"])
    monitor = TrainingMonitor(run_dir) if context.is_main else None
    graph_monitor = MHD_Monitor(
        config.monitor_nodes,
        config.monitor_edges,
        node_states=(
            "feature_message.current_state",
            "gradient_message.current_state",
        ),
    )
    for epoch in range(start_epoch, stage_epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        if training_stage == "crt":
            set_crt_train_mode(raw_graph)
        optimizer.zero_grad(set_to_none=True)
        loss_sum, accuracy_sum, count = 0.0, 0.0, 0
        optimizer_steps = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False, disable=not context.is_main)):
            labels = batch["label"].to(context.device, non_blocking=True)
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            should_step = step - group_start + 1 == group_size
            sync_context = (
                model.no_sync()
                if not should_step and hasattr(model, "no_sync")
                else nullcontext()
            )
            with sync_context:
                with torch.amp.autocast("cuda", enabled=config.amp and context.device.type == "cuda"):
                    outputs = model({
                        "oct_input": batch["oct"].to(context.device, non_blocking=True),
                        "cfp_input": batch["cfp"].to(context.device, non_blocking=True),
                        "label_gt": labels,
                    })
                    loss = outputs["loss"]
                backward_mhd_loss(raw_graph, loss, scaler, backward_levels, group_size)
            should_monitor = (step + 1) % config.monitor_interval_steps == 0 or step + 1 == len(train_loader)
            if should_monitor:
                graph_monitor.monitor_node(raw_graph, prefix="node/")
            if should_step:
                scaler.unscale_(optimizer)
                if should_monitor:
                    graph_monitor.monitor_edge(raw_graph, prefix="edge/", train_mode=True)
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                optimizer_steps += int(float(scaler.get_scale()) >= scale_before)
                optimizer.zero_grad(set_to_none=True)
            batch_count = len(labels)
            loss_sum += float(outputs["loss"].detach().item()) * batch_count
            accuracy_sum += float(outputs["batch_accuracy"].detach().item()) * batch_count
            count += batch_count
        train_parts = all_gather_object({"loss": loss_sum, "accuracy": accuracy_sum, "count": count}, context)
        validation_model = model.module if context.distributed else model
        validation = predict_distributed(validation_model, validation_loader, context)
        metrics = classification_metrics(validation["labels"], validation["probabilities"])
        metrics["cross_entropy"] = float(validation["cross_entropy"])
        total_count = sum(part["count"] for part in train_parts)
        train_loss = sum(part["loss"] for part in train_parts) / max(1, total_count)
        train_accuracy = sum(part["accuracy"] for part in train_parts) / max(1, total_count)
        score = float(metrics[config.primary_metric])
        monitor_parts = all_gather_object(graph_monitor.get_mean_metrics(), context)
        monitor_keys = sorted({key for part in monitor_parts for key in part})
        graph_metrics = {
            key: float(np.mean([part[key] for part in monitor_parts if key in part]))
            for key in monitor_keys
        }
        graph_monitor.reset()
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_batch_accuracy": train_accuracy,
            "validation": metrics,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "optimizer_steps": optimizer_steps,
            "graph_monitor": graph_metrics,
        }
        history.append(record)
        if score > best_score:
            best_score, stale_epochs = score, 0
            if context.is_main:
                temporary = run_dir / "best.pt.tmp"
                torch.save({
                    "architecture_id": raw_graph.architecture_id,
                    "epoch": epoch + 1,
                    "primary_metric": config.primary_metric,
                    "score": score,
                    "graph_state_dict": raw_graph.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.as_dict(),
                    "backbone_training": "complete_modalities_only",
                    "training_strategy": "classifier_retraining",
                    "training_stage": training_stage,
                    "stage1_checkpoint_sha256": stage1_checkpoint_sha256,
                    "framework_version": "MHD V4",
                    "backward_api": "MHD_Graph.backward",
                    "initial_state_sha256": initial_state_sha256,
                    "labels_sha256": sha256(config.labels_csv),
                    "world_size": context.world_size,
                    "classification_loss": classification_loss_metadata(raw_graph),
                }, temporary)
                temporary.replace(run_dir / "best.pt")
        else:
            stale_epochs += 1
        if optimizer_steps:
            scheduler.step()
        if context.is_main:
            write_json_atomic(history, run_dir / "history.json")
            temporary = run_dir / "last.pt.tmp"
            torch.save({
                "epoch": epoch + 1,
                "graph_state_dict": raw_graph.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_score": best_score,
                "stale_epochs": stale_epochs,
                "history": history,
                "world_size": context.world_size,
                "framework_version": "MHD V4",
                "backward_api": "MHD_Graph.backward",
                "initial_state_sha256": initial_state_sha256,
                "classification_loss": classification_loss_metadata(raw_graph),
                "training_stage": training_stage,
                "stage1_checkpoint_sha256": stage1_checkpoint_sha256,
            }, temporary)
            temporary.replace(last_path)
            monitor.append({"event": "epoch", **record})
        mhd_barrier(context)
        if stale_epochs >= stage_patience:
            break
    if context.is_main:
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        write_json_atomic({
            "status": "complete",
            "epochs_completed": len(history),
            "best_epoch": checkpoint["epoch"],
            "best_score": checkpoint["score"],
            "world_size": context.world_size,
            "framework_version": "MHD V4",
            "backward_api": "MHD_Graph.backward",
            "initial_state_sha256": initial_state_sha256,
            "classification_loss": checkpoint["classification_loss"],
            "training_strategy": "classifier_retraining",
            "training_stage": training_stage,
            "stage1_checkpoint_sha256": stage1_checkpoint_sha256,
            "backward_levels": list(backward_levels),
        }, run_dir / "training_complete.json")
        monitor.finalize()
    mhd_barrier(context)
