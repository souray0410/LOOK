from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from .graph import optimizer_parameter_groups, reset_and_forward
from .metrics import classification_metrics
from .reproducibility import sha256, write_json_atomic


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
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    accumulation = config.effective_batch_size // config.micro_batch_size
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
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)):
            oct_tensor = batch["oct"].to(device, non_blocking=True)
            cfp_tensor = batch["cfp"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
                logits = reset_and_forward(graph, oct_tensor, cfp_tensor)
                loss = criterion(logits, labels) / accumulation
            scaler.scale(loss).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * accumulation

        validation = predict(graph, validation_loader, device)
        metrics = classification_metrics(validation["labels"], validation["probabilities"])
        score = float(metrics[config.primary_metric])
        record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, len(train_loader)),
            "validation": metrics,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
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
                    "labels_sha256": sha256(config.labels_csv),
                },
                temporary,
            )
            temporary.replace(run_dir / "best.pt")
        else:
            stale_epochs += 1
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
            },
            temporary,
        )
        temporary.replace(last_path)
        if stale_epochs >= config.patience:
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    graph.load_state_dict(checkpoint["graph_state_dict"])
    write_json_atomic(
        {"status": "complete", "epochs_completed": len(history), "best_epoch": checkpoint["epoch"], "best_score": checkpoint["score"]},
        run_dir / "training_complete.json",
    )
    return graph, history
