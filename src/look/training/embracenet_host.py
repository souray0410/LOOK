"""Resumable one-seed EmbraceNet A training with explicit participant missingness."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from look.data.observed_pair import collate_observed
from look.evaluation.embracenet import EvaluationPaused, evaluate_complete_analytical
from look.evaluation.evaluator import save_prediction_bundle
from look.models.embracenet import (
    capture_embracenet_sampling,
    embracenet_parameter_groups,
    forward_embracenet_host,
    restore_embracenet_sampling,
)
from look.runtime.embracenet_sampling import (
    restore_embracenet_sampling_from_progress,
    stash_embracenet_sampling,
)
from look.runtime.host_checkpoint import atomic_save, cpu_tree, load, save
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.training.observed_host import HostSchedule, validate_config


MASK_SEED_OFFSET = 730019
STATE_NAMES = ("complete", "oct_missing", "cfp_missing")


def _new_mask_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + MASK_SEED_OFFSET)
    return generator


def _mask_schedule(progress, generator, epoch: int, participants: int):
    if progress.get("mask_epoch") == epoch:
        states = progress.get("mask_states")
        if not isinstance(states, torch.Tensor) or states.dtype != torch.int64 or tuple(states.shape) != (participants,):
            raise ValueError("Saved EmbraceNet mask schedule is invalid")
        return states
    if progress["offset"] != 0 or progress.get("mask_epoch") is not None:
        raise ValueError("Cannot replace an in-progress EmbraceNet mask schedule")
    states = torch.randint(3, (participants,), generator=generator, dtype=torch.int64)
    progress["mask_epoch"] = epoch
    progress["mask_states"] = states.clone()
    progress["mask_rng_state"] = generator.get_state().clone()
    progress["mask_schedule_sha256"] = stable_hash({"epoch": epoch, "states": states.tolist()})
    return states


def _availability(states: torch.Tensor, device: torch.device) -> torch.Tensor:
    if states.ndim != 1 or states.dtype != torch.int64 or torch.any((states < 0) | (states > 2)):
        raise ValueError("Invalid EmbraceNet training missing-state vector")
    result = torch.ones((len(states), 2), dtype=torch.float32, device=device)
    result[states == 1, 0] = 0.0
    result[states == 2, 1] = 0.0
    return result


def _state_counts(states: torch.Tensor) -> dict[str, int]:
    return {name: int((states == index).sum()) for index, name in enumerate(STATE_NAMES)}


def _save_boundary(path, *, graph, optimizer, scheduler, identity, progress, node_ids, mask_generator):
    progress["mask_rng_state"] = mask_generator.get_state().clone()
    stash_embracenet_sampling(graph, progress)
    return save(
        path, model=graph, optimizer=optimizer, scheduler=scheduler,
        identity=identity, progress=progress, node_ids=node_ids,
    )


def _load_boundary(path, *, graph, optimizer, scheduler, identity, node_ids, mask_generator):
    progress = load(
        path, model=graph, optimizer=optimizer, scheduler=scheduler,
        identity=identity, node_ids=node_ids,
    )
    state = progress.get("mask_rng_state")
    if not isinstance(state, torch.Tensor):
        raise ValueError("EmbraceNet checkpoint lacks mask RNG state")
    mask_generator.set_state(state.cpu())
    restore_embracenet_sampling_from_progress(graph, progress)
    return progress


def train_embracenet_host(
    graph,
    train,
    development,
    config,
    seed,
    output,
    identity,
    device,
    should_pause=lambda: False,
    *,
    preflight_updates=None,
    preflight_target_updates=None,
):
    """Train exactly one EmbraceNet A; complete-dev selection uses analytical mean logits."""
    validate_config(config)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    if train.split != "train" or development.split != "development":
        raise ValueError("EmbraceNet training requires declared train/development roles")
    if set(train.participant_ids) & set(development.participant_ids):
        raise ValueError("Train/development participants overlap")
    if seed != 3416:
        raise ValueError("This finite EmbraceNet package is locked to seed 3416")
    if preflight_target_updates is not None and (
        isinstance(preflight_target_updates, bool)
        or not isinstance(preflight_target_updates, int)
        or preflight_target_updates < 1
    ):
        raise ValueError("preflight_target_updates must be a positive integer")

    optimizer = torch.optim.AdamW(
        embracenet_parameter_groups(graph, config["pretrained_lr"], config["new_layer_lr"]),
        weight_decay=config["weight_decay"],
    )
    scheduler = HostSchedule(optimizer, config)
    optimizer.zero_grad(set_to_none=True)
    node_ids = [(node.id, node.name) for node in sorted(graph.nodes, key=lambda node: node.id)]
    mask_generator = _new_mask_generator(seed)
    progress = dict(
        epoch=1, offset=0, updates=0, history=[], epoch_loss=0.0, epoch_seen=0,
        seconds=0.0, mask_epoch=None, mask_states=None,
        mask_rng_state=mask_generator.get_state().clone(), mask_schedule_sha256=None,
    )
    stash_embracenet_sampling(graph, progress)
    if (out / "last.pt").exists():
        progress = _load_boundary(
            out / "last.pt", graph=graph, optimizer=optimizer, scheduler=scheduler,
            identity=identity, node_ids=node_ids, mask_generator=mask_generator,
        )

    started = time.monotonic()
    def checkpoint():
        nonlocal started
        now = time.monotonic(); progress["seconds"] += now - started; started = now
        _save_boundary(
            out / "last.pt", graph=graph, optimizer=optimizer, scheduler=scheduler,
            identity=identity, progress=progress, node_ids=node_ids, mask_generator=mask_generator,
        )

    def status(state):
        atomic_write_json({
            "state": state, "identity": identity, "epoch": progress["epoch"],
            "offset": progress["offset"], "updates": progress["updates"],
            "mask_epoch": progress.get("mask_epoch"),
            "mask_schedule_sha256": progress.get("mask_schedule_sha256"),
            "updated_at": time.time(), "test_access": False,
        }, out / "status.json")

    dev_loader = DataLoader(
        development, batch_size=config["microbatch"], collate_fn=collate_observed,
        shuffle=False, num_workers=config["num_workers"],
        generator=torch.Generator().manual_seed(seed),
    )
    if preflight_target_updates is not None and progress["updates"] >= preflight_target_updates:
        checkpoint(); status("paused")
        return {
            "state": "paused", "updates": 0, "total_updates": progress["updates"],
            "reason": "preflight_target_already_reached",
        }

    if not (out / "best.pt").exists():
        if should_pause():
            checkpoint(); status("paused")
            return {"state": "paused", "updates": 0, "total_updates": progress["updates"]}
        try:
            result = evaluate_complete_analytical(graph, dev_loader, device, should_pause)
        except EvaluationPaused:
            checkpoint(); status("paused")
            return {"state": "paused", "updates": 0, "total_updates": progress["updates"]}
        scheduler.step(result["metrics"]["macro_f1"], 0)
        atomic_save(out / "best.pt", {
            "identity": identity, "epoch": 0, "model": cpu_tree(graph.state_dict()),
            "node_ids": node_ids, "sampling_state": capture_embracenet_sampling(graph),
            "selection": "analytical_integrated_mean_logits_development_macro_f1",
        })
        save_prediction_bundle(result, out / "development_predictions.npz")
        checkpoint()

    launch_updates = 0
    while progress["epoch"] <= config["epochs"]:
        if scheduler.stall >= config["patience"] and progress["epoch"] > config["minimum_epochs"]:
            break
        epoch = progress["epoch"]; train.set_epoch(epoch); scheduler.begin(epoch); graph.train()
        states = _mask_schedule(progress, mask_generator, epoch, len(train))
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        while progress["offset"] < len(order):
            if should_pause():
                checkpoint(); status("paused"); return {"state": "paused"}
            block = order[progress["offset"]:progress["offset"] + config["effective_batch"]]
            loader = DataLoader(
                Subset(train, block), batch_size=config["microbatch"], collate_fn=collate_observed,
                shuffle=False, num_workers=config["num_workers"],
                generator=torch.Generator().manual_seed(seed + epoch),
            )
            cursor = 0
            for batch in loader:
                n = len(batch["label"])
                indices = block[cursor:cursor+n]; cursor += n
                expected_ids = [str(train.participant_ids[index]) for index in indices]
                if list(map(str, batch["participant_id"])) != expected_ids:
                    raise ValueError("Training participant order changed within effective batch")
                batch_states = states[torch.as_tensor(indices, dtype=torch.long)]
                availability = _availability(batch_states, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config["precision"] == "bf16"):
                    forward_embracenet_host(
                        graph, batch["oct"].to(device), batch["cfp"].to(device), batch["counts"],
                        availability, labels=batch["label"].to(device), loss_scale=n / len(block),
                    )
                    loss = graph.get_node_by_name("loss").feature_message.current_state
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite EmbraceNet host loss")
                graph.backward(levels=graph.backward_levels)
                progress["epoch_loss"] += float(loss.detach()) * len(block)
                progress["epoch_seen"] += n
            if cursor != len(block):
                raise ValueError("Effective batch participant accounting changed")
            torch.nn.utils.clip_grad_norm_(graph.parameters(), config["clip"], error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            progress["offset"] += len(block); progress["updates"] += 1; launch_updates += 1
            status("training")
            target_reached = (
                preflight_target_updates is not None
                and progress["updates"] >= preflight_target_updates
            )
            pause_requested = should_pause()
            if pause_requested or target_reached or (
                preflight_updates is not None and launch_updates >= preflight_updates
            ):
                checkpoint(); status("paused")
                return {
                    "state": "paused", "updates": launch_updates,
                    "total_updates": progress["updates"],
                    "reason": "signal_or_resource" if pause_requested else "preflight",
                }

        # Every completed epoch reaches an optimizer boundary before development
        # evaluation. A signal during evaluation can therefore resume exactly.
        checkpoint()
        status("validating")
        try:
            result = evaluate_complete_analytical(graph, dev_loader, device, should_pause)
        except EvaluationPaused:
            status("paused")
            return {
                "state": "paused", "updates": launch_updates,
                "total_updates": progress["updates"], "reason": "evaluation_pause",
            }
        score = result["metrics"]["macro_f1"]
        improved = scheduler.step(score, epoch)
        counts = _state_counts(states)
        schedule_sha = progress["mask_schedule_sha256"]
        if improved:
            atomic_save(out / "best.pt", {
                "identity": identity, "epoch": epoch, "model": cpu_tree(graph.state_dict()),
                "node_ids": node_ids, "sampling_state": capture_embracenet_sampling(graph),
                "selection": "analytical_integrated_mean_logits_development_macro_f1",
            })
            save_prediction_bundle(result, out / "development_predictions.npz")
        progress["history"].append({
            "epoch": epoch,
            "loss": progress["epoch_loss"] / max(1, progress["epoch_seen"]),
            "metrics": result["metrics"], "mask_counts": counts,
            "mask_schedule_sha256": schedule_sha,
        })
        progress.update(
            epoch=epoch + 1, offset=0, epoch_loss=0.0, epoch_seen=0,
            mask_epoch=None, mask_states=None, mask_schedule_sha256=None,
        )
        atomic_write_json(progress["history"], out / "history.json")
        checkpoint()

    if scheduler.stall < config["patience"]:
        status("needs_review_epoch_cap")
        return {"state": "needs_review_epoch_cap"}

    selected = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    if selected["identity"] != identity or selected["node_ids"] != node_ids:
        raise ValueError("Selected EmbraceNet host identity changed")
    graph.load_state_dict(selected["model"], strict=True)
    restore_embracenet_sampling(graph, selected["sampling_state"])
    graph.eval()
    try:
        replay = evaluate_complete_analytical(graph, dev_loader, device, should_pause)
    except EvaluationPaused:
        checkpoint(); status("paused")
        return {
            "state": "paused", "updates": launch_updates,
            "total_updates": progress["updates"], "reason": "final_replay_pause",
        }
    saved = np.load(out / "development_predictions.npz", allow_pickle=False)
    if not np.array_equal(saved["participant_ids"].astype(str), replay["participant_ids"].astype(str)) or not np.array_equal(saved["labels"], replay["labels"]):
        raise ValueError("Selected EmbraceNet development identity replay changed")
    if not np.array_equal(saved["logits"], replay["logits"]):
        raise ValueError("Analytical selected EmbraceNet logits are not exact on replay")

    receipt = {
        "schema": "look_embracenet_host_v1", "state": "accepted", "identity": identity,
        "best_epoch": scheduler.best_epoch, "stop_epoch": progress["epoch"] - 1,
        "plateau": True, "selection": "analytical_integrated_mean_logits_development_macro_f1",
        "train_missing_probabilities": {"complete": 1/3, "oct_missing": 1/3, "cfp_missing": 1/3},
        "mask_seed": seed + MASK_SEED_OFFSET, "test_access": False,
        "metrics": replay["metrics"], "seconds": progress["seconds"],
        "source_nodes": graph.embracenet_provenance,
        "files": {name: file_sha256(out / name) for name in (
            "best.pt", "last.pt", "history.json", "development_predictions.npz"
        )},
    }
    atomic_write_json(receipt, out / "accepted.json"); status("completed")
    return receipt
