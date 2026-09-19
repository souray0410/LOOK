"""Resumable Improved Modality Dropout host training for CFP/OCT MHD features."""
from __future__ import annotations
import math, time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from look.data.observed_pair import collate_observed
from look.evaluation.evaluator import save_prediction_bundle
from look.evaluation.stability import logit_metrics, probabilities_from_logits
from look.models.improved_modality_dropout import forward_improved_dropout_host, improved_dropout_parameter_groups
from look.runtime.host_checkpoint import atomic_save, cpu_tree, save, load
from look.runtime.state import atomic_write_json, file_sha256

DEFAULTS = dict(
    epochs=100, patience=15, minimum_epochs=8, warmup_epochs=5,
    microbatch=16, effective_batch=128, fusion_lr=1e-3,
    weight_decay=1e-4, clip=5.0, precision="fp32", num_workers=0,
    loss="complete_plus_two_missing_ce", missing_weight=1.0,
    primary_metric="macro_f1_complete",
)


def validate_config(config):
    if set(config) != set(DEFAULTS):
        raise ValueError("IMD host training configuration must be complete and versioned")
    if config["loss"] != DEFAULTS["loss"] or config["primary_metric"] != DEFAULTS["primary_metric"]:
        raise ValueError("Unregistered IMD loss/selection protocol")
    if config["precision"] not in ("fp32", "bf16"):
        raise ValueError("Explicit validated precision required")
    if config["effective_batch"] % config["microbatch"] or min(config["microbatch"], config["epochs"], config["patience"]) < 1:
        raise ValueError("Invalid IMD batch or stop configuration")
    if config["missing_weight"] < 0 or not math.isfinite(float(config["missing_weight"])):
        raise ValueError("Finite non-negative missing weight required")


class HostSchedule:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer; self.config = config
        self.base = [group["lr"] for group in optimizer.param_groups]
        self.best = -float("inf"); self.best_epoch = 0; self.stall = 0
    def begin(self, epoch):
        warm = self.config["warmup_epochs"]; limit = self.config["epochs"]
        scale = epoch / max(1, warm) if epoch <= warm else .5 * (1 + math.cos(math.pi * (epoch - warm) / max(1, limit - warm)))
        for group, base in zip(self.optimizer.param_groups, self.base): group["lr"] = base * scale
    def step(self, score, epoch):
        if not math.isfinite(score): raise ValueError("Non-finite IMD development selection score")
        improved = score > self.best
        if improved: self.best = score; self.best_epoch = epoch; self.stall = 0
        else: self.stall += 1
        return improved
    def state_dict(self): return dict(best=self.best, best_epoch=self.best_epoch, stall=self.stall, base=self.base, config=self.config)
    def load_state_dict(self, state):
        if state["config"] != self.config: raise ValueError("IMD schedule changed on resume")
        self.best, self.best_epoch, self.stall, self.base = state["best"], state["best_epoch"], state["stall"], state["base"]


@torch.no_grad()
def evaluate_imd_state(graph, loader, device: torch.device, state="complete"):
    graph.eval()
    labels, logits, participants, patterns = [], [], [], []
    for batch in loader:
        out = forward_improved_dropout_host(
            graph, batch["oct"].to(device), batch["cfp"].to(device), batch["counts"], state=state
        )
        labels.append(batch["label"].numpy())
        logits.append(out.detach().cpu().numpy())
        participants.extend(batch["participant_id"])
        patterns.extend([state] * len(batch["participant_id"]))
    y = np.concatenate(labels)
    z = np.concatenate(logits).astype(np.float64)
    return dict(labels=y, probabilities=probabilities_from_logits(z), logits=z, scores=z[:,1]-z[:,0],
                participant_ids=np.asarray(participants), patterns=np.asarray(patterns), metrics=logit_metrics(y,z))


def _state_schedule(length, seed, epoch):
    return torch.randint(3, (length,), generator=torch.Generator().manual_seed(seed + 170_001 + epoch))


def _simultaneous_loss(graph, batch, device, config, scale):
    labels = batch["label"].to(device).long()
    oct_tensor = batch["oct"].to(device)
    cfp_tensor = batch["cfp"].to(device)
    counts = batch["counts"]
    logits_complete = forward_improved_dropout_host(graph, oct_tensor, cfp_tensor, counts, state="complete")
    logits_oct = forward_improved_dropout_host(graph, oct_tensor, cfp_tensor, counts, state="oct_missing")
    logits_cfp = forward_improved_dropout_host(graph, oct_tensor, cfp_tensor, counts, state="cfp_missing")
    return scale * (F.cross_entropy(logits_complete, labels) + config["missing_weight"] * (
        F.cross_entropy(logits_oct, labels) + F.cross_entropy(logits_cfp, labels)))


def train_improved_dropout_host(graph, train, development, config, seed, output, identity, device, should_pause=lambda: False, *, preflight_updates=None):
    validate_config(config)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    if train.split != "train" or development.split != "development" or set(train.participant_ids) & set(development.participant_ids):
        raise ValueError("IMD split provenance or disjointness failed")
    if seed != 3416:
        raise ValueError("This finite IMD package is locked to seed 3416")
    opt = torch.optim.AdamW(improved_dropout_parameter_groups(graph, config["fusion_lr"]), weight_decay=config["weight_decay"])
    ids = [(n.id,n.name) for n in sorted(graph.nodes, key=lambda n:n.id)]
    schedule = HostSchedule(opt, config); opt.zero_grad(set_to_none=True)
    progress = dict(epoch=1, offset=0, updates=0, history=[], epoch_loss=0.0, epoch_seen=0, seconds=0.0)
    if (out/"last.pt").exists():
        progress = load(out/"last.pt", model=graph, optimizer=opt, scheduler=schedule, identity=identity, node_ids=ids)
    started = time.monotonic()
    def checkpoint():
        nonlocal started
        now=time.monotonic(); progress["seconds"] += now-started; started=now
        save(out/"last.pt", model=graph, optimizer=opt, scheduler=schedule, identity=identity, progress=progress, node_ids=ids)
    def status(state):
        atomic_write_json(dict(state=state,identity=identity,epoch=progress["epoch"],offset=progress["offset"],updates=progress["updates"],updated_at=time.time(),test_access=False), out/"status.json")
    dev_loader = DataLoader(development, batch_size=config["microbatch"], collate_fn=collate_observed, shuffle=False, num_workers=config["num_workers"], generator=torch.Generator().manual_seed(seed))
    if preflight_updates is not None and progress["updates"] >= preflight_updates:
        checkpoint(); status("paused")
        return dict(state="paused", updates=0, total_updates=progress["updates"], reason="preflight_target_already_reached")
    if not (out/"best.pt").exists():
        result = evaluate_imd_state(graph, dev_loader, device, "complete")
        schedule.step(result["metrics"]["macro_f1"], 0)
        atomic_save(out/"best.pt", dict(identity=identity, epoch=0, model=cpu_tree(graph.state_dict()), node_ids=ids,
                                        selection="complete_state_development_macro_f1"))
        save_prediction_bundle(result, out/"development_predictions.npz")
        checkpoint()
    launch_updates = 0
    while progress["epoch"] <= config["epochs"]:
        if schedule.stall >= config["patience"] and progress["epoch"] > config["minimum_epochs"]: break
        epoch=progress["epoch"]; train.set_epoch(epoch); schedule.begin(epoch); graph.train()
        states=_state_schedule(len(train), seed, epoch)
        order=torch.randperm(len(train), generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        while progress["offset"] < len(order):
            if should_pause(): checkpoint(); status("paused"); return dict(state="paused")
            block=order[progress["offset"]:progress["offset"]+config["effective_batch"]]
            loader=DataLoader(Subset(train, block), batch_size=config["microbatch"], collate_fn=collate_observed,
                              shuffle=False, num_workers=config["num_workers"], generator=torch.Generator().manual_seed(seed+epoch))
            cursor=0
            for batch in loader:
                n=len(batch["label"]); indices=block[cursor:cursor+n]; cursor += n
                expected_ids=[str(train.participant_ids[index]) for index in indices]
                if list(map(str,batch["participant_id"])) != expected_ids: raise ValueError("IMD participant order changed within effective batch")
                # states are pre-registered for audit, while the target-task paper loss evaluates all three states.
                _ = states[torch.as_tensor(indices, dtype=torch.long)]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config["precision"]=="bf16"):
                    loss=_simultaneous_loss(graph,batch,device,config,n/len(block))
                if not torch.isfinite(loss): raise ValueError("Non-finite IMD host loss")
                loss.backward(); progress["epoch_loss"] += float(loss.detach()) * len(block); progress["epoch_seen"] += n
            if cursor != len(block): raise ValueError("IMD effective batch participant accounting changed")
            torch.nn.utils.clip_grad_norm_(graph.parameters(), config["clip"], error_if_nonfinite=True)
            opt.step(); opt.zero_grad(set_to_none=True)
            progress["offset"] += len(block); progress["updates"] += 1; launch_updates += 1
            status("training")
            if should_pause() or (preflight_updates is not None and progress["updates"] >= preflight_updates):
                checkpoint(); status("paused"); return dict(state="paused", updates=launch_updates, total_updates=progress["updates"])
        checkpoint(); status("validating")
        result=evaluate_imd_state(graph, dev_loader, device, "complete")
        score=result["metrics"]["macro_f1"]
        improved=schedule.step(score, epoch)
        if improved:
            atomic_save(out/"best.pt", dict(identity=identity, epoch=epoch, model=cpu_tree(graph.state_dict()), node_ids=ids,
                                            selection="complete_state_development_macro_f1"))
            save_prediction_bundle(result, out/"development_predictions.npz")
        counts={int(k): int((states==k).sum()) for k in (0,1,2)}
        progress["history"].append(dict(epoch=epoch, loss=progress["epoch_loss"]/max(1,progress["epoch_seen"]), metrics=result["metrics"], state_counts=counts))
        progress.update(epoch=epoch+1, offset=0, epoch_loss=0.0, epoch_seen=0)
        atomic_write_json(progress["history"], out/"history.json"); checkpoint()
    if schedule.stall < config["patience"]:
        status("needs_review_epoch_cap"); return dict(state="needs_review_epoch_cap")
    selected=torch.load(out/"best.pt", map_location="cpu", weights_only=False)
    if selected["identity"] != identity or selected["node_ids"] != ids: raise ValueError("Selected IMD host identity changed")
    graph.load_state_dict(selected["model"], strict=True); graph.eval()
    replay=evaluate_imd_state(graph, dev_loader, device, "complete")
    saved=np.load(out/"development_predictions.npz", allow_pickle=False)
    if not np.array_equal(saved["participant_ids"].astype(str), replay["participant_ids"].astype(str)) or not np.array_equal(saved["labels"], replay["labels"]):
        raise ValueError("IMD selected replay identity mismatch")
    if not np.allclose(saved["probabilities"], replay["probabilities"], rtol=1e-4, atol=1e-5) or not np.array_equal(saved["logits"].argmax(1), replay["logits"].argmax(1)):
        raise ValueError("IMD selected replay numeric mismatch")
    receipt=dict(schema="look_improved_modality_dropout_host_v1",state="accepted",identity=identity,best_epoch=schedule.best_epoch,
        stop_epoch=progress["epoch"]-1,plateau=True,selection="complete_state_development_macro_f1",test_access=False,
        train_loss="complete_plus_missing_oct_plus_missing_cfp",missing_weight=config["missing_weight"],metrics=replay["metrics"],seconds=progress["seconds"],
        source_nodes=graph.improved_dropout_provenance,files={name:file_sha256(out/name) for name in ("best.pt","last.pt","history.json","development_predictions.npz")})
    atomic_write_json(receipt,out/"accepted.json"); status("completed")
    return receipt
