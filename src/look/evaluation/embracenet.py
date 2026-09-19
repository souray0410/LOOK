"""EmbraceNet evaluation paths with sealed-test, sampling-isolation semantics."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import numpy as np
import torch

from look.evaluation.stability import logit_metrics, probabilities_from_logits
from look.models.embracenet import (
    _get_embracenet_operation,
    capture_embracenet_sampling,
    restore_embracenet_sampling,
)
from look.studies.embracenet_adapter import forward_embracenet_with_look


PATTERN_AVAILABILITY = {
    "complete": (1.0, 1.0),
    "oct_missing": (0.0, 1.0),
    "cfp_missing": (1.0, 0.0),
}


class EvaluationPaused(Exception):
    """Safe-boundary pause requested by the owning resource/signal callback."""


def availability_matrix(participant_count: int, pattern: str, device: torch.device) -> torch.Tensor:
    if pattern not in PATTERN_AVAILABILITY or participant_count < 1:
        raise ValueError("Registered EmbraceNet pattern and nonempty batch required")
    return torch.tensor(PATTERN_AVAILABILITY[pattern], dtype=torch.float32, device=device).repeat(participant_count, 1)


@contextmanager
def isolated_embracenet_sampling(graph):
    state = capture_embracenet_sampling(graph)
    try:
        yield state
    finally:
        restore_embracenet_sampling(graph, state)


def _classifier(graph):
    edge = graph.get_edge_by_name("fusion_classifier_edge")
    if edge is None or len(edge.edge_operations) != 1:
        raise ValueError("EmbraceNet graph lacks one classifier operation")
    module = edge.edge_operations[0].function
    if not isinstance(module, torch.nn.Module):
        raise TypeError("EmbraceNet classifier is not a torch module")
    return module


@torch.no_grad()
def complete_participant_moments(graph, oct_tensor, cfp_tensor, counts, *, selection_probabilities=None):
    participants = len(counts)
    availability = availability_matrix(participants, "complete", oct_tensor.device)
    probabilities = (
        torch.full((participants, 2), 0.5, dtype=torch.float32, device=oct_tensor.device)
        if selection_probabilities is None else selection_probabilities
    )
    forward_embracenet_with_look(
        graph, oct_tensor, cfp_tensor, counts, availability,
        selection_probabilities=probabilities, stop_node="joint_participant_feature",
    )
    oct_feature = graph.get_node_by_name("oct_participant_feature").feature_message.current_state
    cfp_feature = graph.get_node_by_name("cfp_participant_feature").feature_message.current_state
    return _get_embracenet_operation(graph).analytical_moments(
        [oct_feature, cfp_feature], availability, probabilities
    )


@torch.no_grad()
def analytical_complete_logits(graph, oct_tensor, cfp_tensor, counts):
    moments = complete_participant_moments(graph, oct_tensor, cfp_tensor, counts)
    graph.eval()
    return _classifier(graph)(moments["mean"]), moments


def _bundle(labels, logits, participants, patterns):
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = probabilities_from_logits(logits)
    return {
        "labels": labels, "logits": logits, "probabilities": probabilities,
        "scores": logits[:, 1] - logits[:, 0],
        "participant_ids": np.asarray(participants), "patterns": np.asarray(patterns),
        "metrics": logit_metrics(labels, logits),
    }


@torch.no_grad()
def evaluate_complete_analytical(graph, loader, device, should_pause=lambda: False):
    if getattr(loader.dataset, "split", None) not in ("development", "train_probe"):
        raise ValueError("Analytical model selection is development/train-probe only")
    graph.eval(); labels=[]; logits=[]; participants=[]
    with isolated_embracenet_sampling(graph):
        for batch in loader:
            if should_pause():
                raise EvaluationPaused()
            values, _ = analytical_complete_logits(
                graph, batch["oct"].to(device, non_blocking=True),
                batch["cfp"].to(device, non_blocking=True), batch["counts"]
            )
            logits.append(values.detach().cpu().numpy())
            labels.extend(batch["label"].tolist())
            participants.extend(map(str, batch["participant_id"]))
    return _bundle(labels, np.concatenate(logits), participants, ["complete"] * len(labels))


@torch.no_grad()
def evaluate_single_missing(
    graph, loader, device, pattern, artifacts: Sequence = (), should_pause=lambda: False
):
    if pattern not in ("oct_missing", "cfp_missing"):
        raise ValueError("Single-missing evaluation requires oct_missing or cfp_missing")
    if getattr(loader.dataset, "split", None) not in ("development", "train_probe"):
        raise ValueError("Missing evaluation is development/train-probe only")
    graph.eval(); labels=[]; logits=[]; participants=[]
    expected = 1 if pattern == "oct_missing" else 0
    with isolated_embracenet_sampling(graph):
        for batch in loader:
            if should_pause():
                raise EvaluationPaused()
            availability = availability_matrix(len(batch["label"]), pattern, device)
            result = forward_embracenet_with_look(
                graph, batch["oct"].to(device, non_blocking=True),
                batch["cfp"].to(device, non_blocking=True), batch["counts"],
                availability, artifacts=artifacts,
            )
            trace = result["sampling_trace"]
            if trace is None or not torch.all(trace["modality_indices"] == expected):
                raise ValueError("Single-missing EmbraceNet trace is not deterministic")
            logits.append(result["output"].detach().cpu().numpy())
            labels.extend(batch["label"].tolist()); participants.extend(map(str, batch["participant_id"]))
    return _bundle(labels, np.concatenate(logits), participants, [pattern] * len(labels))


@torch.no_grad()
def capture_site_mean_variance(graph, batch, site, device, *, pattern="complete", upstream_artifacts: Sequence = ()):
    if pattern not in PATTERN_AVAILABILITY:
        raise ValueError("Unknown EmbraceNet capture pattern")
    oct_tensor=batch["oct"].to(device,non_blocking=True);cfp_tensor=batch["cfp"].to(device,non_blocking=True);counts=batch["counts"]
    if pattern == "complete":
        if upstream_artifacts:
            raise ValueError("Complete reference never applies missing-side LOOK artifacts")
        if site == "embraced_feature":
            moments=complete_participant_moments(graph,oct_tensor,cfp_tensor,counts)
            return moments["mean"].detach(),moments["variance_diagonal"].detach()
        result=forward_embracenet_with_look(
            graph,oct_tensor,cfp_tensor,counts,availability_matrix(len(counts),"complete",device),stop_node=site
        )
        feature=result["output"].detach();return feature,torch.zeros_like(feature)
    result=forward_embracenet_with_look(
        graph,oct_tensor,cfp_tensor,counts,availability_matrix(len(counts),pattern,device),
        artifacts=upstream_artifacts,stop_node=site
    )
    feature=result["output"].detach();return feature,torch.zeros_like(feature)
