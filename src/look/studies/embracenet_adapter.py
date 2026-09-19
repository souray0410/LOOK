"""Finite CPU prototype helpers and the proposed one-A EmbraceNet study contract."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import hashlib
import torch

from look.methods.joint import members, member_shapes, read_site, site_level, write_site
from look.methods.operator import LOOKArtifact, PROTOCOL, apply_artifact
from look.models.embracenet import (
    capture_embracenet_sampling,
    get_embracenet_trace,
    restore_embracenet_sampling,
    set_embracenet_context,
    set_embracenet_replay,
)

TRIAL_PROTOCOL = {
    "schema": "look_embracenet_trial_protocol_v1_proposed",
    "status": "proposed_not_training_authorized",
    "method_A": "EmbraceNet",
    "architecture": "two_observed_eye_resnet18_encoders_then_participant_pool_then_embracenet",
    "data": {
        "name": "ukb_small_20260909_v1",
        "train": 1264,
        "development": 296,
        "test": "sealed",
        "seed": 3416,
    },
    "inherited_training": {
        "initialization": "public_imagenet_v1_fresh_resnet18_encoders",
        "precision": "fp32",
        "microbatch": 16,
        "effective_batch": 128,
        "optimizer": "AdamW",
        "pretrained_lr": 1e-4,
        "new_layer_lr": 1e-3,
        "weight_decay": 1e-4,
        "clip": 5.0,
        "warmup_epochs": 5,
        "minimum_epochs": 8,
        "patience": 15,
        "epochs_cap": 100,
        "loss": "unweighted_cross_entropy",
        "selection": "development_complete_macro_f1",
        "batchnorm": "inherit_existing_R18_host_behavior_no_policy_change",
    },
    "new_scientific_choices": {
        "embracement_size": 256,
        "modality_order": ["oct", "cfp"],
        "selection_probabilities_complete": [0.5, 0.5],
        "missing_training": "non_dedicated_equal_three_state_participant_sampling",
        "training_state_probabilities": {
            "complete": 1 / 3,
            "missing_oct": 1 / 3,
            "missing_cfp": 1 / 3,
        },
        "missing_scope": "both_eyes_of_the_selected_modality_are_unavailable_together",
        "missing_input_safety": "explicit_torch_where_zero_fill_before_any_R18_encoder; availability still controls EmbraceNet selection",
        "evaluation_sampling": "32_fixed_common_random_draws_per_participant_and_state; average logits before metrics",
        "evaluation_draw_seed_rule": "sha256(config_id, participant_id, draw_index) -> 63-bit seed; state is intentionally omitted for common random numbers",
        "evaluation_pairing": "same seed across complete/missing states gives common random numbers; state-specific modality indices are recorded; exact indices are reused only for A versus A+LOOK under the same availability state",
        "selection_metric": "development_complete_macro_f1_inherited",
        "missing_state_metrics": "report_only_not_used_for_model_selection",
        "rejected_without_new_approval": "equal_mean_macro_f1_over_complete_and_two_missing_states",
    },
    "look_comparison": {
        "checkpoint": "same_frozen_A_checkpoint_for_A_and_A_plus_LOOK",
        "missing_states": ["missing_oct", "missing_cfp"],
        "look_arms": ["pca_free_mean", "residual_rrr_free_mean"],
        "fit_split": "train_only_complete_reference_vs_matching_missing_state",
        "evaluation": "reuse_exact_recorded_sampling_draws_for_A_and_A_plus_LOOK",
        "test": "sealed",
    },
}


def proposed_trial_protocol() -> dict[str, object]:
    import copy
    return copy.deepcopy(TRIAL_PROTOCOL)


def evaluation_draw_seed(config_id: str, participant_id: str, draw_index: int) -> int:
    """Stable common-random-number seed; deliberately independent of missing state."""
    if not config_id or not participant_id or type(draw_index) is not int or draw_index < 0:
        raise ValueError("config_id, participant_id and non-negative integer draw_index are required")
    payload = f"{config_id}|{participant_id}|{draw_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def evaluation_replay_indices(
    availabilities: torch.Tensor,
    selection_probabilities: torch.Tensor,
    embracement_size: int,
    seed: int,
) -> torch.Tensor:
    """Generate the exact author-style categorical trace for a fixed evaluation draw."""
    if availabilities.ndim != 2 or selection_probabilities.shape != availabilities.shape:
        raise ValueError("availability/probability matrices must share [batch,modality] shape")
    if not torch.isfinite(availabilities).all() or not torch.all((availabilities == 0) | (availabilities == 1)):
        raise ValueError("availabilities must be finite 0/1")
    probabilities = selection_probabilities.to(dtype=torch.float32) * availabilities.to(dtype=torch.float32)
    total = probabilities.sum(dim=-1, keepdim=True)
    if torch.any(total <= 0) or not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError("each sample requires positive finite available selection probability")
    probabilities = probabilities / total
    generator = torch.Generator(device=probabilities.device).manual_seed(int(seed))
    return torch.multinomial(probabilities, num_samples=int(embracement_size), replacement=True, generator=generator)


def _safe_missing_inputs(
    oct_tensor: torch.Tensor,
    cfp_tensor: torch.Tensor,
    counts,
    availabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero unavailable participant-eye rows before either R18 sees the tensor.

    This matches the accepted project's zero-masked missing-training mechanics,
    but uses torch.where before the encoder so an unavailable NaN placeholder is
    never multiplied by zero inside the network.
    """
    values = counts.tolist() if isinstance(counts, torch.Tensor) else list(counts)
    if any(type(value) is not int or value not in (1, 2) for value in values):
        raise ValueError("observed-eye counts must contain only 1 or 2")
    if sum(values) != len(oct_tensor) or len(cfp_tensor) != len(oct_tensor):
        raise ValueError("packed eye tensors do not match participant counts")
    if availabilities.ndim != 2 or tuple(availabilities.shape) != (len(values), 2):
        raise ValueError("EmbraceNet participant availabilities must have shape [participants,2]")
    availability = availabilities.to(device=oct_tensor.device)
    if not torch.isfinite(availability).all() or not torch.all((availability == 0) | (availability == 1)):
        raise ValueError("availabilities must contain only finite 0/1 values")
    eye_mask = availability.repeat_interleave(torch.tensor(values, device=availability.device), dim=0).bool()
    def safe(value: torch.Tensor, column: int) -> torch.Tensor:
        shape = (len(value),) + (1,) * (value.ndim - 1)
        mask = eye_mask[:, column].reshape(shape)
        return torch.where(mask, value, torch.zeros_like(value))
    return safe(oct_tensor, 0), safe(cfp_tensor, 1)


def _reset_embracenet_inputs(
    graph,
    oct_tensor: torch.Tensor,
    cfp_tensor: torch.Tensor,
    counts,
    availabilities: torch.Tensor,
    selection_probabilities: torch.Tensor | None,
) -> None:
    for node in graph.nodes:
        node.reset()
    safe_oct, safe_cfp = _safe_missing_inputs(oct_tensor, cfp_tensor, counts, availabilities)
    graph.get_node_by_name("oct_input").feature_message.current_state = safe_oct
    graph.get_node_by_name("cfp_input").feature_message.current_state = safe_cfp
    set_embracenet_context(graph, counts, len(oct_tensor), availabilities, selection_probabilities)


def forward_embracenet_with_look(
    graph,
    oct_tensor: torch.Tensor,
    cfp_tensor: torch.Tensor,
    counts,
    availabilities: torch.Tensor,
    artifacts: Sequence[LOOKArtifact] = (),
    *,
    selection_probabilities: torch.Tensor | None = None,
    sampling_state: Mapping[str, object] | None = None,
    replay_indices: torch.Tensor | None = None,
    stop_node: str | None = None,
) -> dict[str, object]:
    """Run the actual MHD DAG with optional LOOK writeback and auditable sampling."""
    if sampling_state is not None and replay_indices is not None:
        raise ValueError("Use either a sampling-state replay or exact modality-index replay, not both")
    if sampling_state is not None:
        restore_embracenet_sampling(graph, sampling_state)
    if replay_indices is not None:
        set_embracenet_replay(graph, replay_indices)
    _reset_embracenet_inputs(
        graph,
        oct_tensor,
        cfp_tensor,
        counts,
        availabilities,
        selection_probabilities,
    )
    by_node = {artifact.node_name: artifact for artifact in artifacts}
    if len(by_node) != len(artifacts):
        raise ValueError("Duplicate LOOK sites")
    target = stop_node or "fusion_logits"
    stop_level = site_level(graph, target)
    for level in range(-1, stop_level + 1):
        if level >= 0:
            graph.forward(levels=[level])
        for node_name, artifact in by_node.items():
            if site_level(graph, node_name) != level:
                continue
            if artifact.protocol != PROTOCOL or artifact.split_rule != "channel_split_and_restore_member_shapes_v1":
                raise ValueError("LOOK protocol or split rule mismatch")
            if artifact.member_names and tuple(artifact.member_names) != members(node_name):
                raise ValueError("LOOK member order mismatch")
            if artifact.member_shapes and tuple(map(tuple, artifact.member_shapes)) != member_shapes(graph, node_name):
                raise ValueError("LOOK member shape mismatch")
            write_site(graph, node_name, apply_artifact(read_site(graph, node_name), artifact))
    result = read_site(graph, target)
    trace = None
    if site_level(graph, "embraced_feature") <= stop_level:
        trace = get_embracenet_trace(graph)
    return {
        "output": result,
        "sampling_trace": trace,
        "sampling_state_after": capture_embracenet_sampling(graph),
        "target": target,
    }
