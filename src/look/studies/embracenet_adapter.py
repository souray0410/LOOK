"""Finite CPU prototype helpers and the proposed one-A EmbraceNet study contract."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import hashlib
import torch

from look.methods.joint import correction_sites, members, member_shapes, read_site, site_level, write_site
from look.methods.operator import LOOKArtifact, PROTOCOL, apply_artifact
from look.models.embracenet import (
    capture_embracenet_sampling,
    embracenet_graph_replay_scope,
    get_embracenet_trace,
    reset_embracenet_inputs,
    restore_embracenet_sampling,
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
        "evaluation_primary_missing": "one ordinary author-style forward per single-missing state; with two modalities the normalized availability is exactly [0,1] or [1,0], so every embraced coordinate is deterministic",
        "evaluation_primary_complete": "exact analytical integrated-mean-logit predictor: compute E[embraced_feature] after the deterministic upstream graph, then apply the current eval linear head; this matches the limit of averaging logits, without Monte Carlo error",
        "evaluation_rng_isolation": "snapshot/restore the A private sampling state around development evaluation so evaluation cannot perturb the stochastic training stream or resume identity",
        "evaluation_pairing": "for the same single-missing state, A and A+LOOK use the same ordinary author-style draw/replay record; indices are deterministic but still logged",
        "evaluation_mc_diagnostic": "K=32 may be retained only as an optional implementation/MC-error diagnostic for the complete stochastic state; it is not model selection or a primary estimator",
        "selection_metric": "development_complete_macro_f1_inherited, computed from the approved integrated-mean-logit complete predictor",
        "missing_state_metrics": "report_only_not_used_for_model_selection",
        "nonlinear_estimand_boundary": "softmax(E[logits]) is not E[softmax(logits)]; expected probability/per-draw NLL/Brier would require a separate preregistered integration protocol",
        "rejected_without_new_approval": "equal_mean macro-F1 over complete/two-missing states, or treating K32 as a mandatory primary estimator",
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
    """Run the actual MHD DAG with optional LOOK writeback and auditable sampling.

    The entry owns both externally prearmed replay and replay_indices supplied to
    this call. Every exit path clears any unconsumed one-shot replay. This
    includes validation failures and successful stops before embraced_feature.
    """
    with embracenet_graph_replay_scope(graph):
        if sampling_state is not None and replay_indices is not None:
            raise ValueError("Use either a sampling-state replay or exact modality-index replay, not both")
        target = stop_node or "fusion_logits"
        stop_level = site_level(graph, target)
        available_sites = correction_sites(graph)
        artifact_sites = [artifact.node_name for artifact in artifacts]
        if len(set(artifact_sites)) != len(artifact_sites):
            raise ValueError("Duplicate LOOK sites")
        if any(site not in available_sites for site in artifact_sites):
            raise ValueError(f"Invalid LOOK sites: {artifact_sites}")
        if artifact_sites != [site for site in available_sites if site in artifact_sites]:
            raise ValueError("LOOK sites must follow forward order")
        if any(site_level(graph, site) > stop_level for site in artifact_sites):
            raise ValueError("LOOK artifact exceeds the requested stop boundary")
        embrace_level = site_level(graph, "embraced_feature")
        if replay_indices is not None and stop_level < embrace_level:
            raise ValueError("Exact EmbraceNet replay requires a stop node at or after embraced_feature")
        for artifact in artifacts:
            if artifact.protocol != PROTOCOL or artifact.split_rule != "channel_split_and_restore_member_shapes_v1":
                raise ValueError("LOOK protocol or split rule mismatch")
        if sampling_state is not None:
            restore_embracenet_sampling(graph, sampling_state)
        if replay_indices is not None:
            set_embracenet_replay(graph, replay_indices)
        reset_embracenet_inputs(
            graph,
            oct_tensor,
            cfp_tensor,
            counts,
            availabilities,
            selection_probabilities,
        )
        by_node = {artifact.node_name: artifact for artifact in artifacts}
        for level in range(-1, stop_level + 1):
            if level >= 0:
                graph.forward(levels=[level])
            for node_name, artifact in by_node.items():
                if site_level(graph, node_name) != level:
                    continue
                if artifact.member_names and tuple(artifact.member_names) != members(node_name):
                    raise ValueError("LOOK member order mismatch")
                if artifact.member_shapes and tuple(map(tuple, artifact.member_shapes)) != member_shapes(graph, node_name):
                    raise ValueError("LOOK member shape mismatch")
                write_site(graph, node_name, apply_artifact(read_site(graph, node_name), artifact))
        result = read_site(graph, target)
        trace = None
        if embrace_level <= stop_level:
            trace = get_embracenet_trace(graph)
        return {
            "output": result,
            "sampling_trace": trace,
            "sampling_state_after": capture_embracenet_sampling(graph),
            "target": target,
        }
