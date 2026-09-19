import copy

import pytest
import torch
from mhd_framework.models import create_model

from look.methods.joint import correction_sites, read_site, write_site
from look.models.embracenet import (
    DEFAULT_EMBRACEMENT_SIZE,
    EmbraceNetFusion,
    build_embracenet_host,
    capture_embracenet_sampling,
    embracenet_parameter_groups,
    forward_embracenet_host,
    get_embracenet_trace,
    restore_embracenet_sampling,
    set_embracenet_replay,
)
from look.models.observed_participant import ObservedParticipantModel
from look.runtime.embracenet_sampling import (
    restore_embracenet_sampling_from_progress,
    stash_embracenet_sampling,
)
from look.studies.embracenet_adapter import (
    evaluation_draw_seed,
    evaluation_replay_indices,
    forward_embracenet_with_look,
    proposed_trial_protocol,
)


def _parents():
    config = dict(name="resnet18", num_classes=2, views=1)
    return [ObservedParticipantModel(create_model(config)) for _ in range(2)]


def test_author_domain_sampling_replay_and_state_restore():
    torch.set_num_threads(2)
    module = EmbraceNetFusion([5, 7], embracement_size=11, sampling_seed=123)
    x = torch.randn(3, 5)
    y = torch.randn(3, 7)
    availability = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.float32)
    probabilities = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.9, 0.1]], dtype=torch.float32)
    before = module.get_sampling_state()
    first = module([x, y], availability, probabilities)
    trace = module.peek_last_trace()
    assert trace is not None
    state_after = module.get_sampling_state()
    assert state_after["draw_count"] == 3 * 11

    module.set_sampling_state(before)
    second = module([x, y], availability, probabilities)
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    module.set_sampling_state(state_after)
    continuation = module([x, y], availability, probabilities)
    module.set_sampling_state(state_after)
    continuation_replay = module([x, y], availability, probabilities)
    torch.testing.assert_close(continuation, continuation_replay, rtol=0, atol=0)

    module.set_replay_indices(trace["modality_indices"])
    exact = module([x, y], availability, probabilities)
    torch.testing.assert_close(first, exact, rtol=0, atol=0)
    assert module.peek_last_trace()["replayed"] is True


def test_unavailable_nan_is_safe_and_has_no_input_or_docking_gradient():
    module = EmbraceNetFusion([4, 4], embracement_size=8, sampling_seed=9)
    missing = torch.full((2, 4), float("nan"), requires_grad=True)
    observed = torch.randn(2, 4, requires_grad=True)
    availability = torch.tensor([[0, 1], [0, 1]], dtype=torch.float32)
    output = module([missing, observed], availability)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert missing.grad is not None and torch.count_nonzero(missing.grad) == 0
    assert observed.grad is not None and observed.grad.abs().sum() > 0
    for parameter in module.docking_0.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0
    assert module.docking_1.weight.grad.abs().sum() > 0


def test_invalid_author_extension_domains_fail_closed():
    module = EmbraceNetFusion([3, 3], embracement_size=4, sampling_seed=2)
    x = torch.randn(2, 3)
    with pytest.raises(ValueError, match="at least one available"):
        module([x, x], torch.zeros(2, 2))
    with pytest.raises(ValueError, match="positive total"):
        module([x, x], torch.ones(2, 2), torch.zeros(2, 2))
    with pytest.raises(ValueError, match="non-finite"):
        module([torch.full_like(x, float("nan")), x], torch.ones(2, 2))


def test_mhd_r18_prototype_nodes_backward_sampling_and_progress_restore():
    torch.set_num_threads(2)
    cfp, oct_parent = _parents()
    graph = build_embracenet_host(cfp, oct_parent, embracement_size=DEFAULT_EMBRACEMENT_SIZE, sampling_seed=3416)
    graph.eval()
    oct_tensor = torch.randn(3, 3, 224, 224)
    cfp_tensor = torch.randn(3, 3, 224, 224)
    counts = [1, 2]
    availability = torch.tensor([[1, 1], [0, 1]], dtype=torch.float32)

    state0 = capture_embracenet_sampling(graph)
    logits = forward_embracenet_host(graph, oct_tensor, cfp_tensor, counts, availability)
    assert logits.shape == (2, 2)
    trace = get_embracenet_trace(graph)
    assert trace["modality_indices"].shape == (2, DEFAULT_EMBRACEMENT_SIZE)

    expected_sites = [
        "joint_input", "joint_stem", "joint_stage1", "joint_stage2", "joint_stage3",
        "joint_stage4", "joint_features", "joint_participant_feature", "embraced_feature",
    ]
    assert correction_sites(graph) == expected_sites
    for site in expected_sites:
        value = read_site(graph, site).clone()
        write_site(graph, site, value)
        torch.testing.assert_close(value, read_site(graph, site), rtol=0, atol=0)

    restore_embracenet_sampling(graph, state0)
    replayed = forward_embracenet_host(graph, oct_tensor, cfp_tensor, counts, availability)
    torch.testing.assert_close(logits, replayed, rtol=0, atol=0)

    set_embracenet_replay(graph, trace["modality_indices"])
    exact = forward_embracenet_host(graph, oct_tensor, cfp_tensor, counts, availability)
    torch.testing.assert_close(logits, exact, rtol=0, atol=0)

    labels = torch.tensor([0, 1])
    graph.zero_grad(set_to_none=True)
    forward_embracenet_host(graph, oct_tensor, cfp_tensor, counts, availability, labels=labels)
    graph.backward(levels=graph.backward_levels)
    embrace_grads = [
        parameter.grad
        for edge in graph.edges
        if edge.name == "embracenet_edge"
        for _, parameter in edge.named_edge_parameters()
    ]
    assert any(grad is not None and grad.abs().sum() > 0 for grad in embrace_grads)

    groups = embracenet_parameter_groups(graph, 1e-4, 1e-3)
    assert [group["group_name"] for group in groups] == ["pretrained_encoders", "embracenet_and_head"]
    assert groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 1e-3

    progress = {}
    stash_embracenet_sampling(graph, progress)
    saved = copy.deepcopy(progress)
    forward_embracenet_host(graph, oct_tensor, cfp_tensor, counts, availability)
    restore_embracenet_sampling_from_progress(graph, saved)
    after_restore = forward_embracenet_host(graph, oct_tensor, cfp_tensor, counts, availability)
    restore_embracenet_sampling_from_progress(graph, saved)
    expected = forward_embracenet_host(graph, oct_tensor, cfp_tensor, counts, availability)
    torch.testing.assert_close(after_restore, expected, rtol=0, atol=0)


def test_study_forward_has_trace_and_concrete_proposed_protocol():
    torch.set_num_threads(2)
    cfp, oct_parent = _parents()
    graph = build_embracenet_host(cfp, oct_parent, embracement_size=16, sampling_seed=3416).eval()
    oct_tensor = torch.randn(2, 3, 224, 224)
    cfp_tensor = torch.randn(2, 3, 224, 224)
    availability = torch.tensor([[1, 1]], dtype=torch.float32)
    first = forward_embracenet_with_look(graph, oct_tensor, cfp_tensor, [2], availability)
    assert first["output"].shape == (1, 2)
    assert first["sampling_trace"]["modality_indices"].shape == (1, 16)
    trace = first["sampling_trace"]
    set_embracenet_replay(graph, trace["modality_indices"])
    second = forward_embracenet_with_look(graph, oct_tensor, cfp_tensor, [2], availability)
    torch.testing.assert_close(first["output"], second["output"], rtol=0, atol=0)

    protocol = proposed_trial_protocol()
    assert protocol["status"] == "proposed_not_training_authorized"
    assert protocol["data"] == {"name": "ukb_small_20260909_v1", "train": 1264, "development": 296, "test": "sealed", "seed": 3416}
    assert protocol["new_scientific_choices"]["embracement_size"] == 256
    assert protocol["inherited_training"]["microbatch"] == 16
    assert protocol["inherited_training"]["effective_batch"] == 128
    assert protocol["inherited_training"]["pretrained_lr"] == 1e-4
    assert protocol["inherited_training"]["new_layer_lr"] == 1e-3


def test_research_entry_masks_unavailable_nan_before_r18_and_has_zero_raw_gradient():
    torch.set_num_threads(2)
    cfp, oct_parent = _parents()
    graph = build_embracenet_host(cfp, oct_parent, embracement_size=16, sampling_seed=3416).eval()
    oct_tensor = torch.full((2, 3, 224, 224), float("nan"), requires_grad=True)
    cfp_tensor = torch.randn(2, 3, 224, 224, requires_grad=True)
    availability = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    result = forward_embracenet_with_look(graph, oct_tensor, cfp_tensor, [2], availability)
    assert torch.isfinite(result["output"]).all()
    result["output"].sum().backward()
    assert oct_tensor.grad is not None and torch.count_nonzero(oct_tensor.grad) == 0
    assert cfp_tensor.grad is not None and cfp_tensor.grad.abs().sum() > 0


def test_evaluation_common_random_seed_and_exact_same_state_replay_trace():
    seed = evaluation_draw_seed("config-A", "participant-17", 3)
    assert seed == evaluation_draw_seed("config-A", "participant-17", 3)
    complete = torch.tensor([[1.0, 1.0]])
    missing_oct = torch.tensor([[0.0, 1.0]])
    probabilities = torch.tensor([[0.5, 0.5]])
    complete_indices = evaluation_replay_indices(complete, probabilities, 32, seed)
    missing_indices = evaluation_replay_indices(missing_oct, probabilities, 32, seed)
    assert torch.equal(complete_indices, evaluation_replay_indices(complete, probabilities, 32, seed))
    assert torch.equal(missing_indices, evaluation_replay_indices(missing_oct, probabilities, 32, seed))
    assert torch.all(missing_indices == 1)
