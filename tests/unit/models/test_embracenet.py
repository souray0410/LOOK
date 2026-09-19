import copy

import pytest
import torch
from mhd_framework.models import create_model

from look.methods.joint import correction_sites, member_shapes, members, read_site, write_site
from look.methods.operator import LOOKArtifact, downsample_flatten
from look.models.embracenet import (
    DEFAULT_EMBRACEMENT_SIZE,
    EmbraceNetFusion,
    build_embracenet_host,
    capture_embracenet_sampling,
    embracenet_parameter_groups,
    forward_embracenet_host,
    get_embracenet_trace,
    prepare_embracenet_inputs,
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
    choices = protocol["new_scientific_choices"]
    assert choices["evaluation_primary_missing"].startswith("one ordinary author-style forward")
    assert "analytical integrated-mean-logit" in choices["evaluation_primary_complete"]
    assert choices["evaluation_mc_diagnostic"].startswith("K=32 may be retained only")
    assert "softmax(E[logits]) is not E[softmax(logits)]" in choices["nonlinear_estimand_boundary"]


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


def _matched_graphs(*, embracement_size=32, sampling_seed=3416):
    first = build_embracenet_host(*_parents(), embracement_size=embracement_size, sampling_seed=sampling_seed)
    second = build_embracenet_host(*_parents(), embracement_size=embracement_size, sampling_seed=sampling_seed)
    second.load_state_dict(copy.deepcopy(first.state_dict()), strict=True)
    return first, second


def _nonzero_artifact(graph, site, feature, missing_pattern, *, bias_value=0.35):
    factor = 16 if feature.ndim == 4 else 1
    flat, down_shape = downsample_flatten(feature.detach(), factor)
    width = flat.shape[1]
    component = torch.ones((1, width), dtype=feature.dtype)
    component = component / component.norm(dim=1, keepdim=True)
    return LOOKArtifact(
        node_name=site,
        missing_pattern=missing_pattern,
        filling_strategy="embracenet_v2_synthetic_nonzero",
        factor=factor,
        latent_dim=1,
        feature_shape=tuple(feature.shape[1:]),
        downsample_shape=down_shape,
        mean=torch.zeros(width, dtype=feature.dtype),
        std=torch.ones(width, dtype=feature.dtype),
        pca_mean=torch.zeros(1, dtype=feature.dtype),
        components=component,
        weight=torch.zeros((1, 1), dtype=feature.dtype),
        bias=torch.tensor([bias_value], dtype=feature.dtype),
        ridge_lambda=0.0,
        train_r2=0.0,
        train_mse=0.0,
        member_names=members(site),
        member_shapes=member_shapes(graph, site),
    )


def _noop_artifact(artifact):
    result = copy.deepcopy(artifact)
    result.bias.zero_()
    result.weight.zero_()
    return result


def _bn_buffers(graph):
    return {
        key: value.detach().clone()
        for key, value in graph.state_dict().items()
        if key.endswith("running_mean") or key.endswith("running_var") or key.endswith("num_batches_tracked")
    }


def test_shared_missing_input_contract_mixed_train_batch_bn_and_gradients():
    torch.set_num_threads(2)
    first, second = _matched_graphs(embracement_size=24, sampling_seed=77)
    first.train()
    second.train()
    counts = [1, 1, 1]
    availability = torch.tensor([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    labels = torch.tensor([0, 1, 0])
    oct_base = torch.randn(3, 3, 224, 224)
    cfp_base = torch.randn(3, 3, 224, 224)
    oct_changed = oct_base.clone()
    cfp_changed = cfp_base.clone()
    oct_changed[1].fill_(float("nan"))
    cfp_changed[2].fill_(float("nan"))

    before = _bn_buffers(first)
    oct_a = oct_base.clone().requires_grad_(True)
    cfp_a = cfp_base.clone().requires_grad_(True)
    oct_b = oct_changed.clone().requires_grad_(True)
    cfp_b = cfp_changed.clone().requires_grad_(True)
    logits_a = forward_embracenet_host(first, oct_a, cfp_a, counts, availability, labels=labels)
    logits_b = forward_embracenet_host(second, oct_b, cfp_b, counts, availability, labels=labels)
    torch.testing.assert_close(logits_a, logits_b, rtol=0, atol=0)
    torch.testing.assert_close(
        first.get_node_by_name("loss").feature_message.current_state,
        second.get_node_by_name("loss").feature_message.current_state,
        rtol=0,
        atol=0,
    )
    after_a, after_b = _bn_buffers(first), _bn_buffers(second)
    assert set(after_a) == set(after_b) == set(before)
    assert all(torch.equal(after_a[key], after_b[key]) for key in after_a)
    assert any(not torch.equal(before[key], after_a[key]) for key in before)

    first.get_node_by_name("loss").feature_message.current_state.backward()
    second.get_node_by_name("loss").feature_message.current_state.backward()
    assert oct_a.grad is not None and torch.count_nonzero(oct_a.grad[1]) == 0
    assert cfp_a.grad is not None and torch.count_nonzero(cfp_a.grad[2]) == 0
    assert oct_b.grad is not None and torch.count_nonzero(oct_b.grad[1]) == 0
    assert cfp_b.grad is not None and torch.count_nonzero(cfp_b.grad[2]) == 0
    assert torch.isfinite(oct_b.grad).all() and torch.isfinite(cfp_b.grad).all()


def test_plain_A_and_A_plus_LOOK_entrypoints_match_without_artifact_eval_and_train():
    torch.set_num_threads(2)
    counts = [1, 1, 1]
    availability = torch.tensor([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    oct_tensor = torch.randn(3, 3, 224, 224)
    cfp_tensor = torch.randn(3, 3, 224, 224)
    oct_tensor[1].fill_(float("nan"))
    cfp_tensor[2].fill_(float("nan"))

    for training in (False, True):
        first, second = _matched_graphs(embracement_size=24, sampling_seed=91)
        first.train(training)
        second.train(training)
        direct = forward_embracenet_host(first, oct_tensor, cfp_tensor, counts, availability)
        via_study = forward_embracenet_with_look(
            second, oct_tensor, cfp_tensor, counts, availability
        )["output"]
        torch.testing.assert_close(direct, via_study, rtol=0, atol=0)
        first_bn, second_bn = _bn_buffers(first), _bn_buffers(second)
        assert all(torch.equal(first_bn[key], second_bn[key]) for key in first_bn)


def test_prepare_shared_contract_masks_both_single_missing_modalities_before_encoder():
    counts = [1, 2]
    availability = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    oct_tensor = torch.randn(3, 3, 8, 8, requires_grad=True)
    cfp_tensor = torch.randn(3, 3, 8, 8, requires_grad=True)
    oct_tensor = oct_tensor.clone()
    cfp_tensor = cfp_tensor.clone()
    oct_tensor[0].fill_(float("nan"))
    cfp_tensor[1:].fill_(float("nan"))
    safe_oct, safe_cfp = prepare_embracenet_inputs(oct_tensor, cfp_tensor, counts, availability)
    assert torch.isfinite(safe_oct).all() and torch.isfinite(safe_cfp).all()
    assert torch.count_nonzero(safe_oct[0]) == 0
    assert torch.count_nonzero(safe_cfp[1:]) == 0


def test_replay_validation_zero_probability_and_fail_clean_contract():
    module = EmbraceNetFusion([4, 4], embracement_size=6, sampling_seed=12)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    with pytest.raises(TypeError, match="integer dtype"):
        module.set_replay_indices(torch.zeros(2, 6, dtype=torch.float32))
    with pytest.raises(ValueError, match="shape"):
        module.set_replay_indices(torch.zeros(2, 5, dtype=torch.long))
    with pytest.raises(ValueError, match="invalid modality"):
        module.set_replay_indices(torch.full((2, 6), 2, dtype=torch.long))

    zero_probability_replay = torch.ones(2, 6, dtype=torch.long)
    module.set_replay_indices(zero_probability_replay)
    with pytest.raises(ValueError, match="zero-probability"):
        module([x, y], torch.ones(2, 2), torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    valid = module([x, y], torch.ones(2, 2))
    assert valid.shape == (2, 6)
    assert module.peek_last_trace()["replayed"] is False

    state = module.get_sampling_state()
    module.set_replay_indices(torch.zeros(2, 6, dtype=torch.long))
    module.set_sampling_state(state)
    module([x, y], torch.ones(2, 2))
    assert module.peek_last_trace()["replayed"] is False

    module.set_replay_indices(torch.zeros(2, 6, dtype=torch.long))
    with pytest.raises(ValueError, match="non-finite"):
        module([torch.full_like(x, float("nan")), y], torch.ones(2, 2))
    module([x, y], torch.ones(2, 2))
    assert module.peek_last_trace()["replayed"] is False


def test_artifact_validation_rejects_duplicate_invalid_order_and_stop_overrun_without_replay_leak():
    torch.set_num_threads(2)
    graph = build_embracenet_host(*_parents(), embracement_size=8, sampling_seed=5).eval()
    counts = [1]
    availability = torch.tensor([[0.0, 1.0]])
    oct_tensor = torch.full((1, 3, 224, 224), float("nan"))
    cfp_tensor = torch.randn(1, 3, 224, 224)
    site = "joint_stage1"
    feature = forward_embracenet_with_look(
        graph, oct_tensor, cfp_tensor, counts, availability, stop_node=site
    )["output"]
    artifact = _nonzero_artifact(graph, site, feature, "missing_oct")
    with pytest.raises(ValueError, match="Duplicate"):
        forward_embracenet_with_look(
            graph, oct_tensor, cfp_tensor, counts, availability, [artifact, copy.deepcopy(artifact)]
        )
    invalid = copy.deepcopy(artifact)
    invalid.node_name = "fusion_logits"
    with pytest.raises(ValueError, match="Invalid LOOK"):
        forward_embracenet_with_look(
            graph, oct_tensor, cfp_tensor, counts, availability, [invalid]
        )
    later_feature = forward_embracenet_with_look(
        graph, oct_tensor, cfp_tensor, counts, availability, stop_node="joint_stage2"
    )["output"]
    later = _nonzero_artifact(graph, "joint_stage2", later_feature, "missing_oct")
    with pytest.raises(ValueError, match="forward order"):
        forward_embracenet_with_look(
            graph, oct_tensor, cfp_tensor, counts, availability, [later, artifact]
        )
    with pytest.raises(ValueError, match="stop boundary"):
        forward_embracenet_with_look(
            graph, oct_tensor, cfp_tensor, counts, availability, [later], stop_node=site
        )
    with pytest.raises(ValueError, match="stop node"):
        forward_embracenet_with_look(
            graph,
            oct_tensor,
            cfp_tensor,
            counts,
            availability,
            stop_node=site,
            replay_indices=torch.ones(1, 8, dtype=torch.long),
        )
    # A rejected pre-embrace exact replay must not remain armed.
    baseline = forward_embracenet_with_look(graph, oct_tensor, cfp_tensor, counts, availability)
    assert baseline["sampling_trace"]["replayed"] is False


@pytest.mark.parametrize("missing_pattern,availability", [
    ("missing_oct", torch.tensor([[0.0, 1.0]])),
    ("missing_cfp", torch.tensor([[1.0, 0.0]])),
])
def test_nonzero_and_noop_real_LOOK_artifact_all_nine_sites_downstream_and_resume(missing_pattern, availability):
    torch.set_num_threads(2)
    graph = build_embracenet_host(*_parents(), embracement_size=12, sampling_seed=101).eval()
    counts = [1]
    oct_tensor = torch.randn(1, 3, 224, 224)
    cfp_tensor = torch.randn(1, 3, 224, 224)
    if missing_pattern == "missing_oct":
        oct_tensor.fill_(float("nan"))
    else:
        cfp_tensor.fill_(float("nan"))

    for site in correction_sites(graph):
        state = capture_embracenet_sampling(graph)
        site_base = forward_embracenet_with_look(
            graph, oct_tensor, cfp_tensor, counts, availability, sampling_state=state, stop_node=site
        )["output"].detach().clone()
        nonzero = _nonzero_artifact(graph, site, site_base, missing_pattern)
        noop = _noop_artifact(nonzero)

        site_changed = forward_embracenet_with_look(
            graph, oct_tensor, cfp_tensor, counts, availability, [nonzero],
            sampling_state=state, stop_node=site
        )["output"].detach()
        assert not torch.equal(site_base, site_changed), site

        baseline = forward_embracenet_with_look(
            graph, oct_tensor, cfp_tensor, counts, availability, sampling_state=state
        )
        trace = baseline["sampling_trace"]["modality_indices"]
        nooped = forward_embracenet_with_look(
            graph, oct_tensor, cfp_tensor, counts, availability, [noop],
            replay_indices=trace
        )
        changed = forward_embracenet_with_look(
            graph, oct_tensor, cfp_tensor, counts, availability, [nonzero],
            replay_indices=trace
        )
        torch.testing.assert_close(baseline["output"], nooped["output"], rtol=0, atol=0)
        assert not torch.equal(baseline["output"], changed["output"]), site

        restored_first = forward_embracenet_with_look(
            graph, oct_tensor, cfp_tensor, counts, availability, [nonzero], sampling_state=state
        )
        restored_second = forward_embracenet_with_look(
            graph, oct_tensor, cfp_tensor, counts, availability, [nonzero], sampling_state=state
        )
        torch.testing.assert_close(restored_first["output"], restored_second["output"], rtol=0, atol=0)
        torch.testing.assert_close(
            restored_first["sampling_trace"]["modality_indices"],
            restored_second["sampling_trace"]["modality_indices"],
            rtol=0,
            atol=0,
        )


def test_both_single_missing_states_are_draw_invariant_complete_state_remains_random():
    torch.set_num_threads(2)
    base = build_embracenet_host(*_parents(), embracement_size=32, sampling_seed=1).eval()
    weights = copy.deepcopy(base.state_dict())
    oct_tensor = torch.randn(1, 3, 224, 224)
    cfp_tensor = torch.randn(1, 3, 224, 224)
    counts = [1]
    outputs = {}
    for name, availability in (
        ("missing_oct", torch.tensor([[0.0, 1.0]])),
        ("missing_cfp", torch.tensor([[1.0, 0.0]])),
    ):
        values = []
        look_values = []
        for seed in (1, 9, 27):
            graph = build_embracenet_host(*_parents(), embracement_size=32, sampling_seed=seed).eval()
            graph.load_state_dict(weights, strict=True)
            missing_oct = oct_tensor.clone()
            missing_cfp = cfp_tensor.clone()
            if name == "missing_oct":
                missing_oct.fill_(float("nan"))
            else:
                missing_cfp.fill_(float("nan"))
            baseline = forward_embracenet_with_look(
                graph, missing_oct, missing_cfp, counts, availability
            )
            values.append(baseline["output"].detach())
            embraced = forward_embracenet_with_look(
                graph, missing_oct, missing_cfp, counts, availability,
                stop_node="embraced_feature"
            )["output"].detach()
            artifact = _nonzero_artifact(graph, "embraced_feature", embraced, name)
            corrected = forward_embracenet_with_look(
                graph, missing_oct, missing_cfp, counts, availability, [artifact]
            )
            look_values.append(corrected["output"].detach())
            expected_index = 1 if name == "missing_oct" else 0
            assert torch.all(baseline["sampling_trace"]["modality_indices"] == expected_index)
        for value in values[1:]:
            torch.testing.assert_close(values[0], value, rtol=0, atol=0)
        for value in look_values[1:]:
            torch.testing.assert_close(look_values[0], value, rtol=0, atol=0)
        outputs[name] = values[0]

    complete_values = []
    for seed in (1, 9, 27):
        graph = build_embracenet_host(*_parents(), embracement_size=32, sampling_seed=seed).eval()
        graph.load_state_dict(weights, strict=True)
        complete_values.append(
            forward_embracenet_with_look(
                graph, oct_tensor, cfp_tensor, counts, torch.tensor([[1.0, 1.0]])
            )["output"].detach()
        )
    assert any(not torch.equal(complete_values[0], value) for value in complete_values[1:])


def test_analytical_moments_single_missing_zero_variance_and_linear_mean_logit_identity():
    torch.manual_seed(123)
    module = EmbraceNetFusion([5, 5], embracement_size=16, sampling_seed=7).eval()
    first, second = torch.randn(3, 5), torch.randn(3, 5)
    for availability in (
        torch.tensor([[1.0, 0.0]] * 3),
        torch.tensor([[0.0, 1.0]] * 3),
    ):
        moments = module.analytical_moments([first, second], availability)
        assert torch.count_nonzero(moments["variance_diagonal"]) == 0
        for seed in (1, 2, 3):
            module.sampling_seed = seed
            module.set_sampling_state({
                "schema": "look_embracenet_sampling_v1",
                "seed": seed,
                "device": None,
                "state": None,
                "draw_count": 0,
            })
            sampled = module([first, second], availability)
            torch.testing.assert_close(sampled, moments["mean"], rtol=0, atol=0)

    complete = module.analytical_moments([first, second], torch.ones(3, 2))
    assert torch.any(complete["variance_diagonal"] > 0)
    head = torch.nn.Linear(16, 2).eval()
    mean_logits = head(complete["mean"])
    manual = complete["mean"] @ head.weight.T + head.bias
    torch.testing.assert_close(mean_logits, manual, rtol=0, atol=0)
