import pytest
import torch
from torchvision.models import ResNet50_Weights, resnet50

import look_core.graph as graph_module

from look_core.graph import (
    BilateralMean,
    FUSION_POSITIONS,
    UNIMODAL_POSITIONS,
    ClassificationLoss,
    build_resnet50_mhd_graph,
    classification_loss_metadata,
    reset_and_forward,
)
from look_core.metrics import (
    classification_metrics,
    validation_binary_auroc,
    validation_macro_f1,
)


@pytest.mark.parametrize("position", FUSION_POSITIONS)
def test_topology_output_and_pruned_state(position):
    graph = build_resnet50_mhd_graph(position, batch_size=1, pretrained=False, device="cpu")
    graph.eval()
    with torch.no_grad():
        logits = reset_and_forward(
            graph, torch.randn(1, 2, 3, 224, 224), torch.randn(1, 2, 3, 224, 224)
        )
    assert logits.shape == (1, 2)
    active_modules = {
        id(operation.function)
        for edge in graph.edges
        for operation in edge.edge_operations
        if isinstance(operation.function, torch.nn.Module)
    }
    registered_modules = {id(module) for module in graph.edge_module_map.values()}
    assert registered_modules == active_modules
    assert graph.state_dict()
    assert all(key.startswith("edge_module_map.") for key in graph.state_dict())
    assert graph.get_node_by_name("fusion_logits") is not None


def test_branch_weights_are_equal_but_not_shared_and_fusion_is_average_identity():
    graph = build_resnet50_mhd_graph("layer1", batch_size=1, pretrained=False, device="cpu")
    oct_stem = graph.get_edge_by_name("oct_stem_edge").edge_operations[0].function[0].weight
    cfp_stem = graph.get_edge_by_name("cfp_stem_edge").edge_operations[0].function[0].weight
    assert torch.equal(oct_stem, cfp_stem)
    assert oct_stem.data_ptr() != cfp_stem.data_ptr()

    projection = graph.get_edge_by_name("fuse_layer1_edge").edge_operations[0].function
    first, second = torch.randn(2, 256, 4, 4), torch.randn(2, 256, 4, 4)
    with torch.no_grad():
        projected = projection.projection(torch.cat([first, second], dim=1))
    assert torch.allclose(projected, (first + second) / 2, atol=1e-6)
    assert not any(isinstance(module, (torch.nn.ReLU, torch.nn.GELU)) for module in projection.modules())


def test_bilateral_mean_is_linear_parameter_free_permutation_invariant_and_propagates():
    operation = BilateralMean()
    eyes = torch.tensor([[1.0, 3.0], [3.0, 5.0], [2.0, 4.0], [6.0, 8.0]], requires_grad=True)
    participants = operation(eyes)
    assert torch.equal(
        participants,
        torch.tensor([[2.0, 4.0], [4.0, 6.0]]),
    )
    swapped = operation(eyes.reshape(2, 2, 2).flip(1).reshape(4, 2))
    assert torch.equal(participants, swapped)
    participants.sum().backward()
    assert torch.isfinite(eyes.grad).all()
    assert torch.all(eyes.grad > 0)
    assert list(operation.parameters()) == []


def test_imagenet_template_weights_are_requested_and_mapped_into_hyperedges(monkeypatch):
    template = resnet50(weights=None)
    with torch.no_grad():
        template.conv1.weight.fill_(0.125)
    requested = []

    def fake_resnet50(*, weights):
        requested.append(weights)
        return template

    monkeypatch.setattr(graph_module, "resnet50", fake_resnet50)
    graph = build_resnet50_mhd_graph("feature", batch_size=1, pretrained=True)
    oct_weight = graph.get_edge_by_name("oct_stem_edge").edge_operations[0].function[0].weight
    cfp_weight = graph.get_edge_by_name("cfp_stem_edge").edge_operations[0].function[0].weight
    assert requested == [ResNet50_Weights.IMAGENET1K_V2]
    assert torch.all(oct_weight == 0.125)
    assert torch.equal(oct_weight, cfp_weight)
    assert oct_weight.data_ptr() != cfp_weight.data_ptr()


def test_cross_entropy_and_metadata_are_inside_loss_edge():
    graph = build_resnet50_mhd_graph(
        "feature", batch_size=1, pretrained=False, label_smoothing=0.1
    )
    operation = graph.get_edge_by_name("classification_loss_edge").edge_operations[0].function
    assert isinstance(operation, ClassificationLoss)
    logits = torch.tensor([[1.0, -0.5]])
    labels = torch.tensor([1])
    expected = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.1)
    assert torch.allclose(operation(logits, labels), expected)
    metadata = classification_loss_metadata(graph)
    assert metadata["name"] == "cross_entropy"
    assert metadata["class_weights"] is None
    assert metadata["label_smoothing"] == 0.1
    assert metadata["inference"] == "raw_logits_standard_softmax"


def test_validation_macro_f1_is_an_exact_task_criterion():
    graph = build_resnet50_mhd_graph("feature", batch_size=6, pretrained=False)
    logits = torch.tensor([
        [8.0, 0.0],
        [8.0, 0.0],
        [8.0, 0.0],
        [0.0, 8.0],
        [0.0, 8.0],
        [0.0, 8.0],
    ])
    labels = torch.tensor([0, 0, 1, 1, 1, 0])
    graph.get_node_by_name("fusion_logits").feature_message.current_state = logits
    graph.get_node_by_name("label_gt").feature_message.current_state = labels
    expected = classification_metrics(
        labels.numpy(), torch.softmax(logits, dim=1).numpy()
    )["macro_f1"]
    assert validation_macro_f1(graph) == pytest.approx(expected)


def test_validation_binary_auroc_matches_sklearn():
    graph = build_resnet50_mhd_graph("feature", batch_size=6, pretrained=False)
    logits = torch.tensor([
        [3.0, 0.0], [2.0, 0.0], [0.0, 3.0],
        [0.0, 2.0], [0.5, 0.5], [0.0, 1.0],
    ])
    labels = torch.tensor([0, 0, 1, 1, 0, 1])
    graph.get_node_by_name("fusion_logits").feature_message.current_state = logits
    graph.get_node_by_name("label_gt").feature_message.current_state = labels
    expected = classification_metrics(
        labels.numpy(), torch.softmax(logits, dim=1).numpy()
    )["macro_auroc_ovr"]
    assert validation_binary_auroc(graph).item() == pytest.approx(expected)


def test_graph_internal_loss_and_backward_messages_are_differentiable():
    graph = build_resnet50_mhd_graph("feature", batch_size=1, pretrained=False, device="cpu")
    outputs = reset_and_forward(
        graph,
        torch.randn(1, 2, 3, 224, 224),
        torch.randn(1, 2, 3, 224, 224),
        torch.tensor([1]),
    )
    assert outputs["loss"].requires_grad
    assert not outputs["batch_accuracy"].requires_grad
    assert outputs["batch_accuracy"].shape == (1,)
    assert set(graph.forward_levels).isdisjoint(graph.backward_levels)
    assert graph.get_node_by_name("validation_macro_f1") is None
    assert graph.get_edge_by_name("validation_macro_f1_edge") is None
    graph.backward(levels=graph.backward_levels)
    classifier = graph.get_edge_by_name("fusion_classifier_edge").edge_operations[0].function
    assert classifier.linear.weight.grad is not None
    assert torch.isfinite(classifier.linear.weight.grad).all()
    logits_gradient = graph.get_node_by_name("fusion_logits").gradient_message.current_state
    assert logits_gradient.shape == outputs["fusion_logits"].shape
    assert torch.isfinite(logits_gradient).all()
    assert graph.get_node_by_name("loss").gradient_message.current_state.item() == 1.0


@pytest.mark.parametrize("position", UNIMODAL_POSITIONS)
def test_unimodal_reference_uses_one_pretrained_branch(position):
    graph = build_resnet50_mhd_graph(position, batch_size=1, pretrained=False)
    branch = position.removesuffix("_only")
    assert graph.get_edge_by_name(f"{branch}_stem_edge") is not None
    other = "cfp" if branch == "oct" else "oct"
    assert graph.get_edge_by_name(f"{other}_stem_edge") is None
    assert graph.get_edge_by_name("fusion_classifier_edge") is not None
    with torch.no_grad():
        logits = reset_and_forward(
            graph, torch.randn(1, 2, 3, 224, 224), torch.randn(1, 2, 3, 224, 224)
        )
    assert logits.shape == (1, 2)
