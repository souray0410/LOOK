import pytest
import torch

from look_core.graph import (
    FUSION_POSITIONS,
    ClassificationLoss,
    build_resnet50_mhd_graph,
    classification_loss_metadata,
    reset_and_forward,
)


@pytest.mark.parametrize("position", FUSION_POSITIONS)
def test_topology_output_and_pruned_state(position):
    graph = build_resnet50_mhd_graph(position, batch_size=1, pretrained=False, device="cpu")
    graph.eval()
    with torch.no_grad():
        logits = reset_and_forward(
            graph, torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)
        )
    assert logits.shape == (1, 5)
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


def test_class_balanced_loss_weights_and_metadata_are_inside_loss_edge():
    counts = [1000, 100, 50, 20, 10]
    graph = build_resnet50_mhd_graph(
        "feature",
        batch_size=1,
        pretrained=False,
        class_counts=counts,
        class_balance_beta=0.999,
        label_smoothing=0.05,
    )
    operation = graph.get_edge_by_name("classification_loss_edge").edge_operations[0].function
    assert isinstance(operation, ClassificationLoss)
    expected = (1.0 - 0.999) / (1.0 - torch.pow(torch.full((5,), 0.999), torch.tensor(counts)))
    expected = expected / expected.mean()
    assert torch.allclose(operation.class_weights, expected)
    metadata = classification_loss_metadata(graph)
    assert metadata["class_counts"] == counts
    assert metadata["label_smoothing"] == 0.05


def test_graph_internal_loss_and_backward_messages_are_differentiable():
    graph = build_resnet50_mhd_graph("feature", batch_size=1, pretrained=False, device="cpu")
    outputs = reset_and_forward(
        graph,
        torch.randn(1, 3, 224, 224),
        torch.randn(1, 3, 224, 224),
        torch.tensor([2]),
    )
    assert outputs["loss"].requires_grad
    assert not outputs["batch_accuracy"].requires_grad
    graph.backward({"loss": None})
    classifier = graph.get_edge_by_name("fusion_classifier_edge").edge_operations[0].function
    assert classifier.linear.weight.grad is not None
    assert torch.isfinite(classifier.linear.weight.grad).all()
    logits_gradient = graph.get_node_by_name("fusion_logits").gradient_message.current_state
    assert logits_gradient.shape == outputs["fusion_logits"].shape
    assert torch.isfinite(logits_gradient).all()
    assert graph.get_node_by_name("loss").gradient_message.current_state.item() == 1.0
