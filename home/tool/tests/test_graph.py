import pytest
import torch

from look_core.graph import FUSION_POSITIONS, ConcatProjection, build_resnet50_mhd_graph, reset_and_forward


@pytest.mark.parametrize("position", FUSION_POSITIONS)
def test_topology_output_and_pruned_state(position):
    graph = build_resnet50_mhd_graph(position, batch_size=1, pretrained=False, device="cpu")
    graph.eval()
    with torch.no_grad():
        logits = reset_and_forward(
            graph, torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)
        )
    assert logits.shape == (1, 5)
    active_edges = {edge.name for edge in graph.edges}
    state_keys = list(graph.state_dict())
    assert all(any(edge_name in key for edge_name in active_edges) for key in state_keys)
    assert graph.get_node_by_name("fusion_logits") is not None


def test_branch_weights_are_equal_but_not_shared_and_fusion_is_average_identity():
    graph = build_resnet50_mhd_graph("layer1", batch_size=1, pretrained=False, device="cpu")
    oct_stem = graph.get_edge_by_name("oct_stem_edge").edge_operations[0][0].weight
    cfp_stem = graph.get_edge_by_name("cfp_stem_edge").edge_operations[0][0].weight
    assert torch.equal(oct_stem, cfp_stem)
    assert oct_stem.data_ptr() != cfp_stem.data_ptr()

    projection = graph.get_edge_by_name("fuse_layer1_edge").edge_operations[0]
    first, second = torch.randn(2, 256, 4, 4), torch.randn(2, 256, 4, 4)
    with torch.no_grad():
        output = projection(first, second)
    assert torch.allclose(output, (first + second) / 2, atol=1e-6)


def test_graph_internal_loss_is_differentiable_and_diagnostic_is_detached():
    graph = build_resnet50_mhd_graph("feature", batch_size=1, pretrained=False, device="cpu")
    outputs = reset_and_forward(
        graph,
        torch.randn(1, 3, 224, 224),
        torch.randn(1, 3, 224, 224),
        torch.tensor([2]),
    )
    assert outputs["loss"].requires_grad
    assert not outputs["batch_accuracy"].requires_grad
    outputs["loss"].backward()
    classifier = graph.get_edge_by_name("fusion_classifier_edge").edge_operations[0]
    assert classifier.weight.grad is not None
    assert torch.isfinite(classifier.weight.grad).all()
