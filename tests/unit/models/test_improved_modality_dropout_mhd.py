import torch
from mhd_framework.models import create_model

from look.models.improved_modality_dropout import (
    build_improved_dropout_host,
    forward_improved_dropout_host,
    improved_dropout_parameter_groups,
)
from look.models.observed_participant import ObservedParticipantModel


def _parents():
    config = dict(name="resnet18", num_classes=2, views=1)
    return [ObservedParticipantModel(create_model(config)) for _ in range(2)]


def test_imd_mhd_forward_masks_missing_modality_and_keeps_encoders_frozen():
    torch.set_num_threads(2)
    graph = build_improved_dropout_host(*_parents(), hidden_dropout=0, classifier_dropout=0)
    assert graph.improved_dropout_provenance["author_commit"] == "8040d96b2dec48cf8fc7d13b45e15af0d07952ed"
    assert "oct_participant_feature" in graph.correction_nodes
    assert "cfp_participant_feature" in graph.correction_nodes
    assert graph.get_node_by_name("imd_state_code") is not None
    pretrained = [p for edge in graph.edges if edge.name.endswith("features_edge") for op in edge.edge_operations for p in op.function.parameters()]
    assert pretrained and all(not p.requires_grad for p in pretrained)
    trainable = [p for p in graph.parameters() if p.requires_grad]
    pretrained_ids = {id(p) for p in pretrained}
    assert trainable and all(id(p) not in pretrained_ids for p in trainable)
    groups = improved_dropout_parameter_groups(graph, 1e-3)
    assert groups[0]["group_name"] == "improved_dropout_fusion_and_head"

    oct_tensor = torch.randn(2, 3, 224, 224)
    cfp_tensor = torch.randn(2, 3, 224, 224)
    counts = [1, 1]
    complete = forward_improved_dropout_host(graph, oct_tensor, cfp_tensor, counts, state="complete")
    oct_missing = forward_improved_dropout_host(graph, torch.full_like(oct_tensor, 99), cfp_tensor, counts, state="oct_missing")
    oct_missing_again = forward_improved_dropout_host(graph, torch.full_like(oct_tensor, -99), cfp_tensor, counts, state="oct_missing")
    assert complete.shape == (2, 2)
    torch.testing.assert_close(oct_missing, oct_missing_again, rtol=0, atol=0)


def test_imd_mhd_vector_state_codes_and_loss_backward_only_fusion_params():
    torch.set_num_threads(2)
    graph = build_improved_dropout_host(*_parents(), hidden_dropout=0, classifier_dropout=0)
    labels = torch.tensor([0, 1])
    logits = forward_improved_dropout_host(
        graph, torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224), [1, 1],
        labels=labels, state=torch.tensor([1, 2])
    )
    loss = graph.get_node_by_name("loss").feature_message.current_state
    loss.backward()
    assert logits.shape == (2, 2)
    trainable = [p for p in graph.parameters() if p.requires_grad]
    assert any(p.grad is not None for p in trainable)
    frozen = [p for edge in graph.edges if edge.name.endswith("features_edge") for op in edge.edge_operations for p in op.function.parameters()]
    assert all(p.grad is None for p in frozen)
