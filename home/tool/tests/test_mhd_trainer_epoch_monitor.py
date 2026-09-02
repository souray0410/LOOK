from __future__ import annotations

import torch
import torch.nn as nn

from MHD_Project.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
from MHD_Project.MHD_Utils_V4 import (
    MHD_Monitor,
    MHD_Trainer,
)


class AccuracyVector(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (logits.argmax(dim=1) == target.long()).float()


class MacroF1(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = logits.argmax(dim=1)
        values = []
        for class_id in range(2):
            predicted = prediction == class_id
            expected = target == class_id
            true_positive = torch.logical_and(predicted, expected).sum().float()
            false_positive = torch.logical_and(predicted, ~expected).sum().float()
            false_negative = torch.logical_and(~predicted, expected).sum().float()
            denominator = 2 * true_positive + false_positive + false_negative
            values.append(
                torch.where(
                    denominator > 0,
                    2 * true_positive / denominator,
                    torch.zeros_like(denominator),
                )
            )
        return torch.stack(values).mean()


def _node(node_id: int, name: str, state: torch.Tensor) -> MHD_Node:
    return MHD_Node(
        node_id,
        name,
        MHD_Node.Message(state),
        aggregation="replace",
    )


def _graph() -> MHD_Graph:
    nodes = {
        _node(0, "input", torch.zeros(2, 2)),
        _node(1, "logits", torch.zeros(2, 2)),
        _node(2, "target", torch.zeros(2, dtype=torch.long)),
        _node(3, "loss", torch.zeros(())),
        _node(4, "batch_accuracy", torch.zeros(2)),
        _node(5, "validation_macro_f1", torch.zeros(())),
    }
    linear = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(2))
    edges = {
        MHD_Edge(0, "linear", [MHD_Edge.Operation(linear)]),
        MHD_Edge(1, "loss", [MHD_Edge.Operation(nn.CrossEntropyLoss())]),
        MHD_Edge(2, "accuracy", [MHD_Edge.Operation(AccuracyVector())]),
        MHD_Edge(3, "macro_f1", [MHD_Edge.Operation(MacroF1())]),
    }
    role_0 = torch.tensor([
        [-1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=torch.int8)
    sort_0 = torch.tensor([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=torch.int8)
    role_1 = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [0, -1, -1, 1, 0, 0],
        [0, -1, -1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=torch.int8)
    sort_1 = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 1, 0, 2, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=torch.int8)
    role_2 = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, -1, -1, 0, 0, 1],
    ], dtype=torch.int8)
    sort_2 = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 2],
    ], dtype=torch.int8)
    backward_0 = -role_0
    backward_1 = -role_1
    backward_1[2].zero_()
    return MHD_Graph(
        nodes,
        edges,
        {MHD_Topo(
            [role_0, role_1, role_2, backward_0, backward_1],
            [sort_0, sort_1, sort_2, sort_0.clone(), sort_1.clone()],
        )},
        device=torch.device("cpu"),
    )


def test_epoch_monitor_uses_the_complete_validation_set_and_saves(tmp_path):
    graph = _graph()
    optimizer = torch.optim.SGD(graph.parameters(), lr=0.1)
    monitor = MHD_Monitor(["batch_accuracy"])
    monitor.register_epoch_node(
        "validation_macro_f1",
        source_nodes=["logits", "target"],
        levels=[2],
    )
    trainer = MHD_Trainer(
        graph,
        optimizer,
        monitor,
        forward_levels=[0, 1],
        backward_levels=[4, 3],
        criteria_node="validation_macro_f1",
        criteria_mode="max",
        save_dir=str(tmp_path),
        input_nodes=["input", "target"],
        input_mapping={"input": "features", "target": "labels"},
        output_nodes=["logits", "loss", "batch_accuracy", "target"],
    )
    evaluation = [
        {
            "features": torch.tensor([[8.0, 0.0], [8.0, 0.0]]),
            "labels": torch.tensor([0, 0]),
            "participant_id": ["a", "b"],
        },
        {
            "features": torch.tensor([[8.0, 0.0], [8.0, 0.0]]),
            "labels": torch.tensor([1, 1]),
            "participant_id": ["c", "d"],
        },
    ]

    metrics = trainer.eval_epoch(evaluation, epoch=0)

    assert metrics["validation_macro_f1"] == torch.tensor(1 / 3).item()
    assert trainer.history["best_epoch"] == 1
    assert trainer.last_eval_epoch_tensors["logits"].shape == (4, 2)
    assert (tmp_path / "best").is_dir()

    trainer.save_last_checkpoint(1)
    expected = next(graph.parameters()).detach().clone()
    with torch.no_grad():
        next(graph.parameters()).zero_()
    restored_epoch = trainer.load_checkpoint(load_last=True)
    assert restored_epoch == 1
    assert torch.equal(next(graph.parameters()), expected)
