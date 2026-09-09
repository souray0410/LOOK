from __future__ import annotations

import torch
import torch.nn as nn

from V4.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
from V4.MHD_Utils_V4 import MHD_Monitor, MHD_Trainer


class AccuracyVector(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (logits.argmax(dim=1) == target.long()).float()


def validation_macro_f1(graph: MHD_Graph) -> torch.Tensor:
    logits = graph.get_node_by_name("logits").feature_message.current_state
    target = graph.get_node_by_name("target").feature_message.current_state
    prediction = logits.argmax(dim=1)
    values = []
    for class_id in range(2):
        predicted = prediction == class_id
        expected = target == class_id
        true_positive = torch.logical_and(predicted, expected).sum().float()
        false_positive = torch.logical_and(predicted, ~expected).sum().float()
        false_negative = torch.logical_and(~predicted, expected).sum().float()
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(torch.where(
            denominator > 0,
            2 * true_positive / denominator,
            torch.zeros_like(denominator),
        ))
    return torch.stack(values).mean()


def validation_loss(graph: MHD_Graph) -> torch.Tensor:
    return graph.get_node_by_name("loss").feature_message.current_state.mean()


def _node(node_id: int, name: str, state: torch.Tensor) -> MHD_Node:
    return MHD_Node(node_id, name, MHD_Node.Message(state), aggregation="replace")


def _graph() -> MHD_Graph:
    nodes = {
        _node(0, "input", torch.zeros(2, 2)),
        _node(1, "logits", torch.zeros(2, 2)),
        _node(2, "target", torch.zeros(2, dtype=torch.long)),
        _node(3, "loss", torch.zeros(())),
        _node(4, "batch_accuracy", torch.zeros(2)),
    }
    linear = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(2))
    edges = {
        MHD_Edge(0, "linear", [MHD_Edge.Operation(linear)]),
        MHD_Edge(1, "loss", [MHD_Edge.Operation(nn.CrossEntropyLoss())]),
        MHD_Edge(2, "accuracy", [MHD_Edge.Operation(AccuracyVector())]),
    }
    role_0 = torch.tensor([
        [-1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.int8)
    sort_0 = torch.tensor([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.int8)
    role_1 = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, -1, -1, 1, 0],
        [0, -1, -1, 0, 1],
    ], dtype=torch.int8)
    sort_1 = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 2, 0],
        [0, 0, 1, 0, 2],
    ], dtype=torch.int8)
    backward_0 = -role_0
    backward_1 = -role_1
    backward_1[2].zero_()
    return MHD_Graph(
        nodes,
        edges,
        {MHD_Topo(
            [role_0, role_1, backward_0, backward_1],
            [sort_0, sort_1, sort_0.clone(), sort_1.clone()],
        )},
        device=torch.device("cpu"),
    )


def test_criteria_uses_complete_validation_graph_state_and_saves(tmp_path):
    graph = _graph()
    trainer = MHD_Trainer(
        graph,
        torch.optim.SGD(graph.parameters(), lr=0.1),
        MHD_Monitor(["batch_accuracy"]),
        forward_levels=[0, 1],
        backward_levels=[3, 2],
        criteria=validation_macro_f1,
        criteria_mode="max",
        save_dir=str(tmp_path),
        input_nodes=["input", "target"],
        input_mapping={"input": "features", "target": "labels"},
        output_nodes=["logits", "loss", "batch_accuracy", "target"],
    )
    evaluation = [
        {
            "features": torch.tensor([[8.0, 0.0], [8.0, 0.0], [8.0, 0.0]]),
            "labels": torch.tensor([0, 0, 0]),
        },
        {
            "features": torch.tensor([[8.0, 0.0]]),
            "labels": torch.tensor([1]),
        },
    ]

    metrics = trainer.eval_epoch(evaluation, epoch=0)

    assert metrics["validation_macro_f1"] == torch.tensor(3 / 7).item()
    assert trainer.history["best_epoch"] == 1
    assert trainer.last_eval_tensors["logits"].shape == (4, 2)
    assert trainer.last_eval_tensors["target"].shape == (4,)
    assert graph.get_node_by_name("target").feature_message.current_state.shape == (2,)
    assert (tmp_path / "best").is_dir()

    trainer.history["train"]["metrics"].append({
        "loss": 0.25,
        "batch_accuracy": 0.75,
    })
    trainer.save_last_checkpoint(1)
    expected = next(graph.parameters()).detach().clone()
    restored_graph = _graph()
    restored_trainer = MHD_Trainer(
        restored_graph,
        torch.optim.SGD(restored_graph.parameters(), lr=0.1),
        MHD_Monitor(["batch_accuracy"]),
        forward_levels=[0, 1],
        backward_levels=[3, 2],
        criteria=validation_macro_f1,
        criteria_mode="max",
        save_dir=str(tmp_path),
        input_nodes=["input", "target"],
        input_mapping={"input": "features", "target": "labels"},
        output_nodes=["logits", "loss", "batch_accuracy", "target"],
    )
    restored_epoch = restored_trainer.load_checkpoint(load_last=True)
    assert restored_epoch == 1
    assert torch.equal(next(restored_graph.parameters()), expected)
    assert restored_trainer.history == trainer.history


def test_criteria_can_use_complete_scalar_node_values(tmp_path):
    graph = _graph()
    trainer = MHD_Trainer(
        graph,
        torch.optim.SGD(graph.parameters(), lr=0.1),
        MHD_Monitor(["batch_accuracy"]),
        forward_levels=[0, 1],
        backward_levels=[3, 2],
        criteria=validation_loss,
        criteria_mode="min",
        save_dir=str(tmp_path),
        input_nodes=["input", "target"],
        input_mapping={"input": "features", "target": "labels"},
        output_nodes=["logits", "loss", "batch_accuracy", "target"],
    )
    metrics = trainer.eval_epoch([{
        "features": torch.tensor([[8.0, 0.0], [0.0, 8.0]]),
        "labels": torch.tensor([0, 1]),
    }], epoch=0)

    assert metrics["validation_loss"] < 0.001
    assert metrics["loss"] < 0.001
