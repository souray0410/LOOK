from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


from MHD_Project.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
from MHD_Project.MHD_Utils_V4 import prune_isolated_graph


FUSION_POSITIONS = ("input", "stem", "layer1", "layer2", "layer3", "layer4", "feature")
STAGES = ("stem", "layer1", "layer2", "layer3", "layer4", "feature")
NODE_SHAPES = {
    "input": (3, 224, 224),
    "stem": (64, 56, 56),
    "layer1": (256, 56, 56),
    "layer2": (512, 28, 28),
    "layer3": (1024, 14, 14),
    "layer4": (2048, 7, 7),
    "feature": (2048,),
    "logits": None,
}


class PoolFlatten(nn.Module):
    def __init__(self, pool: nn.Module) -> None:
        super().__init__()
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(x), 1)


class ConcatProjection(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)
        self.normalization = nn.BatchNorm2d(channels)
        self._initialize_average()

    def _initialize_average(self) -> None:
        channels = self.projection.out_channels
        with torch.no_grad():
            self.projection.weight.zero_()
            indices = torch.arange(channels)
            self.projection.weight[indices, indices, 0, 0] = 0.5
            self.projection.weight[indices, indices + channels, 0, 0] = 0.5

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        fused = self.projection(torch.cat([first, second], dim=1))
        return self.normalization(fused)


class ConcatFeatureProjection(nn.Module):
    def __init__(self, features: int = 2048) -> None:
        super().__init__()
        self.projection = nn.Linear(2 * features, features, bias=False)
        self.normalization = nn.LayerNorm(features)
        with torch.no_grad():
            self.projection.weight.zero_()
            indices = torch.arange(features)
            self.projection.weight[indices, indices] = 0.5
            self.projection.weight[indices, indices + features] = 0.5

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        fused = self.projection(torch.cat([first, second], dim=1))
        return self.normalization(fused)


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(feature))


class ClassificationLoss(nn.Module):
    """Differentiable training objective represented as an MHD edge operation."""

    def __init__(
        self,
        class_counts: Sequence[int],
        beta: float,
        label_smoothing: float,
    ) -> None:
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float64)
        if counts.ndim != 1 or torch.any(counts <= 0):
            raise ValueError("class_counts must contain one positive count per class")
        effective = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
        weights = (1.0 - beta) / effective
        weights = (weights / weights.mean()).float()
        self.register_buffer("class_counts", counts.long())
        self.register_buffer("class_weights", weights)
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits,
            labels.long(),
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )


class BatchAccuracy(nn.Module):
    """Non-differentiable online diagnostic; formal metrics use full-split predictions."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return (logits.detach().argmax(dim=1) == labels.detach().long()).float().mean()


def _resnet_modules(
    model: nn.Module,
    num_classes: int,
    classifier_dropout: float,
) -> Dict[str, nn.Module]:
    return {
        "stem": nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
        "feature": PoolFlatten(model.avgpool),
        "classifier": ClassificationHead(model.fc.in_features, num_classes, classifier_dropout),
    }


def _make_node(node_id: int, name: str, stage: str, batch_size: int, device: torch.device, num_classes: int) -> MHD_Node:
    shape = (num_classes,) if stage == "logits" else NODE_SHAPES[stage]
    state = torch.zeros((batch_size, *shape), device=device)
    return MHD_Node(
        node_id,
        name,
        MHD_Node.Message(state),
        feature_aggregation="replace",
        gradient_aggregation="sum",
    )


def _make_tensor_node(node_id: int, name: str, state: torch.Tensor) -> MHD_Node:
    return MHD_Node(
        node_id,
        name,
        MHD_Node.Message(state),
        feature_aggregation="replace",
        gradient_aggregation="sum",
    )


def _add_edge(edges: List[MHD_Edge], name: str, operation: nn.Module) -> None:
    edges.append(MHD_Edge(len(edges), name, [MHD_Edge.Operation(operation)]))


def _active_edge_groups(fusion_position: str) -> List[List[str]]:
    fusion_index = FUSION_POSITIONS.index(fusion_position)
    groups: List[List[str]] = []
    prefix = STAGES[:fusion_index]
    for stage in prefix:
        groups.append([f"oct_{stage}_edge", f"cfp_{stage}_edge"])
    groups.append([f"fuse_{fusion_position}_edge"])
    for stage in STAGES[fusion_index:]:
        groups.append([f"fusion_{stage}_edge"])
    groups.append(["fusion_classifier_edge"])
    groups.append(["classification_loss_edge", "batch_accuracy_edge"])
    return groups


def _build_topology(
    nodes: Sequence[MHD_Node],
    edges: Sequence[MHD_Edge],
    fusion_position: str,
    device: torch.device,
) -> MHD_Topo:
    node_ids = {node.name: node.id for node in nodes}
    edge_ids = {edge.name: edge.id for edge in edges}
    connections: Dict[str, Tuple[List[str], str]] = {}
    for branch in ("oct", "cfp", "fusion"):
        previous = f"{branch}_input"
        for stage in STAGES:
            connections[f"{branch}_{stage}_edge"] = ([previous], f"{branch}_{stage}")
            previous = f"{branch}_{stage}"
        connections[f"{branch}_classifier_edge"] = ([previous], f"{branch}_logits")
    for position in FUSION_POSITIONS:
        connections[f"fuse_{position}_edge"] = (
            [f"oct_{position}", f"cfp_{position}"],
            f"fusion_{position}",
        )
    connections["classification_loss_edge"] = (
        ["fusion_logits", "label_gt"],
        "loss",
    )
    connections["batch_accuracy_edge"] = (
        ["fusion_logits", "label_gt"],
        "batch_accuracy",
    )

    role_matrices, sort_matrices = [], []
    for group in _active_edge_groups(fusion_position):
        role = torch.zeros((len(edges), len(nodes)), dtype=torch.int8, device=device)
        sort = torch.zeros_like(role)
        for edge_name in group:
            edge_id = edge_ids[edge_name]
            heads, tail = connections[edge_name]
            for order, head in enumerate(heads):
                role[edge_id, node_ids[head]] = -1
                sort[edge_id, node_ids[head]] = order
            role[edge_id, node_ids[tail]] = 1
            sort[edge_id, node_ids[tail]] = len(heads)
        role_matrices.append(role)
        sort_matrices.append(sort)
    return MHD_Topo(role_matrices, sort_matrices)


def build_resnet50_mhd_graph(
    fusion_position: str,
    num_classes: int = 5,
    batch_size: int = 1,
    image_size: int = 224,
    device: torch.device | str = "cpu",
    pretrained: bool = True,
    class_counts: Sequence[int] | None = None,
    class_balance_beta: float = 0.999,
    label_smoothing: float = 0.0,
    classifier_dropout: float = 0.1,
) -> MHD_Graph:
    if fusion_position not in FUSION_POSITIONS:
        raise ValueError(f"fusion_position must be one of {FUSION_POSITIONS}")
    if image_size != 224:
        raise ValueError("The registered node shapes currently require image_size=224")
    device = torch.device(device)
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    template = resnet50(weights=weights)
    models = {name: copy.deepcopy(template) for name in ("oct", "cfp", "fusion")}
    modules = {
        name: _resnet_modules(model, num_classes, classifier_dropout)
        for name, model in models.items()
    }

    nodes: List[MHD_Node] = []
    for branch in ("oct", "cfp", "fusion"):
        for stage in ("input", *STAGES, "logits"):
            nodes.append(
                _make_node(len(nodes), f"{branch}_{stage}", stage, batch_size, device, num_classes)
            )
    nodes.append(_make_tensor_node(
        len(nodes), "label_gt", torch.zeros(batch_size, dtype=torch.long, device=device)
    ))
    nodes.append(_make_tensor_node(len(nodes), "loss", torch.zeros((), device=device)))
    nodes.append(_make_tensor_node(len(nodes), "batch_accuracy", torch.zeros((), device=device)))

    edges: List[MHD_Edge] = []
    for branch in ("oct", "cfp", "fusion"):
        for stage in STAGES:
            _add_edge(edges, f"{branch}_{stage}_edge", modules[branch][stage])
        _add_edge(edges, f"{branch}_classifier_edge", modules[branch]["classifier"])

    channels = {"input": 3, "stem": 64, "layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
    for position in FUSION_POSITIONS:
        operation = (
            ConcatFeatureProjection(2048)
            if position == "feature"
            else ConcatProjection(channels[position])
        )
        _add_edge(edges, f"fuse_{position}_edge", operation)
    resolved_counts = list(class_counts) if class_counts is not None else [1] * num_classes
    if len(resolved_counts) != num_classes:
        raise ValueError("class_counts length must equal num_classes")
    _add_edge(
        edges,
        "classification_loss_edge",
        ClassificationLoss(resolved_counts, class_balance_beta, label_smoothing),
    )
    _add_edge(edges, "batch_accuracy_edge", BatchAccuracy())

    topology = _build_topology(nodes, edges, fusion_position, device)
    graph = MHD_Graph(set(nodes), set(edges), {topology}, device=device)
    prune_isolated_graph(graph, verbose=False)
    graph.architecture_id = f"resnet50_oct_cfp_fusion_{fusion_position}"
    graph.fusion_position = fusion_position
    start = FUSION_POSITIONS.index(fusion_position)
    graph.correction_nodes = [f"fusion_{name}" for name in FUSION_POSITIONS[start:]]
    graph.node_level_map = active_node_levels(graph)
    graph.monitor_nodes = ["loss", "batch_accuracy", "fusion_logits"]
    graph.monitor_edges = ["fusion_classifier_edge"]
    graph.monitor_levels = [graph.num_levels - 1]
    graph.model_levels = list(range(graph.num_levels - 1))
    return graph


def active_node_levels(graph: MHD_Graph) -> Dict[str, int]:
    levels: Dict[str, int] = {}
    for level, role in enumerate(graph.topo.role_matrices):
        for edge_id in range(role.shape[0]):
            tail_ids = torch.where(role[edge_id] > 0)[0].tolist()
            for node_id in tail_ids:
                node = graph.get_node_by_id(node_id)
                levels[node.name] = level
    return levels


def reset_and_forward(
    graph: MHD_Graph,
    oct_tensor: torch.Tensor,
    cfp_tensor: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> torch.Tensor | Dict[str, torch.Tensor]:
    for node in graph.nodes:
        node.reset()
    graph.get_node_by_name("oct_input").feature_message.current_state = oct_tensor
    graph.get_node_by_name("cfp_input").feature_message.current_state = cfp_tensor
    if labels is None:
        graph.forward(levels=graph.model_levels)
        return graph.get_node_by_name("fusion_logits").feature_message.current_state
    graph.get_node_by_name("label_gt").feature_message.current_state = labels.long()
    graph.forward()
    return {
        "fusion_logits": graph.get_node_by_name("fusion_logits").feature_message.current_state,
        "loss": graph.get_node_by_name("loss").feature_message.current_state,
        "batch_accuracy": graph.get_node_by_name("batch_accuracy").feature_message.current_state,
    }


def optimizer_parameter_groups(graph: MHD_Graph, pretrained_lr: float, new_layer_lr: float):
    pretrained, new = [], []
    for edge in graph.edges:
        target = new if edge.name.startswith("fuse_") or edge.name == "fusion_classifier_edge" else pretrained
        for operation in edge.edge_operations:
            if isinstance(operation.function, nn.Module):
                target.extend(operation.function.parameters())
    return [
        {"params": pretrained, "lr": pretrained_lr, "group_name": "pretrained"},
        {"params": new, "lr": new_layer_lr, "group_name": "fusion_and_head"},
    ]


def classification_loss_metadata(graph: MHD_Graph) -> Dict[str, object]:
    operation = graph.get_edge_by_name("classification_loss_edge").edge_operations[0].function
    if not isinstance(operation, ClassificationLoss):
        raise TypeError("classification_loss_edge does not contain ClassificationLoss")
    return {
        "name": "class_balanced_ce",
        "beta": operation.beta,
        "label_smoothing": operation.label_smoothing,
        "class_counts": operation.class_counts.detach().cpu().tolist(),
        "class_weights": operation.class_weights.detach().cpu().tolist(),
    }


def graph_summary(graph: MHD_Graph) -> Dict[str, object]:
    return {
        "framework_version": "MHD V4",
        "architecture_id": graph.architecture_id,
        "nodes": [node.name for node in sorted(graph.nodes, key=lambda item: item.id)],
        "edges": [edge.name for edge in sorted(graph.edges, key=lambda item: item.id)],
        "levels": graph.num_levels,
        "backward_levels": len(graph.topo.backward_role_matrices),
        "backward_topology": "auto_reverse" if graph.topo.backward_is_auto else "explicit",
        "backward_api": "MHD_Graph.backward",
        "correction_nodes": graph.correction_nodes,
        "monitor_nodes": graph.monitor_nodes,
        "monitor_edges": graph.monitor_edges,
        "fusion_operation": "concatenate_linear_projection_normalization",
        "fusion_nonlinearity": False,
        "classification_loss": classification_loss_metadata(graph),
        "parameters": sum(parameter.numel() for parameter in graph.parameters()),
    }
