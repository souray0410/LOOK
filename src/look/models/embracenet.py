"""Author-faithful EmbraceNet fusion and a finite MHD prototype for observed CFP/OCT."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Mapping, Sequence

import torch
from torch import nn

from mhd_framework.core import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
from look.models.graph import ClassificationHead
from look.models.native_host import HostLoss, NativeCut, ObservedMean, STAGE_MAP, set_observed_counts

AUTHOR_REPOSITORY = "https://github.com/idearibosome/embracenet"
AUTHOR_COMMIT = "c61b63dadc6fc8d719bb77475f4734f62697144e"
AUTHOR_LICENSE = "MIT"
AUTHOR_FILE_SHA256 = {
    "LICENSE": "63def471f62dc22e08dddc85b61c2c1e46a853542e81081951694d8feabe6ccc",
    "README.md": "28949a2a0d0ab8211e784eb8b2847b3f18557c219bdf7c79152ab0f90ab3333f",
    "embracenet_pytorch/embracenet.py": "8b3414181bd77cf37bb2f5b39d43bb2ce6cbd73bcf5afa8dbccb6b4e1d11df05",
}
DEFAULT_EMBRACEMENT_SIZE = 256


class EmbraceNetFusion(nn.Module):
    """PyTorch EmbraceNet with explicit validation, replay, and private sampling state.

    For finite, legal author inputs this preserves Linear+ReLU docking, availability
    weighting/renormalization, and per-coordinate torch.multinomial selection.
    Project-only extensions are fail-closed validation, NaN-safe unavailable rows,
    an independent Generator, and exact trace replay.
    """

    def __init__(
        self,
        input_size_list: Sequence[int],
        embracement_size: int = DEFAULT_EMBRACEMENT_SIZE,
        *,
        sampling_seed: int = 0,
        bypass_docking: bool = False,
    ) -> None:
        super().__init__()
        if not input_size_list or any(type(size) is not int or size <= 0 for size in input_size_list):
            raise ValueError("input_size_list must contain positive integers")
        if type(embracement_size) is not int or embracement_size <= 0:
            raise ValueError("embracement_size must be a positive integer")
        self.input_size_list = tuple(input_size_list)
        self.embracement_size = int(embracement_size)
        self.bypass_docking = bool(bypass_docking)
        if self.bypass_docking and any(size != self.embracement_size for size in self.input_size_list):
            raise ValueError("bypass_docking requires every input width to equal embracement_size")
        if not self.bypass_docking:
            for index, input_size in enumerate(self.input_size_list):
                setattr(self, f"docking_{index}", nn.Linear(input_size, self.embracement_size))
        self.sampling_seed = int(sampling_seed)
        self._sampling_generator: torch.Generator | None = None
        self._sampling_device: torch.device | None = None
        self._draw_count = 0
        self._next_replay_indices: torch.Tensor | None = None
        self._last_trace: dict[str, object] | None = None

    @property
    def num_modalities(self) -> int:
        return len(self.input_size_list)

    def _generator_for(self, device: torch.device) -> torch.Generator:
        device = torch.device(device)
        if self._sampling_generator is None:
            self._sampling_generator = torch.Generator(device=device)
            self._sampling_generator.manual_seed(self.sampling_seed)
            self._sampling_device = device
        elif self._sampling_device != device:
            raise ValueError(
                f"EmbraceNet sampling generator is bound to {self._sampling_device}, not {device}; "
                "restore an explicit device-matched sampling state"
            )
        return self._sampling_generator

    def get_sampling_state(self) -> dict[str, object]:
        if self._sampling_generator is None:
            return {
                "schema": "look_embracenet_sampling_v1",
                "seed": self.sampling_seed,
                "device": None,
                "state": None,
                "draw_count": self._draw_count,
            }
        return {
            "schema": "look_embracenet_sampling_v1",
            "seed": self.sampling_seed,
            "device": str(self._sampling_device),
            "state": self._sampling_generator.get_state().cpu().clone(),
            "draw_count": self._draw_count,
        }

    def set_sampling_state(self, state: Mapping[str, object]) -> None:
        # Restoring a checkpoint is also a control-flow boundary: stale exact
        # replay instructions and stale traces must never survive it, even when
        # the supplied state is invalid and validation below raises.
        self._next_replay_indices = None
        self._last_trace = None
        if state.get("schema") != "look_embracenet_sampling_v1":
            raise ValueError("Unknown EmbraceNet sampling-state schema")
        if int(state.get("seed", self.sampling_seed)) != self.sampling_seed:
            raise ValueError("EmbraceNet sampling seed identity changed")
        device_name = state.get("device")
        raw_state = state.get("state")
        draw_count = int(state.get("draw_count", 0))
        if draw_count < 0:
            raise ValueError("Invalid EmbraceNet draw count")
        if device_name is None:
            if raw_state is not None or draw_count != 0:
                raise ValueError("Uninitialized sampling state is inconsistent")
            self._sampling_generator = None
            self._sampling_device = None
            self._draw_count = 0
            return
        if not isinstance(raw_state, torch.Tensor):
            raise TypeError("EmbraceNet sampling state must contain a tensor")
        device = torch.device(str(device_name))
        generator = torch.Generator(device=device)
        generator.set_state(raw_state.cpu())
        self._sampling_generator = generator
        self._sampling_device = device
        self._draw_count = draw_count

    def set_replay_indices(self, indices: torch.Tensor) -> None:
        if self._next_replay_indices is not None:
            raise ValueError("A replay trace is already pending")
        if not isinstance(indices, torch.Tensor):
            raise TypeError("Replay modality_indices must be a tensor")
        if indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("Replay modality_indices must use an integer dtype")
        if indices.ndim != 2 or indices.shape[1] != self.embracement_size:
            raise ValueError(
                f"Replay modality_indices must have shape [batch,{self.embracement_size}]"
            )
        if torch.any(indices < 0) or torch.any(indices >= self.num_modalities):
            raise ValueError("Replay modality_indices contain an invalid modality")
        self._next_replay_indices = indices.detach().clone()

    def clear_pending_replay(self) -> None:
        self._next_replay_indices = None

    def pop_last_trace(self) -> dict[str, object]:
        if self._last_trace is None:
            raise RuntimeError("No EmbraceNet sampling trace is available")
        trace = self._last_trace
        self._last_trace = None
        return trace

    def peek_last_trace(self) -> dict[str, object] | None:
        if self._last_trace is None:
            return None
        trace = dict(self._last_trace)
        for key in ("modality_indices", "probabilities"):
            value = trace.get(key)
            if isinstance(value, torch.Tensor):
                trace[key] = value.clone()
        for key in ("sampling_state_before", "sampling_state_after"):
            state = trace.get(key)
            if isinstance(state, dict):
                state = dict(state)
                if isinstance(state.get("state"), torch.Tensor):
                    state["state"] = state["state"].clone()
                trace[key] = state
        return trace

    @staticmethod
    def _matrix(
        value: torch.Tensor | None,
        *,
        name: str,
        batch_size: int,
        num_modalities: int,
        device: torch.device,
        dtype: torch.dtype,
        default: float,
    ) -> torch.Tensor:
        if value is None:
            return torch.full((batch_size, num_modalities), default, dtype=dtype, device=device)
        if value.ndim != 2 or tuple(value.shape) != (batch_size, num_modalities):
            raise ValueError(f"{name} must have shape {(batch_size, num_modalities)}")
        return value.to(device=device, dtype=dtype)

    def _validate_inputs(self, input_list: Sequence[torch.Tensor]) -> tuple[int, torch.device, torch.dtype]:
        if len(input_list) != self.num_modalities:
            raise ValueError("input_list modality count does not match input_size_list")
        first = input_list[0]
        if first.ndim != 2 or not first.is_floating_point():
            raise ValueError("EmbraceNet inputs must be floating [batch, feature] tensors")
        batch_size, device, dtype = first.shape[0], first.device, first.dtype
        for index, (value, width) in enumerate(zip(input_list, self.input_size_list)):
            if value.ndim != 2 or tuple(value.shape) != (batch_size, width):
                raise ValueError(f"modality {index} must have shape {(batch_size, width)}")
            if value.device != device or value.dtype != dtype:
                raise ValueError("All EmbraceNet inputs must share device and dtype")
        return batch_size, device, dtype

    def analytical_moments(
        self,
        input_list: Sequence[torch.Tensor],
        availabilities: torch.Tensor | None = None,
        selection_probabilities: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Exact first/coordinatewise-second moments of the author embracement.

        This is an evaluation/statistics helper only. It does not replace the
        stochastic forward used for training or ordinary author-domain replay.
        Coordinate selections are conditionally independent in the author
        implementation, so E[zz^T] = mu mu^T + diag(var).
        """
        batch_size, device, dtype = self._validate_inputs(input_list)
        availability = self._matrix(
            availabilities,
            name="availabilities",
            batch_size=batch_size,
            num_modalities=self.num_modalities,
            device=device,
            dtype=dtype,
            default=1.0,
        )
        if not torch.isfinite(availability).all() or not torch.all((availability == 0) | (availability == 1)):
            raise ValueError("availabilities must contain only finite 0/1 values")
        if torch.any(availability.sum(dim=1) == 0):
            raise ValueError("Every sample must have at least one available modality")
        probabilities = self._matrix(
            selection_probabilities,
            name="selection_probabilities",
            batch_size=batch_size,
            num_modalities=self.num_modalities,
            device=device,
            dtype=dtype,
            default=1.0,
        )
        if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
            raise ValueError("selection_probabilities must be finite and non-negative")
        probabilities = probabilities * availability
        total = probabilities.sum(dim=-1, keepdim=True)
        if torch.any(total <= 0):
            raise ValueError("Available modalities must have positive total selection probability")
        probabilities = probabilities / total
        docking = []
        for index, input_data in enumerate(input_list):
            row_available = availability[:, index].bool().unsqueeze(1)
            available_values = input_data[row_available.squeeze(1)]
            if available_values.numel() and not torch.isfinite(available_values).all():
                raise ValueError(f"Available modality {index} contains non-finite values")
            safe_input = torch.where(row_available, input_data, torch.zeros_like(input_data))
            value = safe_input if self.bypass_docking else torch.relu(getattr(self, f"docking_{index}")(safe_input))
            docking.append(value)
        stack = torch.stack(docking, dim=-1)
        weights = probabilities.unsqueeze(1)
        mean = (stack * weights).sum(dim=-1)
        second = (stack.square() * weights).sum(dim=-1)
        variance = (second - mean.square()).clamp_min(0)
        return {
            "mean": mean,
            "second_moment_diagonal": second,
            "variance_diagonal": variance,
            "probabilities": probabilities,
        }

    def forward(
        self,
        input_list: Sequence[torch.Tensor],
        availabilities: torch.Tensor | None = None,
        selection_probabilities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Consume/clear any pending exact replay before validations. A failed
        # call must never leave replay armed for a later unrelated forward.
        replay = self._next_replay_indices
        self._next_replay_indices = None
        self._last_trace = None
        batch_size, device, dtype = self._validate_inputs(input_list)
        availability = self._matrix(
            availabilities,
            name="availabilities",
            batch_size=batch_size,
            num_modalities=self.num_modalities,
            device=device,
            dtype=dtype,
            default=1.0,
        )
        if not torch.isfinite(availability).all() or not torch.all((availability == 0) | (availability == 1)):
            raise ValueError("availabilities must contain only finite 0/1 values")
        if torch.any(availability.sum(dim=1) == 0):
            raise ValueError("Every sample must have at least one available modality")
        probabilities = self._matrix(
            selection_probabilities,
            name="selection_probabilities",
            batch_size=batch_size,
            num_modalities=self.num_modalities,
            device=device,
            dtype=dtype,
            default=1.0,
        )
        if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
            raise ValueError("selection_probabilities must be finite and non-negative")
        probabilities = probabilities * availability
        probability_sum = probabilities.sum(dim=-1, keepdim=True)
        if torch.any(probability_sum <= 0):
            raise ValueError("Available modalities must have positive total selection probability")
        probabilities = probabilities / probability_sum

        docking_output_list: list[torch.Tensor] = []
        for index, input_data in enumerate(input_list):
            row_available = availability[:, index].bool().unsqueeze(1)
            available_values = input_data[row_available.squeeze(1)]
            if available_values.numel() and not torch.isfinite(available_values).all():
                raise ValueError(f"Available modality {index} contains non-finite values")
            safe_input = torch.where(row_available, input_data, torch.zeros_like(input_data))
            if self.bypass_docking:
                docking_output = safe_input
            else:
                docking_output = torch.relu(getattr(self, f"docking_{index}")(safe_input))
            docking_output_list.append(docking_output)

        docking_output_stack = torch.stack(docking_output_list, dim=-1)
        state_before = self.get_sampling_state()
        if replay is None:
            generator = self._generator_for(device)
            modality_indices = torch.multinomial(
                probabilities,
                num_samples=self.embracement_size,
                replacement=True,
                generator=generator,
            )
            self._draw_count += batch_size * self.embracement_size
            replayed = False
        else:
            modality_indices = replay.to(device=device, dtype=torch.long)
            if tuple(modality_indices.shape) != (batch_size, self.embracement_size):
                raise ValueError("Replay modality_indices shape does not match the current batch")
            if torch.any(modality_indices < 0) or torch.any(modality_indices >= self.num_modalities):
                raise ValueError("Replay modality_indices contain an invalid modality")
            selected_available = availability.gather(1, modality_indices)
            if not torch.all(selected_available == 1):
                raise ValueError("Replay trace selects a modality unavailable in the current sample")
            selected_probability = probabilities.gather(1, modality_indices)
            if torch.any(selected_probability <= 0):
                raise ValueError("Replay trace selects a zero-probability modality")
            replayed = True

        embraced = docking_output_stack.gather(2, modality_indices.unsqueeze(-1)).squeeze(-1)
        state_after = self.get_sampling_state()
        self._last_trace = {
            "schema": "look_embracenet_trace_v1",
            "modality_indices": modality_indices.detach().cpu().clone(),
            "probabilities": probabilities.detach().cpu().clone(),
            "sampling_state_before": state_before,
            "sampling_state_after": state_after,
            "replayed": replayed,
        }
        return embraced


class EmbraceNetMHDOperation(nn.Module):
    """MHD edge adapter; keeps the author-style EmbraceNetFusion call contract intact."""

    def __init__(self, fusion: EmbraceNetFusion) -> None:
        super().__init__()
        self.fusion = fusion

    def forward(
        self,
        oct_feature: torch.Tensor,
        cfp_feature: torch.Tensor,
        availabilities: torch.Tensor,
        selection_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        return self.fusion([oct_feature, cfp_feature], availabilities, selection_probabilities)


def _get_embracenet_operation(graph: MHD_Graph) -> EmbraceNetFusion:
    edge = graph.get_edge_by_name("embracenet_edge")
    if edge is None or len(edge.edge_operations) != 1:
        raise ValueError("Graph does not expose one EmbraceNet operation")
    operation = edge.edge_operations[0].function
    if not isinstance(operation, EmbraceNetMHDOperation):
        raise TypeError("embracenet_edge does not contain EmbraceNetMHDOperation")
    return operation.fusion


def capture_embracenet_sampling(graph: MHD_Graph) -> dict[str, object]:
    return _get_embracenet_operation(graph).get_sampling_state()


def restore_embracenet_sampling(graph: MHD_Graph, state: Mapping[str, object]) -> None:
    _get_embracenet_operation(graph).set_sampling_state(state)


def get_embracenet_trace(graph: MHD_Graph) -> dict[str, object]:
    trace = _get_embracenet_operation(graph).peek_last_trace()
    if trace is None:
        raise RuntimeError("No EmbraceNet trace is available")
    return trace


def set_embracenet_replay(graph: MHD_Graph, modality_indices: torch.Tensor) -> None:
    _get_embracenet_operation(graph).set_replay_indices(modality_indices)


def clear_embracenet_replay(graph: MHD_Graph) -> None:
    _get_embracenet_operation(graph).clear_pending_replay()


@contextmanager
def embracenet_graph_replay_scope(graph: MHD_Graph):
    """Own the public graph-entry one-shot replay lifecycle.

    External callers may prearm exact replay before entering either public graph
    forward. Success consumes it inside EmbraceNet. Any exception or a normal
    stop before embracement must clear it here so the next unrelated call is
    stochastic. Clearing does not advance or roll back the sampling generator.
    """
    try:
        yield
    finally:
        clear_embracenet_replay(graph)


def build_embracenet_host(
    cfp_parent,
    oct_parent,
    device: torch.device | str = "cpu",
    *,
    embracement_size: int = DEFAULT_EMBRACEMENT_SIZE,
    sampling_seed: int = 3416,
    classifier_dropout: float = 0.0,
) -> MHD_Graph:
    """Build a 2-D observed-eye R18 EmbraceNet MHD prototype.

    The two encoders remain complete modality-specific native parents. Each is
    pooled across the actually observed eyes, then EmbraceNet docks/embraces the
    two participant features and a new binary head consumes the embraced feature.
    """
    a, b = cfp_parent.graph, oct_parent.graph
    if a.configuration() != b.configuration():
        raise ValueError("EmbraceNet prototype requires same-architecture complete parents")
    config = a.configuration()
    if config.get("name") != "resnet18" or config.get("views", 1) != 1 or config.get("spatial_dims", 2) != 2:
        raise ValueError("This finite EmbraceNet prototype is locked to observed-eye 2-D ResNet18 parents")
    stages = STAGE_MAP["resnet18"]
    device = torch.device(device)
    nodes: list[MHD_Node] = []
    edges: list[MHD_Edge] = []
    definitions: list[tuple[int, list[int], int]] = []
    groups: list[list[int]] = []
    levels: dict[str, int] = {}

    def node(name: str) -> int:
        item = MHD_Node(len(nodes), name, MHD_Node.Message(torch.zeros(1, device=device)), aggregation="sum", memory=False)
        nodes.append(item)
        return item.id

    roots = {name: node(name) for name in (
        "oct_input", "cfp_input", "eye_counts", "embrace_availability", "embrace_selection_probabilities", "label_gt"
    )}

    def edge(name: str, module: nn.Module, inputs: list[int], output_name: str) -> tuple[int, int]:
        output = node(output_name)
        item = MHD_Edge(len(edges), name, [MHD_Edge.Operation(module)])
        edges.append(item)
        definitions.append((item.id, inputs, output))
        return item.id, output

    previous = {"oct": roots["oct_input"], "cfp": roots["cfp_input"]}
    source = {"oct": b, "cfp": a}
    for stage_index, stage in enumerate(stages):
        group: list[int] = []
        for modality in ("oct", "cfp"):
            start = "input" if stage_index == 0 else stages[stage_index - 1]
            edge_id, output = edge(
                f"{modality}_{stage}_edge",
                NativeCut(source[modality], start, stage),
                [previous[modality]],
                f"{modality}_{stage}",
            )
            previous[modality] = output
            group.append(edge_id)
        groups.append(group)

    pool_group: list[int] = []
    participant: dict[str, int] = {}
    for modality in ("oct", "cfp"):
        edge_id, output = edge(
            f"{modality}_observed_pool_edge",
            ObservedMean(),
            [previous[modality], roots["eye_counts"]],
            f"{modality}_participant_feature",
        )
        participant[modality] = output
        pool_group.append(edge_id)
    groups.append(pool_group)

    head = NativeCut(a, "features", "logits")
    feature_width = next(module.in_features for module in head.modules() if isinstance(module, nn.Linear))
    embrace = EmbraceNetFusion(
        [feature_width, feature_width],
        embracement_size,
        sampling_seed=sampling_seed,
    )
    embrace_id, embraced = edge(
        "embracenet_edge",
        EmbraceNetMHDOperation(embrace),
        [
            participant["oct"],
            participant["cfp"],
            roots["embrace_availability"],
            roots["embrace_selection_probabilities"],
        ],
        "embraced_feature",
    )
    groups.append([embrace_id])

    classifier_id, logits = edge(
        "fusion_classifier_edge",
        ClassificationHead(embracement_size, int(config["num_classes"]), classifier_dropout),
        [embraced],
        "fusion_logits",
    )
    groups.append([classifier_id])
    loss_id, _ = edge("classification_loss_edge", HostLoss(), [logits, roots["label_gt"]], "loss")
    groups.append([loss_id])

    roles: list[torch.Tensor] = []
    sorts: list[torch.Tensor] = []
    for level, group in enumerate(groups):
        role = torch.zeros(len(edges), len(nodes), dtype=torch.long, device=device)
        order = torch.zeros_like(role)
        for edge_id in group:
            _, inputs, output = definitions[edge_id]
            for index, input_id in enumerate(inputs):
                role[edge_id, input_id] = -1
                order[edge_id, input_id] = index
            role[edge_id, output] = 1
            order[edge_id, output] = len(inputs)
            levels[nodes[output].name] = level
        roles.append(role)
        sorts.append(order)

    topology = MHD_Topo(roles + [-role for role in roles], sorts + [order.clone() for order in sorts])
    graph = MHD_Graph(set(nodes), set(edges), {topology}, device=device)
    graph.forward_levels = list(range(len(groups)))
    graph.backward_levels = list(range(2 * len(groups) - 1, len(groups) - 1, -1))
    graph.model_levels = graph.forward_levels[:-1]
    graph.node_level_map = levels
    graph.correction_nodes = [
        "joint_input",
        "joint_stem",
        "joint_stage1",
        "joint_stage2",
        "joint_stage3",
        "joint_stage4",
        "joint_features",
        "joint_participant_feature",
        "embraced_feature",
    ]
    graph.fusion_position = "embracenet"
    graph.num_classes = int(config["num_classes"])
    graph.observed_eye_input = True
    graph.architecture_id = "resnet18_observed_embracenet"
    graph.embracenet_missing_method = True
    graph.embracenet_provenance = {
        "schema": "look_embracenet_mhd_prototype_v1",
        "author_repository": AUTHOR_REPOSITORY,
        "author_commit": AUTHOR_COMMIT,
        "author_license": AUTHOR_LICENSE,
        "author_file_sha256": dict(AUTHOR_FILE_SHA256),
        "modality_order": ["oct", "cfp"],
        "participant_pooling": "valid_eye_feature_mean_v1_before_embracement",
        "embracement_size": embracement_size,
        "sampling_seed": sampling_seed,
        "sampling": "torch.multinomial_per_coordinate_with_replacement",
        "docking": "Linear+ReLU",
        "classifier": f"new_linear_head_{embracement_size}_to_{config['num_classes']}",
        "source_config": config,
        "full_positive_tree_sites": list(graph.correction_nodes),
    }
    return graph


def _packed_counts(counts, eye_count: int) -> list[int]:
    values = counts.detach().cpu().tolist() if isinstance(counts, torch.Tensor) else list(counts)
    if any(type(value) is not int or value not in (1, 2) for value in values) or sum(values) != eye_count:
        raise ValueError("Invalid observed-eye counts")
    return values


def _validated_availability(counts, eye_count: int, availabilities: torch.Tensor, device: torch.device) -> tuple[list[int], torch.Tensor]:
    values = _packed_counts(counts, eye_count)
    if availabilities.ndim != 2 or tuple(availabilities.shape) != (len(values), 2):
        raise ValueError("EmbraceNet participant availabilities must have shape [participants,2]")
    availability = availabilities.to(device=device, dtype=torch.float32)
    if not torch.isfinite(availability).all() or not torch.all((availability == 0) | (availability == 1)):
        raise ValueError("availabilities must contain only finite 0/1 values")
    if torch.any(availability.sum(dim=1) == 0):
        raise ValueError("Every participant must have at least one available modality")
    return values, availability


def prepare_embracenet_inputs(
    oct_tensor: torch.Tensor,
    cfp_tensor: torch.Tensor,
    counts,
    availabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared train/A/A+LOOK missing-input contract before either R18 encoder."""
    if oct_tensor.ndim != cfp_tensor.ndim or len(oct_tensor) != len(cfp_tensor):
        raise ValueError("OCT and CFP packed-eye tensors must have matching batch structure")
    if oct_tensor.device != cfp_tensor.device or oct_tensor.dtype != cfp_tensor.dtype:
        raise ValueError("OCT and CFP packed-eye tensors must share device and dtype")
    values, availability = _validated_availability(counts, len(oct_tensor), availabilities, oct_tensor.device)
    repeats = torch.tensor(values, dtype=torch.long, device=availability.device)
    eye_availability = availability.repeat_interleave(repeats, dim=0).bool()

    def safe(value: torch.Tensor, column: int) -> torch.Tensor:
        shape = (len(value),) + (1,) * (value.ndim - 1)
        mask = eye_availability[:, column].reshape(shape)
        observed = value[eye_availability[:, column]]
        if observed.numel() and not torch.isfinite(observed).all():
            raise ValueError("Available modality input contains non-finite values")
        return torch.where(mask, value, torch.zeros_like(value))

    return safe(oct_tensor, 0), safe(cfp_tensor, 1)


def set_embracenet_context(
    graph: MHD_Graph,
    counts,
    eye_count: int,
    availabilities: torch.Tensor,
    selection_probabilities: torch.Tensor | None = None,
) -> None:
    if not getattr(graph, "embracenet_missing_method", False):
        raise ValueError("Graph is not the EmbraceNet prototype")
    set_observed_counts(graph, counts, eye_count)
    _, availability = _validated_availability(counts, eye_count, availabilities, graph.device)
    probabilities = (
        torch.ones_like(availability, dtype=torch.float32, device=graph.device)
        if selection_probabilities is None
        else selection_probabilities.to(device=graph.device, dtype=torch.float32)
    )
    if tuple(probabilities.shape) != tuple(availability.shape):
        raise ValueError("EmbraceNet selection probabilities must have shape [participants,2]")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError("selection_probabilities must be finite and non-negative")
    if torch.any((probabilities * availability).sum(dim=1) <= 0):
        raise ValueError("Available modalities must have positive total selection probability")
    graph.get_node_by_name("embrace_availability").feature_message.current_state = availability
    graph.get_node_by_name("embrace_selection_probabilities").feature_message.current_state = probabilities


def reset_embracenet_inputs(
    graph: MHD_Graph,
    oct_tensor: torch.Tensor,
    cfp_tensor: torch.Tensor,
    counts,
    availabilities: torch.Tensor,
    selection_probabilities: torch.Tensor | None = None,
) -> None:
    for item in graph.nodes:
        item.reset()
    safe_oct, safe_cfp = prepare_embracenet_inputs(oct_tensor, cfp_tensor, counts, availabilities)
    graph.get_node_by_name("oct_input").feature_message.current_state = safe_oct
    graph.get_node_by_name("cfp_input").feature_message.current_state = safe_cfp
    set_embracenet_context(graph, counts, len(oct_tensor), availabilities, selection_probabilities)


def forward_embracenet_host(
    graph: MHD_Graph,
    oct_tensor: torch.Tensor,
    cfp_tensor: torch.Tensor,
    counts,
    availabilities: torch.Tensor,
    *,
    selection_probabilities: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    loss_scale: float = 1.0,
) -> torch.Tensor:
    # The public entry owns any replay prearmed through set_embracenet_replay.
    # Failures during raw-input validation, labels/loss validation, or the graph
    # itself therefore cannot leak a one-shot replay into the next call.
    with embracenet_graph_replay_scope(graph):
        reset_embracenet_inputs(
            graph,
            oct_tensor,
            cfp_tensor,
            counts,
            availabilities,
            selection_probabilities,
        )
        if labels is not None:
            if not 0 < loss_scale <= 1:
                raise ValueError("Invalid accumulation loss scale")
            graph.get_edge_by_name("classification_loss_edge").edge_operations[0].function.scale = loss_scale
            graph.get_node_by_name("label_gt").feature_message.current_state = labels.long()
        graph.forward(levels=graph.model_levels if labels is None else graph.forward_levels)
        return graph.get_node_by_name("fusion_logits").feature_message.current_state


def embracenet_parameter_groups(graph: MHD_Graph, pretrained_lr: float, new_layer_lr: float):
    """Keep two ImageNet encoders at the inherited LR; docking/head are new layers."""
    pretrained: list[nn.Parameter] = []
    new: list[nn.Parameter] = []
    for edge in graph.edges:
        target = new if edge.name in {"embracenet_edge", "fusion_classifier_edge"} else pretrained
        for operation in edge.edge_operations:
            function = operation.function
            if isinstance(function, nn.Module):
                target.extend(parameter for parameter in function.parameters() if parameter.requires_grad)
    all_parameters = [parameter for parameter in graph.parameters() if parameter.requires_grad]
    owned = pretrained + new
    if len({id(parameter) for parameter in owned}) != len(owned) or {id(parameter) for parameter in owned} != {id(parameter) for parameter in all_parameters}:
        raise ValueError("EmbraceNet optimizer groups omit or duplicate trainable parameters")
    return [
        {"params": pretrained, "lr": pretrained_lr, "group_name": "pretrained_encoders"},
        {"params": new, "lr": new_layer_lr, "group_name": "embracenet_and_head"},
    ]
