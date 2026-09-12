"""Stage-defined LOOK fusion hosts assembled from two complete MHD parents.

Only observed eyes enter either encoder. Two prefixes join once; the suffix and
head are copied from the first (CFP) parent, an explicit fixed initialization.
Internal source-node mappings are retained in each stage cut's provenance.
"""
import copy
import torch
from torch import nn
from mhd_framework.core import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph
from look.models.graph import ConcatProjection, ConcatFeatureProjection, ClassificationLoss

STAGE_MAP = {
    "resnet50": ("stem", "stage1", "stage2", "stage3", "stage4", "features"),
    "densenet121": ("stem", "denseblock1", "denseblock2", "denseblock3", "denseblock4", "features"),
    "swin_b": ("patch_embed", "stage1", "stage2", "stage3", "stage4", "features"),
}
STARTS = {"middle": 2, "deep": 4, "features": 5}


class NativeCut(nn.Module):
    """Copy a closed DAG cut, with reversible layout conversion at its boundaries."""
    def __init__(self, graph, start, end):
        super().__init__()
        self.start = graph.endpoint_nodes[start]
        end_id = graph.endpoint_nodes[end]
        first = graph._producer_levels.get(self.start, -1) + 1
        last = graph._producer_levels[end_id] + 1
        self.definitions = graph._definitions[first:last]
        available = {self.start}
        ops = []
        for edge_id, inputs, output in self.definitions:
            if not set(inputs).issubset(available):
                raise ValueError("Native endpoint is not a closed graph cut")
            operations = graph.get_edge_by_id(edge_id).edge_operations
            if len(operations) != 1 or not isinstance(operations[0].function, nn.Module):
                raise ValueError("Unsupported native operation contract")
            ops.append(copy.deepcopy(operations[0].function))
            available.add(output)
        self.operations = nn.ModuleList(ops)
        self.end = end_id
        swin = graph.config.name.startswith("swin_")
        self.input_nhwc = swin and start not in ("input", "features")
        self.output_nhwc = swin and end not in ("features", "logits")

    def forward(self, x):
        if self.input_nhwc:
            x = x.permute(0, 2, 3, 1).contiguous()
        values = {self.start: x}
        for (_, inputs, output), operation in zip(self.definitions, self.operations):
            values[output] = operation(*(values[i] for i in inputs))
        x = values[self.end]
        return x.permute(0, 3, 1, 2).contiguous() if self.output_nhwc else x


class HostLoss(ClassificationLoss):
    scale = 1.0

    def forward(self, logits, labels):
        return super().forward(logits, labels) * self.scale


class ObservedMean(nn.Module):
    def forward(self, features, counts):
        values = counts.detach().cpu().tolist()
        if counts.ndim != 1 or counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("Integer observed-eye counts required")
        if any(n not in (1, 2) for n in values) or sum(values) != len(features):
            raise ValueError("Observed-eye ownership mismatch")
        return torch.stack([x.mean(0) for x in features.split(values)])


def build_native_host(first, second, position, device="cpu"):
    """first=CFP, second=OCT; neither input parent is mutated or registered twice."""
    a, b = first.graph, second.graph
    if a.configuration() != b.configuration():
        raise ValueError("This protocol requires same-architecture complete parents")
    config = a.configuration()
    if config.get("views", 1) != 1 or config.get("spatial_dims", 2) != 2:
        raise ValueError("Expanded LOOK host expects observed-eye 2D parents")
    stages = STAGE_MAP[config["name"]]
    split = STARTS[position]
    nodes, edges, definitions, groups, levels = [], [], [], [], {}
    def node(name):
        n = MHD_Node(len(nodes), name, MHD_Node.Message(torch.zeros(1)), aggregation="replace")
        nodes.append(n); return n.id
    roots = {name: node(name) for name in ("oct_input", "cfp_input", "eye_counts", "label_gt")}
    def edge(name, module, inputs, output_name):
        output = node(output_name)
        e = MHD_Edge(len(edges), name, [MHD_Edge.Operation(module)])
        edges.append(e); definitions.append((e.id, inputs, output))
        return e.id, output
    previous = {"oct": roots["oct_input"], "cfp": roots["cfp_input"]}
    source = {"oct": b, "cfp": a}
    sites = ["joint_input"]
    for i, stage in enumerate(stages[:split + 1]):
        group = []
        for modality in ("oct", "cfp"):
            eid, out = edge(f"{modality}_{stage}_edge", NativeCut(source[modality], "input" if i == 0 else stages[i-1], stage),
                            [previous[modality]], f"{modality}_{stage}")
            group.append(eid); previous[modality] = out
        groups.append(group); sites.append(f"joint_{stage}")
    cut = stages[split]
    # The feature width is declared by the native classifier, not guessed from shape.
    if cut == "features":
        head = NativeCut(a, "features", "logits")
        width = next(m.in_features for m in head.modules() if isinstance(m, nn.Linear))
    else:
        width = a.feature_channels[cut]
    projection = ConcatFeatureProjection(width) if cut == "features" else ConcatProjection(width)
    eid, previous_f = edge(f"fuse_{cut}_edge", projection, [previous["oct"], previous["cfp"]], f"fusion_{cut}")
    groups.append([eid]); sites.append(f"fusion_{cut}")
    for i in range(split + 1, len(stages)):
        stage = stages[i]
        eid, previous_f = edge(f"fusion_{stage}_edge", NativeCut(a, stages[i-1], stage), [previous_f], f"fusion_{stage}")
        groups.append([eid]); sites.append(f"fusion_{stage}")
    eid, pooled = edge("fusion_observed_pool_edge", ObservedMean(), [previous_f, roots["eye_counts"]], "fusion_participant_feature")
    groups.append([eid]); sites.append("fusion_participant_feature")
    eid, logits = edge("fusion_classifier_edge", NativeCut(a, "features", "logits"), [pooled], "fusion_logits")
    groups.append([eid])
    eid, _ = edge("classification_loss_edge", HostLoss(), [logits, roots["label_gt"]], "loss")
    groups.append([eid])
    roles, sorts = [], []
    for level, group in enumerate(groups):
        role = torch.zeros(len(edges), len(nodes), dtype=torch.long)
        order = torch.zeros_like(role)
        for e in group:
            _, inputs, out = definitions[e]
            for j, n in enumerate(inputs):
                role[e,n] = -1; order[e,n] = j
            role[e,out] = 1; order[e,out] = len(inputs)
            levels[nodes[out].name] = level
        roles.append(role); sorts.append(order)
    graph = MHD_Graph(set(nodes), set(edges), {MHD_Topo(roles + [-r for r in roles], sorts + [s.clone() for s in sorts])}, device=torch.device(device))
    graph.forward_levels = list(range(len(groups)))
    graph.backward_levels = list(range(2 * len(groups)-1, len(groups)-1, -1))
    graph.model_levels = graph.forward_levels[:-1]
    graph.node_level_map = levels
    graph.correction_nodes = sites
    graph.fusion_position = "native_" + position
    graph.num_classes = config["num_classes"]
    graph.observed_eye_input = True
    graph.architecture_id = config["name"] + "_observed_" + position
    graph.native_host_provenance = dict(schema="look_native_stage_host_v1", stages=stages,
        fusion_endpoint=cut, suffix_initialization="cfp_selected_parent", pooling="valid_eye_feature_mean_v1",
        boundary_layout="channels_first", source_config=config,
        cuts={e.name: e.edge_operations[0].function.definitions for e in edges if isinstance(e.edge_operations[0].function, NativeCut)})
    return graph


def set_observed_counts(graph, counts, eye_count):
    if not getattr(graph, "observed_eye_input", False):
        if counts is not None:
            raise ValueError("Legacy host does not accept variable observed-eye counts")
        return
    if counts is None:
        raise ValueError("Observed-eye host requires counts on every forward")
    values = counts.tolist() if isinstance(counts, torch.Tensor) else list(counts)
    if any(type(n) is not int or n not in (1, 2) for n in values) or sum(values) != eye_count:
        raise ValueError("Invalid observed-eye counts")
    graph.get_node_by_name("eye_counts").feature_message.current_state = torch.tensor(values, dtype=torch.long, device=graph.device)


def forward_host(graph, oct_tensor, cfp_tensor, counts, labels=None, loss_scale=1.0):
    """One MHD forward trace so explicit reverse levels remain valid."""
    from look.methods.operator import _reset_inputs
    _reset_inputs(graph, oct_tensor, cfp_tensor, counts)
    if labels is not None:
        if not 0 < loss_scale <= 1: raise ValueError("Invalid accumulation loss scale")
        graph.get_edge_by_name("classification_loss_edge").edge_operations[0].function.scale = loss_scale
        graph.get_node_by_name("label_gt").feature_message.current_state = labels.long()
    graph.forward(levels=graph.model_levels if labels is None else graph.forward_levels)
    return graph.get_node_by_name("fusion_logits").feature_message.current_state
