"""Source-pinned core for MICCAI-2025 improved modality dropout.

This implements the author TNF MLP fusion, learnable missing tokens,
simultaneous target-task modality dropout, and the author sigmoid contrastive
primitive. It is a CFP/OCT project adaptation; it is not a reproduction of the
authors' image+tabular datasets or full training pipeline.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

AUTHOR_REPOSITORY="omron-sinicx/medical-modality-dropout"
AUTHOR_COMMIT="8040d96b2dec48cf8fc7d13b45e15af0d07952ed"
AUTHOR_LICENSE="MIT"
AUTHOR_FILE_SHA256={
    "LICENSE":"37c8a634b6163f3d097ebab2326ceef793769000e08a06263544bdf421ad25b6",
    "README.md":"522b0d7e5001541f1336e6947f27b1adae35f69c035e8f8fff78692cef1d89f9",
    "losses/contrastive_loss.py":"c1b5a48eb2f99cb4934a49910639b724c66013fd6bfea7f778d5fce9fbeab549",
    "networks/tnf.py":"c5a9d5c72bb1545df07f563ce22df2ec6908492cb652ccd259500558d8f41c18",
}


class EmptyToken(nn.Module):
    """Author EmptyToken semantics: learnable zero-initialized vector."""
    def __init__(self,dim:int):
        super().__init__()
        if dim<1: raise ValueError("Token dimension must be positive")
        self.token=nn.Parameter(torch.zeros(1,dim))

    def forward(self,batch_size:int)->torch.Tensor:
        if batch_size<1: raise ValueError("Batch size must be positive")
        return self.token.expand(batch_size,-1)


class AuthorMLP(nn.Module):
    """Author MLP order: LayerNorm -> ReLU -> Dropout -> Linear."""
    def __init__(self,in_channels:int,out_channels:int,*,hidden_channels=(),dropout:float=.1):
        super().__init__()
        if isinstance(hidden_channels,int): hidden_channels=(hidden_channels,)
        widths=[in_channels,*hidden_channels,out_channels]
        self.norms=nn.ModuleList(nn.LayerNorm(widths[i]) for i in range(len(widths)-1))
        self.projs=nn.ModuleList(nn.Linear(widths[i],widths[i+1]) for i in range(len(widths)-1))
        self.dropouts=nn.ModuleList(nn.Dropout(dropout) if dropout>0 else nn.Identity() for _ in self.projs)

    def forward(self,x):
        for norm,drop,proj in zip(self.norms,self.dropouts,self.projs):
            x=proj(drop(F.relu(norm(x))))
        return x


class ImprovedDropoutFusion(nn.Module):
    """TNF MLP fusion adapted to two participant-level image features."""
    def __init__(self,oct_width:int,cfp_width:int,num_classes:int,*,dropout:float=.1,classifier_dropout:float=.1):
        super().__init__()
        if min(oct_width,cfp_width,num_classes)<1: raise ValueError("Positive dimensions required")
        hidden=round((oct_width+cfp_width)/2.)
        self.empty_oct=EmptyToken(oct_width)
        self.empty_cfp=EmptyToken(cfp_width)
        self.norm_oct=nn.LayerNorm(oct_width)
        self.norm_cfp=nn.LayerNorm(cfp_width)
        self.fusor=AuthorMLP(oct_width+cfp_width,hidden,dropout=dropout)
        self.classifier_norm=nn.LayerNorm(hidden)
        self.classifier_dropout=nn.Dropout(classifier_dropout) if classifier_dropout>0 else nn.Identity()
        self.classifier=nn.Linear(hidden,num_classes)
        self.hidden_width=hidden
        self.provenance={
            "author_repository":AUTHOR_REPOSITORY,
            "author_commit":AUTHOR_COMMIT,
            "author_license":AUTHOR_LICENSE,
            "author_file_sha256":dict(AUTHOR_FILE_SHA256),
            "adaptation":"two-image CFP/OCT participant features; not author image+tabular dataset reproduction",
            "missingness":"learnable zero-initialized modality tokens at fusion input",
        }

    def fused_feature(self,oct_feature,cfp_feature,*,state="complete"):
        if oct_feature.ndim!=2 or cfp_feature.ndim!=2 or len(oct_feature)!=len(cfp_feature):
            raise ValueError("Aligned participant feature matrices required")
        if isinstance(state, torch.Tensor):
            if state.ndim != 1 or len(state) != len(oct_feature):
                raise ValueError("One missing-state code per participant required")
            codes = state.to(device=oct_feature.device, dtype=torch.long)
            oct_value = oct_feature.clone()
            cfp_value = cfp_feature.clone()
            oct_value[codes == 1] = self.empty_oct(len(oct_feature)).to(oct_value)[codes == 1]
            cfp_value[codes == 2] = self.empty_cfp(len(cfp_feature)).to(cfp_value)[codes == 2]
        elif state=="complete": oct_value,cfp_value=oct_feature,cfp_feature
        elif state=="oct_missing": oct_value,cfp_value=self.empty_oct(len(cfp_feature)),cfp_feature
        elif state=="cfp_missing": oct_value,cfp_value=oct_feature,self.empty_cfp(len(oct_feature))
        else: raise ValueError("Unknown modality state")
        return self.fusor(torch.cat((self.norm_oct(oct_value),self.norm_cfp(cfp_value)),dim=1))

    def forward(self,oct_feature,cfp_feature,*,state="complete"):
        fused=self.fused_feature(oct_feature,cfp_feature,state=state)
        return self.classifier(self.classifier_dropout(self.classifier_norm(fused)))


def simultaneous_modality_dropout_loss(model,oct_feature,cfp_feature,labels,*,missing_weight:float=1.):
    """Paper target-task objective: complete CE + lambda*(two single-modality CEs)."""
    if missing_weight<0 or not math.isfinite(float(missing_weight)):
        raise ValueError("Finite non-negative missing weight required")
    logits={state:model(oct_feature,cfp_feature,state=state) for state in ("complete","oct_missing","cfp_missing")}
    losses={state:F.cross_entropy(value,labels.long()) for state,value in logits.items()}
    total=losses["complete"]+missing_weight*(losses["oct_missing"]+losses["cfp_missing"])
    return total,logits


class SigmoidContrastiveLoss(nn.Module):
    """Author BCE/sigmoid supervised contrastive primitive."""
    def __init__(self):
        super().__init__()
        self.log_scale=nn.Parameter(torch.log(torch.tensor(10.,dtype=torch.float32)))
        self.bias=nn.Parameter(torch.tensor(-10.,dtype=torch.float32))

    def forward(self,query,key,labels):
        if query.ndim!=2 or key.ndim!=2 or query.shape!=key.shape:
            raise ValueError("Aligned 2-D contrastive representations required")
        if labels.ndim!=1 or len(labels)!=len(query):
            raise ValueError("One label per participant required")
        targets=(labels[:,None]==labels[None,:]).to(query)
        logits=(query@key.T)*torch.exp(self.log_scale)+self.bias
        return F.binary_cross_entropy_with_logits(logits,targets)


def multimodal_contrastive_loss(z_oct,z_cfp,z_fused,labels,criterion):
    """Registered three-pair fused/unimodal contrastive objective."""
    return criterion(z_oct,z_cfp,labels)+criterion(z_oct,z_fused,labels)+criterion(z_cfp,z_fused,labels)


class ImprovedDropoutMHDLogitsOperation(nn.Module):
    """MHD edge wrapper for the adapted author TNF/EmptyToken fusion."""
    def __init__(self, fusion: ImprovedDropoutFusion):
        super().__init__()
        self.fusion = fusion

    def forward(self, oct_feature: torch.Tensor, cfp_feature: torch.Tensor, state_code: torch.Tensor) -> torch.Tensor:
        return self.fusion(oct_feature, cfp_feature, state=state_code)


def _imd_state_codes(state, batch_size: int, device: torch.device) -> torch.Tensor:
    mapping = {"complete": 0, "oct_missing": 1, "cfp_missing": 2}
    if isinstance(state, str):
        if state not in mapping: raise ValueError("Unknown IMD state")
        return torch.full((batch_size,), mapping[state], dtype=torch.long, device=device)
    value = torch.as_tensor(state, dtype=torch.long, device=device)
    if value.ndim != 1 or len(value) != batch_size or not torch.isin(value, torch.tensor([0,1,2], device=device)).all():
        raise ValueError("IMD state code must be 0/1/2 per participant")
    return value


def build_improved_dropout_host(first, second, device="cpu", *, hidden_dropout: float = 0.1, classifier_dropout: float = 0.1) -> "MHD_Graph":
    """Build a frozen-encoder CFP/OCT MHD host for Improved Modality Dropout.

    The encoders are copied from two complete observed-eye parents and frozen.
    Only the author-style EmptyToken/TNF fusion and task head are trainable.
    """
    from mhd_framework.core import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
    from look.models.native_host import HostLoss, NativeCut, ObservedMean

    a, b = first.graph, second.graph
    if a.configuration() != b.configuration():
        raise ValueError("Improved Modality Dropout requires matched observed-eye parents")
    config = a.configuration()
    if config.get("views", 1) != 1 or config.get("spatial_dims", 2) != 2:
        raise ValueError("IMD host expects observed-eye 2D parents")
    for parameter in first.parameters(): parameter.requires_grad_(False)
    for parameter in second.parameters(): parameter.requires_grad_(False)

    nodes, edges, definitions, groups, levels = [], [], [], [], {}
    def node(name):
        item = MHD_Node(len(nodes), name, MHD_Node.Message(torch.zeros(1, device=device)), aggregation="sum", memory=False)
        nodes.append(item); return item.id
    roots = {name: node(name) for name in ("oct_input", "cfp_input", "eye_counts", "imd_state_code", "label_gt")}
    def edge(name, module, inputs, output_name):
        output = node(output_name)
        item = MHD_Edge(len(edges), name, [MHD_Edge.Operation(module)])
        edges.append(item); definitions.append((item.id, inputs, output)); return item.id, output

    source = {"cfp": a, "oct": b}
    previous = {"cfp": roots["cfp_input"], "oct": roots["oct_input"]}
    groups.append([])
    sites = []
    for modality in ("oct", "cfp"):
        eid, feat = edge(f"{modality}_features_edge", NativeCut(source[modality], "input", "features"), [previous[modality]], f"{modality}_features")
        groups.append([eid]); sites.append(f"{modality}_features")
        eid, pooled = edge(f"{modality}_observed_pool_edge", ObservedMean(), [feat, roots["eye_counts"]], f"{modality}_participant_feature")
        groups.append([eid]); sites.append(f"{modality}_participant_feature")
        previous[modality] = pooled
    width = next(module.in_features for module in NativeCut(a, "features", "logits").modules() if isinstance(module, nn.Linear))
    fusion = ImprovedDropoutFusion(width, width, config["num_classes"], dropout=hidden_dropout, classifier_dropout=classifier_dropout)
    eid, logits = edge("improved_dropout_logits_edge", ImprovedDropoutMHDLogitsOperation(fusion), [previous["oct"], previous["cfp"], roots["imd_state_code"]], "fusion_logits")
    groups.append([eid]); sites.append("fusion_logits")
    eid, _ = edge("classification_loss_edge", HostLoss(), [logits, roots["label_gt"]], "loss")
    groups.append([eid])

    roles, sorts = [], []
    for level, group in enumerate(groups):
        role = torch.zeros(len(edges), len(nodes), dtype=torch.long)
        order = torch.zeros_like(role)
        for e in group:
            _, inputs, out = definitions[e]
            for j, n in enumerate(inputs):
                role[e, n] = -1; order[e, n] = j
            role[e, out] = 1; order[e, out] = len(inputs)
            levels[nodes[out].name] = level
        roles.append(role); sorts.append(order)
    graph = MHD_Graph(set(nodes), set(edges), {MHD_Topo(roles + [-r for r in roles], sorts + [s.clone() for s in sorts])}, device=torch.device(device))
    graph.forward_levels = list(range(len(groups)))
    graph.backward_levels = list(range(2 * len(groups)-1, len(groups)-1, -1))
    graph.model_levels = graph.forward_levels[:-1]
    graph.node_level_map = levels
    graph.correction_nodes = sites
    graph.fusion_position = "improved_dropout_features"
    graph.num_classes = config["num_classes"]
    graph.observed_eye_input = True
    graph.architecture_id = config["name"] + "_observed_improved_dropout"
    graph.improved_dropout_provenance = {
        "schema": "look_improved_modality_dropout_mhd_v1",
        "author_repository": AUTHOR_REPOSITORY,
        "author_commit": AUTHOR_COMMIT,
        "author_license": AUTHOR_LICENSE,
        "encoder_policy": "frozen complete observed-eye parents; fusion/head trainable",
        "missing_state_codes": {"complete": 0, "oct_missing": 1, "cfp_missing": 2},
        "adaptation": "CFP/OCT participant features; not author image+tabular dataset reproduction",
    }
    return graph


def set_improved_dropout_state(graph, state, batch_size: int) -> None:
    graph.get_node_by_name("imd_state_code").feature_message.current_state = _imd_state_codes(state, batch_size, graph.device)


def forward_improved_dropout_host(graph, oct_tensor, cfp_tensor, counts, labels=None, state="complete", loss_scale=1.0):
    from look.methods.operator import _reset_inputs
    from look.models.native_host import set_observed_counts
    _reset_inputs(graph, oct_tensor, cfp_tensor, counts)
    set_observed_counts(graph, counts, len(oct_tensor))
    set_improved_dropout_state(graph, state, len(counts))
    if labels is not None:
        if not 0 < loss_scale <= 1: raise ValueError("Invalid accumulation loss scale")
        graph.get_edge_by_name("classification_loss_edge").edge_operations[0].function.scale = loss_scale
        graph.get_node_by_name("label_gt").feature_message.current_state = labels.long()
    graph.forward(levels=graph.model_levels if labels is None else graph.forward_levels)
    return graph.get_node_by_name("fusion_logits").feature_message.current_state


def improved_dropout_parameter_groups(graph, fusion_lr: float):
    trainable = [p for p in graph.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("IMD host has no trainable fusion parameters")
    return [{"params": trainable, "lr": fusion_lr, "group_name": "improved_dropout_fusion_and_head"}]
