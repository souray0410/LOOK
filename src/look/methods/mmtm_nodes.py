"""Explicit MHD nodes for the audited MMTM gates, without a training policy."""
import torch
from torch import nn
from look.methods.external_mmtm import MMTM


class JointDescriptor(nn.Module):
    def forward(self, first, second):
        if first.shape[0] != second.shape[0]:
            raise ValueError('Unpaired MMTM inputs')
        return torch.cat([x.flatten(2).mean(-1) if x.ndim>2 else x for x in (first,second)],1)


class ApplyGate(nn.Module):
    def __init__(self, linear, scale):
        super().__init__(); self.linear=linear; self.scale=float(scale)

    def forward(self, feature, shared):
        gate=self.scale*torch.sigmoid(self.linear(shared))
        return feature*gate.reshape(*gate.shape,*([1]*(feature.ndim-2)))


def add_mmtm_nodes(edge, groups, previous, channels, *, ratio, gate_scale, stage):
    """Use the host builder; OCT is author first/visual, CFP second/skeleton.

    Each head owns distinct parameters; shared squeeze is evaluated once. All
    new parameters use fuse_ names so the host's new-layer LR applies.
    """
    module=MMTM(channels,channels,ratio,gate_scale)
    e, descriptor=edge('fuse_mmtm_descriptor_edge',JointDescriptor(),
                       [previous['oct'],previous['cfp']],'mmtm_descriptor')
    groups.append([e])
    e, shared=edge('fuse_mmtm_squeeze_edge',nn.Sequential(module.fc_squeeze,nn.ReLU()),
                   [descriptor],'mmtm_shared')
    groups.append([e]); gates=[]
    for modality,linear in [('oct',module.fc_visual),('cfp',module.fc_skeleton)]:
        e,out=edge('fuse_mmtm_'+modality+'_edge',ApplyGate(linear,gate_scale),
                    [previous[modality],shared],'mmtm_'+modality+'_'+stage)
        gates.append(e);previous[modality]=out
    groups.append(gates)
