"""MMTM channel gates for adaptation audit only; no approved training arm yet."""
import torch
from torch import nn


class MMTM(nn.Module):
    def __init__(self,first_channels,second_channels,ratio,gate_scale):
        super().__init__()
        if ratio<=0 or gate_scale not in (1.,2.):raise ValueError('Explicit author-code or paper gate scale required')
        width=int(2*(first_channels+second_channels)/ratio)
        if width<1:raise ValueError('Invalid MMTM reduction')
        self.fc_squeeze=nn.Linear(first_channels+second_channels,width)
        self.fc_visual=nn.Linear(width,first_channels)
        self.fc_skeleton=nn.Linear(width,second_channels)
        self.gate_scale=float(gate_scale)

    def forward(self,first,second):
        if first.shape[0]!=second.shape[0]:raise ValueError('Unpaired MMTM inputs')
        descriptor=torch.cat([x.flatten(2).mean(-1) if x.ndim>2 else x for x in (first,second)],1)
        shared=torch.relu(self.fc_squeeze(descriptor))
        outputs=[]
        for x,head in ((first,self.fc_visual),(second,self.fc_skeleton)):
            gate=self.gate_scale*torch.sigmoid(head(shared))
            outputs.append(x*gate.reshape(*gate.shape,*([1]*(x.ndim-2))))
        return tuple(outputs)
