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
        if state=="complete": oct_value,cfp_value=oct_feature,cfp_feature
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
