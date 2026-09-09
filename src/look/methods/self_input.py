"""Remove sample-specific retained-branch information, retaining the shared PCA.

This is an information-source ablation, not a branch-specific PCA or heterogeneous
backbone implementation. Mean-clamping occurs BEFORE PCA projection, not after:
a joint PCA latent otherwise already contains retained-branch information.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from look.methods.operator import downsample_flatten, apply_artifact

POLICY = 'self_input_missing_only'

def neutralize_retained(flat, basis, pattern):
    if not basis.node_name.startswith('joint_'):
        return flat
    if pattern not in ('oct_missing', 'cfp_missing'):
        raise ValueError('Self-input correction requires a fixed missing direction')
    expected = ('oct_' + basis.node_name[6:], 'cfp_' + basis.node_name[6:])
    if tuple(basis.member_names) != expected or len(basis.member_shapes) != 2:
        raise ValueError('Self-input correction requires explicit ordered member metadata')
    channel_axis = 1 if basis.node_name == 'joint_input' else 0
    channels = [s[channel_axis] for s in basis.member_shapes]
    shape = tuple(basis.downsample_shape)
    if sum(channels) != shape[0] or flat.shape[1] != math.prod(shape):
        raise ValueError('Self-input PCA member geometry mismatch')
    cut = channels[0] * math.prod(shape[1:])
    retained = slice(cut, None) if pattern == 'oct_missing' else slice(0, cut)
    out = flat.clone()
    out[:, retained] = basis.mean.to(flat.device)[retained]
    return out


def apply_self_input(feature, artifact):
    if not artifact.node_name.startswith('joint_'):
        return apply_artifact(feature, artifact)
    flat, shape = downsample_flatten(feature, artifact.factor)
    if tuple(shape) != tuple(artifact.downsample_shape):
        raise ValueError('Self-input artifact shape mismatch')
    flat = neutralize_retained(flat, artifact, artifact.missing_pattern)
    mean, std, pca_mean, components, weight, bias = [x.to(feature.device) for x in
        (artifact.mean, artifact.std, artifact.pca_mean, artifact.components, artifact.weight, artifact.bias)]
    latent = ((flat - mean) / std - pca_mean) @ components.T
    residual = (((latent @ weight + bias) @ components) * std).reshape(feature.shape[0], *shape)
    if feature.ndim == 4 and residual.shape[-2:] != feature.shape[-2:]:
        residual = F.interpolate(residual, size=feature.shape[-2:], mode='bilinear', align_corners=False)
    return feature + residual
