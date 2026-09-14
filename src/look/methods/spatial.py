"""Fixed linear spatial reductions for frozen-host LOOK, without learned layers."""
import torch.nn.functional as F

METHODS = ('interpolate', 'direct', 'average_pool')


def reduce_spatial(feature, factor, method):
    if method not in METHODS or type(factor) is not int or factor < 1:
        raise ValueError('Unknown spatial method or nonpositive integer factor')
    if method == 'direct' and factor != 1:
        raise ValueError('Direct PCA preserves spatial resolution; factor must be one')
    if feature.ndim not in (2, 4):
        raise ValueError('This LOOK protocol supports vector and 2D spatial features')
    if feature.ndim == 2 or method == 'direct':
        return feature
    size = tuple(max(1, n // factor) for n in feature.shape[-2:])
    if method == 'average_pool':
        return F.adaptive_avg_pool2d(feature, size)
    return F.interpolate(feature, size=size, mode='bilinear', align_corners=False, antialias=False)
