from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

from .data import IMAGENET_MEAN, IMAGENET_STD


class MissingModalityFiller(ABC):
    name: str

    @abstractmethod
    def fill(
        self,
        oct_tensor: torch.Tensor,
        cfp_tensor: torch.Tensor,
        missing_pattern: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class NormalizedMeanFiller(MissingModalityFiller):
    """Zero after ImageNet normalization, i.e. the normalization channel mean."""

    name = "normalized_mean"

    def fill(self, oct_tensor, cfp_tensor, missing_pattern):
        if missing_pattern == "oct_missing":
            return torch.zeros_like(oct_tensor), cfp_tensor
        if missing_pattern == "cfp_missing":
            return oct_tensor, torch.zeros_like(cfp_tensor)
        if missing_pattern == "complete":
            return oct_tensor, cfp_tensor
        raise ValueError(f"Unknown missing pattern: {missing_pattern}")


class PairedCGANFiller(MissingModalityFiller):
    name = "paired_cgan"

    def __init__(self, cfp_to_oct, oct_to_cfp, device: torch.device) -> None:
        self.generators = {
            "cfp_to_oct": cfp_to_oct.to(device).eval(),
            "oct_to_cfp": oct_to_cfp.to(device).eval(),
        }
        self.device = device

    @staticmethod
    def _classifier_to_gan(tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.new_tensor(IMAGENET_MEAN)[None, :, None, None]
        std = tensor.new_tensor(IMAGENET_STD)[None, :, None, None]
        return ((tensor * std + mean).clamp(0, 1) * 2) - 1

    @staticmethod
    def _gan_to_classifier(tensor: torch.Tensor) -> torch.Tensor:
        image = (tensor.clamp(-1, 1) + 1) / 2
        mean = image.new_tensor(IMAGENET_MEAN)[None, :, None, None]
        std = image.new_tensor(IMAGENET_STD)[None, :, None, None]
        return (image - mean) / std

    @torch.no_grad()
    def fill(self, oct_tensor, cfp_tensor, missing_pattern):
        if missing_pattern == "oct_missing":
            generated = self.generators["cfp_to_oct"](self._classifier_to_gan(cfp_tensor))
            return self._gan_to_classifier(generated), cfp_tensor
        if missing_pattern == "cfp_missing":
            generated = self.generators["oct_to_cfp"](self._classifier_to_gan(oct_tensor))
            return oct_tensor, self._gan_to_classifier(generated)
        if missing_pattern == "complete":
            return oct_tensor, cfp_tensor
        raise ValueError(f"Unknown missing pattern: {missing_pattern}")


def fill_mixed_batch(
    filler: MissingModalityFiller,
    oct_tensor: torch.Tensor,
    cfp_tensor: torch.Tensor,
    patterns: Iterable[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    patterns = list(patterns)
    oct_result, cfp_result = oct_tensor.clone(), cfp_tensor.clone()
    for pattern in sorted(set(patterns)):
        if pattern == "complete":
            continue
        indices = [index for index, value in enumerate(patterns) if value == pattern]
        index_tensor = torch.as_tensor(indices, device=oct_tensor.device)
        filled_oct, filled_cfp = filler.fill(
            oct_tensor.index_select(0, index_tensor),
            cfp_tensor.index_select(0, index_tensor),
            pattern,
        )
        oct_result[index_tensor] = filled_oct
        cfp_result[index_tensor] = filled_cfp
    return oct_result, cfp_result
