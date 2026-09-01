from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_CLASSES = {
    0: "normal",
    1: "diabetes_related_eye_disease",
    2: "glaucoma",
    3: "cataract",
    4: "macular_degeneration",
}


def _square_pad(image: Image.Image, fill: int = 0) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    return TF.pad(image, [left, top, side - width - left, side - height - top], fill=fill)


def crop_fundus_roi(image: Image.Image, margin: float = 0.05) -> Image.Image:
    array = np.asarray(image.convert("RGB"))
    foreground = array.max(axis=2) > 8
    ys, xs = np.where(foreground)
    if not len(xs):
        return _square_pad(image.convert("RGB"))
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = int(max(x1 - x0, y1 - y0) * margin)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(array.shape[1], x1 + pad), min(array.shape[0], y1 + pad)
    return _square_pad(image.crop((x0, y0, x1, y1)).convert("RGB"))


@dataclass(frozen=True)
class PairedAugmentation:
    hflip: bool
    angle: float
    scale: float
    brightness: float
    contrast: float
    saturation: float

    @classmethod
    def sample(cls) -> "PairedAugmentation":
        return cls(
            hflip=random.random() < 0.5,
            angle=random.uniform(-5.0, 5.0),
            scale=random.uniform(0.95, 1.05),
            brightness=random.uniform(0.9, 1.1),
            contrast=random.uniform(0.9, 1.1),
            saturation=random.uniform(0.9, 1.1),
        )


def _geometric_transform(image: Image.Image, aug: PairedAugmentation) -> Image.Image:
    if aug.hflip:
        image = TF.hflip(image)
    return TF.affine(
        image,
        angle=aug.angle,
        translate=[0, 0],
        scale=aug.scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )


class UKBPairedEyeDataset(Dataset):
    """One item is a laterality-specific CFP/OCT eye-visit pair."""

    def __init__(
        self,
        labels_csv: Path,
        data_root: Path,
        split: str,
        image_size: int = 224,
        augment: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        frame = pd.read_csv(labels_csv, dtype={"participant_id": str})
        frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        if limit is not None:
            frame = frame.iloc[:limit].copy()
        self.frame = frame
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.augment = augment
        self.labels = frame["label_id"].astype(int).to_numpy()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.frame.iloc[index]
        cfp = crop_fundus_roi(Image.open(self.data_root / row.fundus_path))
        oct_image = _square_pad(Image.open(self.data_root / row.oct_path).convert("L")).convert("RGB")
        cfp = TF.resize(cfp, [self.image_size, self.image_size], antialias=True)
        oct_image = TF.resize(oct_image, [self.image_size, self.image_size], antialias=True)

        if self.augment:
            aug = PairedAugmentation.sample()
            cfp = _geometric_transform(cfp, aug)
            oct_image = _geometric_transform(oct_image, aug)
            cfp = TF.adjust_brightness(cfp, aug.brightness)
            cfp = TF.adjust_contrast(cfp, aug.contrast)
            cfp = TF.adjust_saturation(cfp, aug.saturation)
            oct_image = TF.adjust_brightness(oct_image, aug.brightness)
            oct_image = TF.adjust_contrast(oct_image, aug.contrast)

        cfp_tensor = TF.normalize(TF.to_tensor(cfp), IMAGENET_MEAN, IMAGENET_STD)
        oct_tensor = TF.normalize(TF.to_tensor(oct_image), IMAGENET_MEAN, IMAGENET_STD)
        return {
            "oct": oct_tensor,
            "cfp": cfp_tensor,
            "label": int(row.label_id),
            "participant_id": str(row.participant_id),
            "eye": row.eye,
            "instance": int(row.instance),
        }


def validate_reference_table(labels_csv: Path, data_root: Path, check_paths: bool = False) -> Dict[str, object]:
    frame = pd.read_csv(labels_csv, dtype={"participant_id": str})
    required = {
        "participant_id", "instance", "eye", "fundus_path", "oct_path",
        "label_id", "label_name", "split", "reference_source",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing reference columns: {sorted(missing)}")
    observed = dict(
        frame[["label_id", "label_name"]].drop_duplicates().sort_values("label_id").values
    )
    if observed != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected class mapping: {observed}")
    participant_splits = frame.groupby("participant_id")["split"].nunique()
    if int(participant_splits.max()) != 1:
        raise ValueError("Participant leakage detected across splits")
    missing_paths = []
    if check_paths:
        for column in ("fundus_path", "oct_path"):
            for relative in frame[column]:
                if not (Path(data_root) / relative).is_file():
                    missing_paths.append(relative)
                    if len(missing_paths) >= 20:
                        break
    if missing_paths:
        raise FileNotFoundError(f"Missing image files (first 20): {missing_paths}")
    return {
        "rows": len(frame),
        "participants": int(frame.participant_id.nunique()),
        "split_counts": frame.split.value_counts().sort_index().to_dict(),
        "class_counts": frame.label_name.value_counts().to_dict(),
        "participant_split_leakage": 0,
    }


def weighted_sampler(labels: Sequence[int], power: float = 0.5, seed: int = 0) -> WeightedRandomSampler:
    labels_array = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_array)
    class_weights = np.power(counts, -power, where=counts > 0)
    weights = torch.as_tensor(class_weights[labels_array], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)


def make_loader(
    dataset: UKBPairedEyeDataset,
    batch_size: int,
    num_workers: int,
    train: bool,
    seed: int,
    sampler_power: float = 0.5,
) -> DataLoader:
    sampler = weighted_sampler(dataset.labels, sampler_power, seed) if train else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=train,
    )


def participant_missing_pattern(participant_id: str, ratio: float, seed: int) -> str:
    """Stable participant-level assignment: complete, OCT missing, or CFP missing."""
    token = f"{seed}:{participant_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
    if value >= ratio:
        return "complete"
    return "oct_missing" if value < ratio / 2 else "cfp_missing"


def apply_missingness(
    oct_tensor: torch.Tensor,
    cfp_tensor: torch.Tensor,
    patterns: Iterable[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    oct_result, cfp_result = oct_tensor.clone(), cfp_tensor.clone()
    for index, pattern in enumerate(patterns):
        if pattern == "oct_missing":
            oct_result[index].zero_()
        elif pattern == "cfp_missing":
            cfp_result[index].zero_()
        elif pattern != "complete":
            raise ValueError(f"Unknown missing pattern: {pattern}")
    return oct_result, cfp_result
