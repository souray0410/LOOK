from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
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
PREPROCESS_CACHE_VERSION = "paired_roi_square_resize_rgb_v1"


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


def _preprocess_pair(
    data_root: Path,
    fundus_path: str,
    oct_path: str,
    image_size: int,
) -> tuple[Image.Image, Image.Image]:
    with Image.open(data_root / fundus_path) as source:
        cfp = crop_fundus_roi(source)
    with Image.open(data_root / oct_path) as source:
        oct_image = _square_pad(source.convert("L")).convert("RGB")
    cfp = TF.resize(cfp, [image_size, image_size], antialias=True)
    oct_image = TF.resize(oct_image, [image_size, image_size], antialias=True)
    return cfp, oct_image


def preprocess_cache_path(
    fundus_path: str,
    oct_path: str,
    image_size: int,
    cache_root: Path,
) -> Path:
    identity = f"{PREPROCESS_CACHE_VERSION}\0{image_size}\0{fundus_path}\0{oct_path}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return cache_root / PREPROCESS_CACHE_VERSION / digest[:2] / f"{digest}.npy"


def ensure_preprocessed_pair_cache(
    data_root: Path,
    fundus_path: str,
    oct_path: str,
    image_size: int,
    cache_root: Path,
) -> Path:
    cache_path = preprocess_cache_path(fundus_path, oct_path, image_size, cache_root)
    if cache_path.is_file():
        try:
            pair = np.load(cache_path, allow_pickle=False, mmap_mode="r")
            if pair.shape == (2, image_size, image_size, 3) and pair.dtype == np.uint8:
                return cache_path
        except (OSError, ValueError):
            pass

    cfp, oct_image = _preprocess_pair(data_root, fundus_path, oct_path, image_size)
    pair = np.stack(
        (np.asarray(cfp, dtype=np.uint8), np.asarray(oct_image, dtype=np.uint8)), axis=0
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f"{cache_path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, pair, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return cache_path


def _cached_preprocess_pair(
    data_root: Path,
    fundus_path: str,
    oct_path: str,
    image_size: int,
    cache_root: Path | None,
) -> tuple[Image.Image, Image.Image]:
    if cache_root is None:
        return _preprocess_pair(data_root, fundus_path, oct_path, image_size)
    cache_path = ensure_preprocessed_pair_cache(
        data_root, fundus_path, oct_path, image_size, cache_root
    )
    pair = np.load(cache_path, allow_pickle=False)
    return Image.fromarray(pair[0]), Image.fromarray(pair[1])


@dataclass(frozen=True)
class PairedAugmentation:
    hflip: bool
    angle: float
    scale: float
    brightness: float
    contrast: float
    saturation: float

    @classmethod
    def sample(cls, rng: random.Random | None = None) -> "PairedAugmentation":
        rng = rng or random
        return cls(
            hflip=rng.random() < 0.5,
            angle=rng.uniform(-5.0, 5.0),
            scale=rng.uniform(0.95, 1.05),
            brightness=rng.uniform(0.9, 1.1),
            contrast=rng.uniform(0.9, 1.1),
            saturation=rng.uniform(0.9, 1.1),
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
        base_seed: int = 3407,
        preprocess_cache_root: Path | None = None,
    ) -> None:
        frame = pd.read_csv(labels_csv, dtype={"participant_id": str})
        frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        if limit is not None:
            groups = {
                int(label): list(group.index)
                for label, group in frame.groupby("label_id", sort=True)
            }
            selected = []
            offset = 0
            while len(selected) < limit and groups:
                exhausted = []
                for label, indices in groups.items():
                    if offset < len(indices):
                        selected.append(indices[offset])
                        if len(selected) == limit:
                            break
                    else:
                        exhausted.append(label)
                for label in exhausted:
                    groups.pop(label)
                offset += 1
            frame = frame.loc[selected].reset_index(drop=True)
        self.frame = frame
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.augment = augment
        self.labels = frame["label_id"].astype(int).to_numpy()
        self.fundus_paths = frame["fundus_path"].astype(str).tolist()
        self.oct_paths = frame["oct_path"].astype(str).tolist()
        self.participant_ids = frame["participant_id"].astype(str).tolist()
        self.eyes = frame["eye"].astype(str).tolist()
        self.instances = frame["instance"].astype(int).to_numpy()
        self.base_seed = int(base_seed)
        self.preprocess_cache_root = (
            Path(preprocess_cache_root) if preprocess_cache_root is not None else None
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, object]:
        participant_id = self.participant_ids[index]
        eye = self.eyes[index]
        instance = int(self.instances[index])
        cfp, oct_image = _cached_preprocess_pair(
            self.data_root,
            self.fundus_paths[index],
            self.oct_paths[index],
            self.image_size,
            self.preprocess_cache_root,
        )

        if self.augment:
            identity = f"{self.base_seed}:{self.epoch}:{participant_id}:{eye}:{instance}"
            sample_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
            aug = PairedAugmentation.sample(random.Random(sample_seed))
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
            "label": int(self.labels[index]),
            "participant_id": participant_id,
            "eye": eye,
            "instance": instance,
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


def reference_training_class_counts(labels_csv: Path, num_classes: int) -> list[int]:
    """Return fixed class counts from the complete training split, never a smoke subset."""
    frame = pd.read_csv(labels_csv, usecols=["split", "label_id"])
    labels = frame.loc[frame["split"] == "train", "label_id"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=num_classes).tolist()
    if len(counts) != num_classes or any(count <= 0 for count in counts):
        raise ValueError(f"The complete training split must contain all {num_classes} classes: {counts}")
    return counts


class DistributedShuffleSampler(Sampler[int]):
    """Deterministically shuffle each sample once and shard without padding."""

    def __init__(self, size: int, seed: int, rank: int, world_size: int):
        self.size = int(size)
        self.seed, self.rank, self.world_size, self.epoch = seed, rank, world_size, 0
        self.global_size = (self.size // world_size) * world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator)[:self.global_size]
        return iter(indices[self.rank:self.global_size:self.world_size].tolist())

    def __len__(self) -> int:
        return self.global_size // self.world_size


class DistributedClassBalancedSampler(Sampler[int]):
    """Uniformly sample classes, then examples, and shard one global draw stream."""

    def __init__(self, labels, seed: int, rank: int, world_size: int):
        values = np.asarray(labels, dtype=np.int64)
        self.class_indices = [
            torch.as_tensor(np.flatnonzero(values == label), dtype=torch.long)
            for label in sorted(np.unique(values).tolist())
        ]
        if len(self.class_indices) < 2 or any(len(indices) == 0 for indices in self.class_indices):
            raise ValueError("Class-balanced sampling requires at least two non-empty classes")
        self.size = len(values)
        self.seed, self.rank, self.world_size, self.epoch = seed, rank, world_size, 0
        self.global_size = (self.size // world_size) * world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        class_draws = torch.randint(
            len(self.class_indices), (self.global_size,), generator=generator
        )
        samples = torch.empty(self.global_size, dtype=torch.long)
        for class_id, indices in enumerate(self.class_indices):
            positions = torch.where(class_draws == class_id)[0]
            if len(positions):
                choices = torch.randint(len(indices), (len(positions),), generator=generator)
                samples[positions] = indices[choices]
        return iter(samples[self.rank:self.global_size:self.world_size].tolist())

    def __len__(self) -> int:
        return self.global_size // self.world_size


class DistributedEvalSampler(Sampler[int]):
    """Non-padding evaluation shard: every row appears on exactly one rank."""

    def __init__(self, size: int, rank: int, world_size: int):
        self.indices = list(range(rank, size, world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def make_loader(
    dataset: UKBPairedEyeDataset,
    batch_size: int,
    num_workers: int,
    train: bool,
    seed: int,
    sampling_strategy: str = "natural_without_replacement",
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    if train and sampling_strategy == "natural_without_replacement":
        sampler = DistributedShuffleSampler(len(dataset), seed, rank, world_size)
    elif train and sampling_strategy == "class_balanced_with_replacement":
        sampler = DistributedClassBalancedSampler(
            dataset.labels, seed, rank, world_size
        )
    elif not train:
        sampler = DistributedEvalSampler(len(dataset), rank, world_size)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        multiprocessing_context="spawn" if num_workers > 0 else None,
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
