import numpy as np
import pandas as pd
import torch
from PIL import Image

from look_core.data import (
    DistributedEvalSampler,
    DistributedShuffleSampler,
    UKBPairedEyeDataset,
    apply_missingness,
    participant_missing_pattern,
    reference_training_class_counts,
)


def test_preprocess_cache_is_lossless_atomic_and_self_repairing(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    fundus = np.zeros((18, 24, 3), dtype=np.uint8)
    fundus[2:16, 3:21] = np.array([120, 80, 40], dtype=np.uint8)
    oct_image = np.arange(20 * 12, dtype=np.uint8).reshape(20, 12)
    Image.fromarray(fundus).save(image_root / "fundus.png")
    Image.fromarray(oct_image).save(image_root / "oct.png")
    labels = tmp_path / "reference_labels.csv"
    pd.DataFrame([{
        "participant_id": "1001",
        "eye": "left",
        "instance": 0,
        "split": "train",
        "label_id": 0,
        "fundus_path": "fundus.png",
        "oct_path": "oct.png",
    }]).to_csv(labels, index=False)

    uncached = UKBPairedEyeDataset(labels, image_root, "train", image_size=16)[0]
    cache_root = tmp_path / "cache"
    dataset = UKBPairedEyeDataset(
        labels, image_root, "train", image_size=16, preprocess_cache_root=cache_root
    )
    cached = dataset[0]
    assert torch.equal(cached["cfp"], uncached["cfp"])
    assert torch.equal(cached["oct"], uncached["oct"])
    cache_files = list(cache_root.rglob("*.npy"))
    assert len(cache_files) == 1
    assert not list(cache_root.rglob("*.partial.*"))

    cache_files[0].write_bytes(b"truncated")
    repaired = dataset[0]
    assert torch.equal(repaired["cfp"], uncached["cfp"])
    assert torch.equal(repaired["oct"], uncached["oct"])
    assert np.load(cache_files[0], allow_pickle=False).shape == (2, 16, 16, 3)


def test_missing_pattern_is_participant_stable():
    first = participant_missing_pattern("1000015", 0.6, 3407)
    second = participant_missing_pattern("1000015", 0.6, 3407)
    assert first == second
    assert first in {"complete", "oct_missing", "cfp_missing"}


def test_normalized_mean_filling_is_zero_after_normalization():
    oct_tensor = torch.randn(3, 3, 8, 8)
    cfp_tensor = torch.randn(3, 3, 8, 8)
    oct_result, cfp_result = apply_missingness(
        oct_tensor, cfp_tensor, ["oct_missing", "cfp_missing", "complete"]
    )
    assert torch.count_nonzero(oct_result[0]) == 0
    assert torch.count_nonzero(cfp_result[1]) == 0
    assert torch.equal(oct_result[2], oct_tensor[2])
    assert torch.equal(cfp_result[2], cfp_tensor[2])


def test_distributed_evaluation_shards_are_exact_and_nonoverlapping():
    shards = [set(DistributedEvalSampler(11, rank, 3)) for rank in range(3)]
    assert set.union(*shards) == set(range(11))
    assert all(
        shards[left].isdisjoint(shards[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )


def test_distributed_shuffle_sampler_is_deterministic_nonoverlapping_and_without_replacement():
    first = DistributedShuffleSampler(11, 3407, 0, 2)
    second = DistributedShuffleSampler(11, 3407, 1, 2)
    repeated = DistributedShuffleSampler(11, 3407, 0, 2)
    first_values, second_values = list(first), list(second)
    assert first_values == list(repeated)
    assert len(set(first_values + second_values)) == 10
    assert set(first_values).isdisjoint(second_values)
    first.set_epoch(1)
    assert list(first) != first_values


def test_class_counts_always_use_the_complete_training_split(tmp_path):
    labels = tmp_path / "reference_labels.csv"
    pd.DataFrame(
        [
            {"split": "train", "label_id": label}
            for label in [0, 0, 1, 2, 3, 4]
        ]
        + [{"split": "validation", "label_id": 4}]
    ).to_csv(labels, index=False)
    assert reference_training_class_counts(labels, 5) == [2, 1, 1, 1, 1]
