import torch

from look_core.data import (
    DistributedEvalSampler,
    DistributedWeightedSampler,
    apply_missingness,
    participant_missing_pattern,
)


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


def test_distributed_weighted_sampler_is_deterministic_per_epoch():
    labels = [0, 0, 1, 1, 2, 2, 3, 3]
    first = DistributedWeightedSampler(labels, 0.5, 3407, 0, 2)
    second = DistributedWeightedSampler(labels, 0.5, 3407, 0, 2)
    assert list(first) == list(second)
    first.set_epoch(1)
    assert list(first) != list(second)
