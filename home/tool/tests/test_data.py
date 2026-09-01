import torch

from look_core.data import apply_missingness, participant_missing_pattern


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
