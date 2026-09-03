import pandas as pd
import torch
import torch.nn as nn

from look_core.filling import NormalizedMeanFiller, PairedCGANFiller, RawZeroFiller
from look_core.gan import PairedUNetGenerator, gan_internal_split_indices


class IdentityGenerator(nn.Module):
    def forward(self, tensor):
        return tensor


def test_normalized_mean_fill_zeros_only_the_missing_modality():
    oct_tensor = torch.ones(2, 3, 8, 8)
    cfp_tensor = torch.full_like(oct_tensor, 2.0)
    filled_oct, kept_cfp = NormalizedMeanFiller().fill(oct_tensor, cfp_tensor, "oct_missing")
    assert torch.count_nonzero(filled_oct) == 0
    assert torch.equal(kept_cfp, cfp_tensor)


def test_raw_zero_fill_is_distinct_from_normalized_mean():
    tensor = torch.ones(2, 2, 3, 8, 8)
    raw_zero, _ = RawZeroFiller().fill(tensor, tensor, "oct_missing")
    normalized_mean, _ = NormalizedMeanFiller().fill(tensor, tensor, "oct_missing")
    assert torch.count_nonzero(normalized_mean) == 0
    assert not torch.equal(raw_zero, normalized_mean)


def test_paired_cgan_filler_preserves_observed_modality_and_shape():
    filler = PairedCGANFiller(IdentityGenerator(), IdentityGenerator(), torch.device("cpu"))
    oct_tensor = torch.randn(2, 2, 3, 8, 8)
    cfp_tensor = torch.randn(2, 2, 3, 8, 8)
    generated_oct, kept_cfp = filler.fill(oct_tensor, cfp_tensor, "oct_missing")
    assert generated_oct.shape == oct_tensor.shape
    assert torch.equal(kept_cfp, cfp_tensor)


def test_generator_preserves_224_image_shape():
    generator = PairedUNetGenerator(base_channels=4).eval()
    with torch.no_grad():
        output = generator(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 3, 224, 224)


def test_gan_split_is_deterministic_participant_disjoint_and_nonempty():
    frame = pd.DataFrame({"participant_id": ["1", "1", "2", "3", "4"]})
    first = gan_internal_split_indices(frame, 0.1, 3407)
    second = gan_internal_split_indices(frame, 0.1, 3407)
    assert first == second
    train, validation = first
    train_ids = set(frame.iloc[train].participant_id)
    validation_ids = set(frame.iloc[validation].participant_id)
    assert train and validation
    assert train_ids.isdisjoint(validation_ids)
