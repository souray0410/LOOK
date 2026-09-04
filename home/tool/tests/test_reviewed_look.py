from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
import torch

from look_core.look import _fit_incremental_pca, _gcv_lambda


@pytest.mark.parametrize('n', [33, 64, 65, 129, 140])
def test_pca_uses_entire_training_tail(n):
    values = torch.randn(n, 12, generator=torch.Generator().manual_seed(9))
    def pairs():
        for row in values.split(7):
            yield row, row, (12,), (12,)
    model = _fit_incremental_pca(pairs, torch.zeros(12), torch.ones(12), 8, 32)
    assert model.n_samples_seen_ == n
    np.testing.assert_allclose(model.mean_, values.double().mean(0).numpy(), atol=1e-6)


def test_gcv_matches_explicit_ridge_residual_and_hat_trace():
    generator = torch.Generator().manual_seed(12)
    x = torch.randn(73, 9, dtype=torch.float64, generator=generator)
    y = x @ torch.randn(9, 9, dtype=torch.float64, generator=generator) + 2 * torch.randn(73, 9, dtype=torch.float64, generator=generator)
    x -= x.mean(0)
    y -= y.mean(0)
    cxx, cxy = x.T @ x, x.T @ y
    def direct(log_lambda):
        regularized = cxx + np.exp(log_lambda) * torch.eye(9, dtype=torch.float64)
        weight = torch.linalg.solve(regularized, cxy)
        residual = float((y - x @ weight).square().sum())
        df = 1 + float(torch.trace(torch.linalg.solve(regularized, cxx)))
        return residual / (73 * 9) / (1 - df / 73) ** 2
    expected = np.exp(minimize_scalar(direct, bounds=(-13.8, 4.6), method='bounded').x)
    actual = _gcv_lambda(cxx, cxy, float(y.square().sum()), 73, 9)
    assert actual == pytest.approx(expected, rel=1e-4)


def test_reviewed_queue_is_one_configuration_three_seeds_nine_cases():
    api = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'pipeline/34_run_reviewed_look_study.py'))
    from look_core.study_grid import calibration_classifier_profiles
    candidate = {'winner': {'fusion_position': 'layer3'}, 'classifier_profile': calibration_classifier_profiles()[4]}
    stages = api['study_stages'](candidate, [4, 8, 16], [8, 16, 32, 64, 128, 256])
    assert sum(len(g.seeds) * len(g.filling_strategies) for _, g in stages) == 9
    assert stages[0][1].seeds == [3407]
    assert stages[0][1].filling_strategies == ['raw_zero', 'normalized_mean']
    assert stages[-1][1].filling_strategies == ['paired_cgan']
    for _, grid in stages:
        grid.validate()
        assert grid.fusion_positions == ['layer3']
        assert grid.classifier_profiles == [candidate['classifier_profile']]
        look = grid.look_profiles[0]
        assert not any('alpha' in k for k in look)
        assert look['downsample_factors'] == [4, 8, 16]
        assert look['missing_patterns'] == ['oct_missing', 'cfp_missing']
