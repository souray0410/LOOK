import json

import numpy as np
import torch

from look_core.look import LOOKArtifact
from look_core.matrix_analysis import (
    aggregate_matrix_analyses,
    analyze_correction_matrix,
    analyze_look_bank,
)


def _artifact():
    return LOOKArtifact(
        node_name="fusion_feature",
        missing_pattern="oct_missing",
        filling_strategy="normalized_mean",
        factor=1,
        latent_dim=3,
        feature_shape=(3,),
        downsample_shape=(3,),
        mean=torch.zeros(3),
        std=torch.ones(3),
        pca_mean=torch.zeros(3),
        components=torch.eye(3),
        weight=torch.diag(torch.tensor([3.0, 1.0, 0.0])),
        bias=torch.tensor([1.0, 0.0, 0.0]),
        ridge_lambda=0.1,
        train_r2=0.5,
        train_mse=0.2,
    )


def test_matrix_diagnostics_have_expected_rank_and_applied_map():
    result = analyze_correction_matrix(_artifact())
    assert result["original_dimension"] == 3
    assert result["compressed_dimension"] == 3
    assert result["compression_ratio"] == 1.0
    assert result["correction_weight"]["numerical_rank"] == 2
    assert np.isclose(result["correction_weight"]["spectral_norm"], 3.0)
    assert np.isclose(result["correction_weight"]["stable_rank"], 10.0 / 9.0)
    assert np.isclose(result["applied_latent_map"]["spectral_norm"], 4.0)
    assert result["complex_eigenvalues"] == 0


def test_matrix_bank_writes_json_csv_and_eigenvectors(tmp_path):
    records = analyze_look_bank([_artifact()], tmp_path)
    assert len(records) == 1
    assert (tmp_path / "matrix_analysis.json").is_file()
    assert (tmp_path / "matrix_analysis.csv").is_file()
    assert (tmp_path / "matrix_summary.png").is_file()
    assert (tmp_path / "matrix_summary.pdf").is_file()
    spectra = np.load(tmp_path / "matrix_spectra" / "01_fusion_feature.npz")
    assert spectra["right_eigenvectors"].shape == (3, 3)
    assert spectra["left_eigenvectors"].shape == (3, 3)
    assert (tmp_path / "matrix_spectra" / "01_fusion_feature.png").is_file()
    payload = json.loads((tmp_path / "matrix_analysis.json").read_text())
    assert "non-normal" in payload["interpretation_note"]


def test_recursive_cross_experiment_aggregation(tmp_path):
    bank = tmp_path / "pipeline_smoke" / "experiments" / "exp1" / "look" / "oct_missing"
    analyze_look_bank([_artifact()], bank)
    result_path = bank.parents[1] / "experiment_result.json"
    result_path.write_text(
        json.dumps(
            {
                "experiment_id": "exp1",
                "selection": {
                    "fusion_position": "feature",
                    "seed": 3407,
                    "filling_strategy": "normalized_mean",
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "aggregate.csv"
    assert aggregate_matrix_analyses(tmp_path, output) == 1
    assert "fusion_feature" in output.read_text(encoding="utf-8")
