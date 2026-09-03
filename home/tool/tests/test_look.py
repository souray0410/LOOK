import json

import torch
import pytest

import look_core.look as look_module
from look_core.graph import build_resnet50_mhd_graph
from look_core.look import (
    LOOKArtifact,
    apply_artifact,
    downsample_flatten,
    forward_with_look,
    validate_global_factor_bank,
)


def test_zero_linear_correction_is_identity():
    feature = torch.randn(2, 8, 7, 7)
    dimension = 8 * 2 * 2
    latent = 4
    artifact = LOOKArtifact(
        node_name="fusion_layer4",
        missing_pattern="oct_missing",
        filling_strategy="normalized_mean",
        factor=3,
        latent_dim=latent,
        feature_shape=(8, 7, 7),
        downsample_shape=(8, 2, 2),
        mean=torch.zeros(dimension),
        std=torch.ones(dimension),
        pca_mean=torch.zeros(dimension),
        components=torch.randn(latent, dimension),
        weight=torch.zeros(latent, latent),
        bias=torch.zeros(latent),
        ridge_lambda=0.01,
        train_r2=0.0,
        train_mse=0.0,
    )
    assert torch.equal(apply_artifact(feature, artifact), feature)


def test_inference_stops_at_logits_before_label_monitor_level():
    graph = build_resnet50_mhd_graph(
        "feature", batch_size=2, pretrained=False, device="cpu"
    )
    with torch.no_grad():
        logits = forward_with_look(
            graph,
            torch.randn(1, 2, 3, 224, 224),
            torch.randn(1, 2, 3, 224, 224),
        )
    assert logits.shape == (1, 2)


def test_vector_features_are_identity_compressed():
    feature = torch.randn(3, 2048)
    flattened, shape = downsample_flatten(feature, factor=16)
    assert torch.equal(flattened, feature)
    assert shape == (2048,)


def test_predicted_residual_is_lifted_to_the_native_spatial_shape():
    feature = torch.zeros(1, 1, 4, 4)
    artifact = LOOKArtifact(
        node_name="fusion_layer4",
        missing_pattern="oct_missing",
        filling_strategy="normalized_mean",
        factor=2,
        latent_dim=4,
        feature_shape=(1, 4, 4),
        downsample_shape=(1, 2, 2),
        mean=torch.zeros(4),
        std=torch.ones(4),
        pca_mean=torch.zeros(4),
        components=torch.eye(4),
        weight=torch.zeros(4, 4),
        bias=torch.ones(4),
        ridge_lambda=0.01,
        train_r2=0.0,
        train_mse=0.0,
    )
    corrected = apply_artifact(feature, artifact)
    assert corrected.shape == feature.shape
    assert torch.equal(corrected, torch.ones_like(feature))


def test_global_factor_bank_allows_identity_only_for_vector_nodes():
    spatial = LOOKArtifact(
        node_name="fusion_layer4",
        missing_pattern="oct_missing",
        filling_strategy="normalized_mean",
        factor=8,
        latent_dim=1,
        feature_shape=(2, 7, 7),
        downsample_shape=(2, 1, 1),
        mean=torch.zeros(2),
        std=torch.ones(2),
        pca_mean=torch.zeros(2),
        components=torch.ones(1, 2),
        weight=torch.zeros(1, 1),
        bias=torch.zeros(1),
        ridge_lambda=0.01,
        train_r2=0.0,
        train_mse=0.0,
    )
    vector = LOOKArtifact(
        node_name="fusion_feature",
        missing_pattern="oct_missing",
        filling_strategy="normalized_mean",
        factor=1,
        latent_dim=1,
        feature_shape=(2,),
        downsample_shape=(2,),
        mean=torch.zeros(2),
        std=torch.ones(2),
        pca_mean=torch.zeros(2),
        components=torch.ones(1, 2),
        weight=torch.zeros(1, 1),
        bias=torch.zeros(1),
        ridge_lambda=0.01,
        train_r2=0.0,
        train_mse=0.0,
    )
    validate_global_factor_bank(
        [spatial, vector], ["fusion_layer4", "fusion_feature"], 8
    )
    with pytest.raises(ValueError, match="uses factor"):
        validate_global_factor_bank(
            [replace_artifact_factor(spatial, 4), vector],
            ["fusion_layer4", "fusion_feature"],
            8,
        )


def replace_artifact_factor(artifact: LOOKArtifact, factor: int) -> LOOKArtifact:
    from dataclasses import replace

    return replace(artifact, factor=factor)


def test_global_factor_selection_writes_one_selected_bank(monkeypatch, tmp_path):
    calls = []

    def fake_fit_factor_bank(
        graph,
        train_loader,
        validation_loader,
        missing_pattern,
        correction_nodes,
        factor,
        latent_dims,
        max_rank,
        device,
        output_dir,
        **kwargs,
    ):
        calls.append(factor)
        artifact = LOOKArtifact(
            node_name="fusion_layer4",
            missing_pattern=missing_pattern,
            filling_strategy="normalized_mean",
            factor=factor,
            latent_dim=1,
            feature_shape=(1, 7, 7),
            downsample_shape=(1, 1, 1),
            mean=torch.zeros(1),
            std=torch.ones(1),
            pca_mean=torch.zeros(1),
            components=torch.ones(1, 1),
            weight=torch.zeros(1, 1),
            bias=torch.zeros(1),
            ridge_lambda=0.01,
            train_r2=0.0,
            train_mse=0.0,
        )
        score = {4: 0.4, 8: 0.8, 16: 0.6}[factor]
        return [artifact], [{"bank_factor": factor}], {
            "factor": factor,
            "primary_score": score,
        }

    monkeypatch.setattr(look_module, "_fit_factor_bank", fake_fit_factor_bank)
    artifacts, _ = look_module.greedy_fit_look(
        None,
        None,
        None,
        "oct_missing",
        ["fusion_layer4"],
        [4, 8, 16],
        [1],
        1,
        torch.device("cpu"),
        tmp_path,
    )
    selection = json.loads((tmp_path / "factor_selection.json").read_text())
    assert calls == [4, 8, 16]
    assert selection["selected_factor"] == 8
    assert artifacts[0].factor == 8
    assert look_module.load_selected_bank(tmp_path)[0].factor == 8
