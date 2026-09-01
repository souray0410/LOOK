import torch

from look_core.graph import build_resnet50_mhd_graph
from look_core.look import LOOKArtifact, apply_artifact, forward_with_look


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
        alpha=1.0,
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
            torch.randn(1, 3, 224, 224),
            torch.randn(1, 3, 224, 224),
        )
    assert logits.shape == (1, 5)
