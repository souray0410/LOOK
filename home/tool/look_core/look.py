from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm

from .filling import MissingModalityFiller, NormalizedMeanFiller


@dataclass
class LOOKArtifact:
    node_name: str
    missing_pattern: str
    filling_strategy: str
    factor: int
    latent_dim: int
    alpha: float
    feature_shape: Tuple[int, ...]
    downsample_shape: Tuple[int, ...]
    mean: torch.Tensor
    std: torch.Tensor
    pca_mean: torch.Tensor
    components: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    ridge_lambda: float
    train_r2: float
    train_mse: float

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: Path) -> "LOOKArtifact":
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


def load_selected_bank(output_dir: Path) -> List[LOOKArtifact]:
    selected_dir = Path(output_dir) / "selected"
    paths = sorted(selected_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No selected LOOK artifacts found in {selected_dir}")
    return [LOOKArtifact.load(path) for path in paths]


class StreamingMoments:
    def __init__(self) -> None:
        self.count = 0
        self.sum: Optional[torch.Tensor] = None
        self.sum_squares: Optional[torch.Tensor] = None

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(dtype=torch.float64, device="cpu")
        batch_sum = values.sum(dim=0)
        batch_squares = values.square().sum(dim=0)
        self.sum = batch_sum if self.sum is None else self.sum + batch_sum
        self.sum_squares = batch_squares if self.sum_squares is None else self.sum_squares + batch_squares
        self.count += values.shape[0]

    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.count < 2 or self.sum is None or self.sum_squares is None:
            raise RuntimeError("At least two samples are required for feature standardization")
        mean = self.sum / self.count
        variance = (self.sum_squares / self.count - mean.square()).clamp_min(1e-12)
        return mean.float(), variance.sqrt().float()


class LatentSufficientStatistics:
    def __init__(self, dimension: int) -> None:
        self.count = 0
        self.sum_x = torch.zeros(dimension, dtype=torch.float64)
        self.sum_y = torch.zeros(dimension, dtype=torch.float64)
        self.xtx = torch.zeros((dimension, dimension), dtype=torch.float64)
        self.xty = torch.zeros((dimension, dimension), dtype=torch.float64)
        self.yty_diagonal = torch.zeros(dimension, dtype=torch.float64)

    def update(self, missing: torch.Tensor, full: torch.Tensor) -> None:
        x = missing.detach().to(dtype=torch.float64, device="cpu")
        y = full.detach().to(dtype=torch.float64, device="cpu") - x
        self.count += x.shape[0]
        self.sum_x += x.sum(0)
        self.sum_y += y.sum(0)
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.yty_diagonal += y.square().sum(dim=0)

    def centered(self, dimension: int):
        n = self.count
        sum_x, sum_y = self.sum_x[:dimension], self.sum_y[:dimension]
        cxx = self.xtx[:dimension, :dimension] - torch.outer(sum_x, sum_x) / n
        cxy = self.xty[:dimension, :dimension] - torch.outer(sum_x, sum_y) / n
        yty = self.yty_diagonal[:dimension].sum() - sum_y.square().sum() / n
        mean_x, mean_y = sum_x / n, sum_y / n
        return cxx, cxy, float(yty), mean_x, mean_y, n


def downsample_flatten(feature: torch.Tensor, factor: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    if feature.ndim == 2:
        return feature, tuple(feature.shape[1:])
    if feature.ndim != 4:
        raise ValueError(f"Expected 2D or 4D feature tensor, got {feature.shape}")
    height = max(1, feature.shape[-2] // factor)
    width = max(1, feature.shape[-1] // factor)
    downsampled = F.adaptive_avg_pool2d(feature, (height, width))
    return downsampled.flatten(1), tuple(downsampled.shape[1:])


def _prepare_inputs(
    batch,
    device: torch.device,
    missing_pattern: str,
    filler: MissingModalityFiller,
):
    oct_tensor = batch["oct"].to(device, non_blocking=True)
    cfp_tensor = batch["cfp"].to(device, non_blocking=True)
    return filler.fill(oct_tensor, cfp_tensor, missing_pattern)


def _reset_inputs(graph, oct_tensor: torch.Tensor, cfp_tensor: torch.Tensor) -> None:
    graph.update_batch_size(oct_tensor.shape[0])
    for node in graph.nodes:
        node.reset()
    graph.get_node_by_name("oct_input").current_state = oct_tensor
    graph.get_node_by_name("cfp_input").current_state = cfp_tensor


def apply_artifact(feature: torch.Tensor, artifact: LOOKArtifact) -> torch.Tensor:
    device = feature.device
    flat, down_shape = downsample_flatten(feature, artifact.factor)
    if tuple(down_shape) != tuple(artifact.downsample_shape):
        raise ValueError(f"Artifact shape {artifact.downsample_shape} != observed {down_shape}")
    mean = artifact.mean.to(device)
    std = artifact.std.to(device)
    pca_mean = artifact.pca_mean.to(device)
    components = artifact.components.to(device)
    weight = artifact.weight.to(device)
    bias = artifact.bias.to(device)
    standardized = (flat - mean) / std
    latent = (standardized - pca_mean) @ components.T
    delta_latent = latent @ weight + bias
    residual_flat = (delta_latent @ components) * std
    residual = residual_flat.reshape(feature.shape[0], *artifact.downsample_shape)
    if feature.ndim == 4 and tuple(residual.shape[-2:]) != tuple(feature.shape[-2:]):
        residual = F.interpolate(residual, size=feature.shape[-2:], mode="trilinear" if feature.ndim == 5 else "bilinear", align_corners=False)
    return feature + artifact.alpha * residual


def forward_with_look(
    graph,
    oct_tensor: torch.Tensor,
    cfp_tensor: torch.Tensor,
    artifacts: Sequence[LOOKArtifact] = (),
    stop_node: Optional[str] = None,
) -> torch.Tensor:
    _reset_inputs(graph, oct_tensor, cfp_tensor)
    by_node = {artifact.node_name: artifact for artifact in artifacts}
    stop_level = graph.node_level_map.get(stop_node, graph.num_levels - 1)
    for level in range(stop_level + 1):
        graph.forward(levels=[level])
        for node_name, artifact in by_node.items():
            if graph.node_level_map[node_name] == level:
                node = graph.get_node_by_name(node_name)
                node.current_state = apply_artifact(node.current_state, artifact)
    target = stop_node or "fusion_logits"
    return graph.get_node_by_name(target).current_state


@torch.no_grad()
def iter_feature_pairs(
    graph,
    loader,
    node_name: str,
    missing_pattern: str,
    factor: int,
    device: torch.device,
    upstream_artifacts: Sequence[LOOKArtifact] = (),
    filler: MissingModalityFiller | None = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], Tuple[int, ...]]]:
    graph.eval()
    filler = filler or NormalizedMeanFiller()
    for batch in loader:
        full_oct, full_cfp = _prepare_inputs(batch, device, "complete", filler)
        full_feature = forward_with_look(graph, full_oct, full_cfp, stop_node=node_name).detach()
        missing_oct, missing_cfp = _prepare_inputs(batch, device, missing_pattern, filler)
        missing_feature = forward_with_look(
            graph, missing_oct, missing_cfp, upstream_artifacts, stop_node=node_name
        ).detach()
        full_flat, down_shape = downsample_flatten(full_feature, factor)
        missing_flat, observed_shape = downsample_flatten(missing_feature, factor)
        if down_shape != observed_shape:
            raise RuntimeError("Full and missing feature shapes differ")
        yield full_flat.cpu(), missing_flat.cpu(), tuple(full_feature.shape[1:]), down_shape


def _fit_incremental_pca(
    pair_factory,
    mean: torch.Tensor,
    std: torch.Tensor,
    max_rank: int,
    fit_batch_size: int = 512,
) -> IncrementalPCA:
    model = IncrementalPCA(n_components=max_rank, batch_size=fit_batch_size)
    buffer: List[np.ndarray] = []
    buffered = 0
    for full, _, _, _ in pair_factory():
        values = ((full - mean) / std).numpy().astype(np.float32, copy=False)
        buffer.append(values)
        buffered += len(values)
        if buffered >= fit_batch_size:
            joined = np.concatenate(buffer, axis=0)
            usable = (len(joined) // fit_batch_size) * fit_batch_size
            model.partial_fit(joined[:usable])
            buffer = [joined[usable:]] if usable < len(joined) else []
            buffered = len(joined) - usable
    if buffer:
        remainder = np.concatenate(buffer, axis=0)
        if len(remainder) >= max_rank:
            model.partial_fit(remainder)
    return model


def _gcv_lambda(cxx: torch.Tensor, cxy: torch.Tensor, tss: float, n: int, dimension: int) -> float:
    eigenvalues, vectors = torch.linalg.eigh(cxx)
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    vectors = vectors.flip(1)
    projected = vectors.T @ cxy
    projected_norm = projected.square().sum(dim=1) / eigenvalues.clamp_min(1e-15)

    def objective(log_lambda: float) -> float:
        ridge = float(np.exp(log_lambda))
        shrinkage = eigenvalues / (eigenvalues + ridge)
        residual = max(0.0, tss - float((shrinkage.square() * projected_norm).sum()))
        scale = 1.0 - float(shrinkage.sum()) / n
        return residual / max(1, n * dimension) / max(scale * scale, 1e-15)

    result = minimize_scalar(objective, bounds=(-13.8, 4.6), method="bounded")
    return float(np.exp(result.x)) if result.success else 0.01


def fit_look_node(
    graph,
    loader,
    node_name: str,
    missing_pattern: str,
    factor: int,
    latent_dims: Sequence[int],
    max_rank: int,
    device: torch.device,
    upstream_artifacts: Sequence[LOOKArtifact] = (),
    filler: MissingModalityFiller | None = None,
) -> Dict[int, LOOKArtifact]:
    def pairs():
        return iter_feature_pairs(
            graph, loader, node_name, missing_pattern, factor, device, upstream_artifacts, filler
        )

    moments = StreamingMoments()
    feature_shape = down_shape = None
    for full, _, feature_shape, down_shape in tqdm(pairs(), desc=f"Stats {node_name} x{factor}"):
        moments.update(full)
    mean, std = moments.finalize()
    feature_dimension = mean.numel()
    rank = min(max_rank, moments.count, feature_dimension)
    pca = _fit_incremental_pca(pairs, mean, std, rank)
    components = torch.from_numpy(pca.components_).float()
    pca_mean = torch.from_numpy(pca.mean_).float()
    statistics = LatentSufficientStatistics(rank)
    for full, missing, _, _ in tqdm(pairs(), desc=f"Latent {node_name} x{factor}"):
        full_z = (((full - mean) / std) - pca_mean) @ components.T
        missing_z = (((missing - mean) / std) - pca_mean) @ components.T
        statistics.update(missing_z, full_z)

    artifacts = {}
    for dimension in sorted(set(min(value, rank) for value in latent_dims if value > 0)):
        cxx, cxy, tss, mean_x, mean_y, n = statistics.centered(dimension)
        ridge_lambda = _gcv_lambda(cxx, cxy, tss, n, dimension)
        weight = torch.linalg.solve(cxx + ridge_lambda * torch.eye(dimension, dtype=cxx.dtype), cxy)
        bias = mean_y - mean_x @ weight
        cross_term = float(torch.trace(weight.T @ cxy))
        fitted_term = float(torch.trace(weight.T @ cxx @ weight))
        residual_sum_squares = max(0.0, tss - 2.0 * cross_term + fitted_term)
        mse = residual_sum_squares / max(1, n * dimension)
        artifacts[dimension] = LOOKArtifact(
            node_name=node_name,
            missing_pattern=missing_pattern,
            filling_strategy=(filler or NormalizedMeanFiller()).name,
            factor=factor,
            latent_dim=dimension,
            alpha=1.0,
            feature_shape=tuple(feature_shape),
            downsample_shape=tuple(down_shape),
            mean=mean,
            std=std,
            pca_mean=pca_mean,
            components=components[:dimension],
            weight=weight.float(),
            bias=bias.float(),
            ridge_lambda=ridge_lambda,
            train_r2=1.0 - residual_sum_squares / tss if tss > 0 else 0.0,
            train_mse=float(mse),
        )
    return artifacts


def greedy_fit_look(
    graph,
    train_loader,
    validation_loader,
    missing_pattern: str,
    correction_nodes: Sequence[str],
    factors: Sequence[int],
    latent_dims: Sequence[int],
    alpha_grid: Sequence[float],
    max_rank: int,
    device: torch.device,
    output_dir: Path,
    filler: MissingModalityFiller | None = None,
    primary_metric: str = "macro_f1",
) -> Tuple[List[LOOKArtifact], List[Dict[str, object]]]:
    """Fit and validation-select LOOK corrections in shallow-to-deep order."""
    from .evaluate import evaluate_missing

    output_dir.mkdir(parents=True, exist_ok=True)
    selected: List[LOOKArtifact] = []
    search_history: List[Dict[str, object]] = []
    for node_name in correction_nodes:
        node_best = None
        node_best_score = -float("inf")
        node_records = []
        node_factors = (1,) if graph.get_node_by_name(node_name).current_state.ndim == 2 else factors
        for factor in node_factors:
            candidates = fit_look_node(
                graph=graph,
                loader=train_loader,
                node_name=node_name,
                missing_pattern=missing_pattern,
                factor=factor,
                latent_dims=latent_dims,
                max_rank=max_rank,
                device=device,
                upstream_artifacts=selected,
                filler=filler,
            )
            for latent_dim, candidate in candidates.items():
                candidate.save(output_dir / "candidates" / f"{node_name}_x{factor}_d{latent_dim}.pt")
                for alpha in alpha_grid:
                    configured = replace(candidate, alpha=float(alpha))
                    result = evaluate_missing(
                        graph,
                        validation_loader,
                        device,
                        fixed_pattern=missing_pattern,
                        artifact_banks={missing_pattern: [*selected, configured]},
                        filler=filler,
                    )
                    score = float(result["metrics"][primary_metric])
                    record = {
                        "node": node_name,
                        "factor": factor,
                        "latent_dim": latent_dim,
                        "alpha": float(alpha),
                        "primary_metric": primary_metric,
                        "primary_score": score,
                    }
                    node_records.append(record)
                    if score > node_best_score:
                        node_best_score = score
                        node_best = configured
        if node_best is None:
            raise RuntimeError(f"No valid LOOK candidate for {node_name}")
        selected.append(node_best)
        node_best.save(output_dir / "selected" / f"{len(selected):02d}_{node_name}.pt")
        search_history.extend(node_records)
        (output_dir / "search_history.json").write_text(
            json.dumps(search_history, indent=2), encoding="utf-8"
        )
    return selected, search_history
