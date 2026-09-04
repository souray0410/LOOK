from __future__ import annotations

import json
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm

from .filling import MissingModalityFiller, NormalizedMeanFiller
from .reproducibility import write_json_atomic
from .state import PipelineState, file_sha256, quarantine, stable_hash


@dataclass
class FullFeaturePCA:
    node_name: str
    factor: int
    feature_shape: Tuple[int, ...]
    downsample_shape: Tuple[int, ...]
    mean: torch.Tensor
    std: torch.Tensor
    pca_mean: torch.Tensor
    components: torch.Tensor
    explained_variance_ratio: torch.Tensor
    sample_count: int
    fit_seconds: float
    peak_rss_bytes: int
    source_id: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.partial')
        torch.save(asdict(self), temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> 'FullFeaturePCA':
        return cls(**torch.load(path, map_location='cpu', weights_only=False))


@dataclass
class LOOKArtifact:
    node_name: str
    missing_pattern: str
    filling_strategy: str
    factor: int
    latent_dim: int
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
    pca_explained_variance: float = 0.0
    pca_fit_seconds: float = 0.0
    pca_peak_rss_bytes: int = 0
    pca_source_id: str = ''

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(asdict(self), temporary)
        temporary.replace(path)

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
    for node in graph.nodes:
        node.reset()
    graph.get_node_by_name("oct_input").feature_message.current_state = oct_tensor
    graph.get_node_by_name("cfp_input").feature_message.current_state = cfp_tensor


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
    return feature + residual


def forward_with_look(
    graph,
    oct_tensor: torch.Tensor,
    cfp_tensor: torch.Tensor,
    artifacts: Sequence[LOOKArtifact] = (),
    stop_node: Optional[str] = None,
) -> torch.Tensor:
    _reset_inputs(graph, oct_tensor, cfp_tensor)
    by_node = {artifact.node_name: artifact for artifact in artifacts}
    target = stop_node or "fusion_logits"
    stop_level = graph.node_level_map[target]
    for level in range(stop_level + 1):
        graph.forward(levels=[level])
        for node_name, artifact in by_node.items():
            if graph.node_level_map[node_name] == level:
                node = graph.get_node_by_name(node_name)
                node.feature_message.current_state = apply_artifact(
                    node.feature_message.current_state, artifact
                )
    return graph.get_node_by_name(target).feature_message.current_state


@torch.no_grad()
def iter_complete_features(graph, loader, node_name, factor, device):
    """Only original complete inputs: no filler, correction or missing forward."""
    graph.eval()
    for batch in loader:
        feature = forward_with_look(
            graph, batch['oct'].to(device), batch['cfp'].to(device), stop_node=node_name
        ).detach()
        flat, shape = downsample_flatten(feature, factor)
        yield flat.cpu(), tuple(feature.shape[1:]), shape


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
    feature_factory,
    mean: torch.Tensor,
    std: torch.Tensor,
    max_rank: int,
    fit_batch_size: int = 512,
) -> IncrementalPCA:
    fit_batch_size = max(fit_batch_size, max_rank)
    model = IncrementalPCA(n_components=max_rank, batch_size=fit_batch_size)
    buffer: List[np.ndarray] = []
    buffered = 0
    for full, _, _ in feature_factory():
        values = ((full - mean) / std).numpy().astype(np.float32, copy=False)
        buffer.append(values)
        buffered += len(values)
        # Keep a rank-sized tail so the final partial_fit uses every sample.
        if buffered >= fit_batch_size + max_rank:
            joined = np.concatenate(buffer, axis=0)
            usable = ((len(joined) - max_rank) // fit_batch_size) * fit_batch_size
            model.partial_fit(joined[:usable])
            buffer = [joined[usable:]] if usable < len(joined) else []
            buffered = len(joined) - usable
    if buffer:
        remainder = np.concatenate(buffer, axis=0)
        model.partial_fit(remainder)
    return model


def _process_peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def fit_complete_pca(graph, loader, node_name, factor, max_rank, device, source_id):
    def features():
        return iter_complete_features(graph, loader, node_name, factor, device)

    moments = StreamingMoments()
    feature_shape = down_shape = None
    for full, feature_shape, down_shape in tqdm(features(), desc=f'PCA stats {node_name} x{factor}'):
        moments.update(full)
    mean, std = moments.finalize()
    rank = min(max_rank, moments.count - 1, mean.numel())
    started = time.perf_counter()
    pca = _fit_incremental_pca(features, mean, std, rank)
    if pca.n_samples_seen_ != moments.count:
        raise RuntimeError('PCA must consume every complete training feature')
    return FullFeaturePCA(
        node_name=node_name, factor=factor, feature_shape=feature_shape,
        downsample_shape=down_shape, mean=mean, std=std,
        pca_mean=torch.from_numpy(pca.mean_).float(),
        components=torch.from_numpy(pca.components_).float(),
        explained_variance_ratio=torch.from_numpy(pca.explained_variance_ratio_).float(),
        sample_count=moments.count, fit_seconds=time.perf_counter() - started,
        peak_rss_bytes=_process_peak_rss_bytes(), source_id=source_id,
    )


def prepare_complete_pca_bank(
    graph, loader, correction_nodes, factors, max_rank, device,
    pca_root: Path, identity: Mapping[str, object], quarantine_root: Path,
    *, load_only: bool = False,
) -> Dict[Tuple[str, int], FullFeaturePCA]:
    """Build/verify a complete-training-only bank shared by all missing scenarios."""
    if getattr(loader.dataset, 'split', 'train') != 'train' or getattr(loader.dataset, 'augment', False):
        raise ValueError('Shared PCA requires the unaugmented training split')
    if getattr(loader, 'drop_last', False):
        raise ValueError('Shared PCA cannot drop training samples')
    identity = dict(identity, max_rank=max_rank, pca_code_sha256=file_sha256(Path(__file__)))
    bank_id = stable_hash(identity)[:16]
    output = Path(pca_root) / bank_id
    graph.eval()
    first = next(iter(loader))
    with torch.no_grad():
        forward_with_look(graph, first['oct'].to(device), first['cfp'].to(device))
    entries = []
    for node in correction_nodes:
        shape = graph.get_node_by_name(node).feature_message.current_state.shape
        node_factors = [1] if len(shape) == 2 else list(dict.fromkeys(factors))
        entries.extend((node, int(factor)) for factor in node_factors)
    state_config = dict(identity, requested_entries=entries)
    bank = {}
    output.mkdir(parents=True, exist_ok=True)
    with PipelineState(output / 'state', 'prepare', state_config, []) as state:
        for index, (node, factor) in enumerate(entries):
            path = output / f'{node}_x{factor}.pt'
            metadata_path = path.with_suffix('.json')
            source_id = f'{bank_id}/{node}_x{factor}'
            progress = dict(status='running', bank_id=bank_id, node=node, factor=factor,
                            completed_entries=index, total_entries=len(entries))
            write_json_atomic(progress, output / 'progress.json')
            basis = None
            if path.exists() or metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text())
                    if metadata['source_id'] != source_id or metadata['sha256'] != file_sha256(path):
                        raise ValueError('PCA source identity or hash mismatch')
                    basis = FullFeaturePCA.load(path)
                    if basis.source_id != source_id or basis.node_name != node or basis.factor != factor:
                        raise ValueError('PCA metadata mismatch')
                    if not all(torch.isfinite(t).all() for t in
                               (basis.mean, basis.std, basis.pca_mean, basis.components, basis.explained_variance_ratio)):
                        raise ValueError('Non-finite shared PCA')
                except (OSError, ValueError, KeyError, RuntimeError, EOFError, TypeError):
                    if load_only:
                        raise RuntimeError(f'Frozen PCA entry is invalid: {path}')
                    for invalid in (path, metadata_path):
                        if invalid.exists():
                            quarantine(invalid, quarantine_root, 'invalid shared PCA entry')
                    basis = None
            if basis is None:
                if load_only:
                    raise FileNotFoundError(path)
                print(f'FIT shared PCA {source_id} (complete train only)', flush=True)
                basis = fit_complete_pca(graph, loader, node, factor, max_rank, device, source_id)
                basis.save(path)
                write_json_atomic(dict(source_id=source_id, sha256=file_sha256(path),
                    bytes=path.stat().st_size, samples=basis.sample_count), metadata_path)
            else:
                print(f'REUSE shared PCA {source_id}', flush=True)
            bank[(node, factor)] = basis
            state.checkpoint(completed_entries=index + 1, last_source_id=source_id)
        manifest = dict(bank_id=bank_id, identity=identity, entries=[
            dict(node=node, factor=factor, source_id=bank[(node, factor)].source_id,
                 path=str(output / f'{node}_x{factor}.pt'),
                 sha256=file_sha256(output / f'{node}_x{factor}.pt'))
            for node, factor in entries])
        write_json_atomic(manifest, output / 'bank_manifest.json')
        write_json_atomic(dict(status='complete', bank_id=bank_id,
            completed_entries=len(entries), total_entries=len(entries)), output / 'progress.json')
        state.complete([output / 'bank_manifest.json', *(output / f'{node}_x{factor}.pt' for node, factor in entries)])
    return bank


def _gcv_lambda(cxx: torch.Tensor, cxy: torch.Tensor, tss: float, n: int, dimension: int) -> float:
    eigenvalues, vectors = torch.linalg.eigh(cxx)
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    vectors = vectors.flip(1)
    projected = vectors.T @ cxy
    projected_norm = projected.square().sum(dim=1) / eigenvalues.clamp_min(1e-15)

    def objective(log_lambda: float) -> float:
        ridge = float(np.exp(log_lambda))
        shrinkage = eigenvalues / (eigenvalues + ridge)
        # ||Y-XW||^2 = TSS - sum((2*s-s^2)*||u^T Y||^2).
        residual = max(0.0, tss - float(((2 * shrinkage - shrinkage.square()) * projected_norm).sum()))
        scale = 1.0 - (1.0 + float(shrinkage.sum())) / n
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
    pca: FullFeaturePCA,
    upstream_artifacts: Sequence[LOOKArtifact] = (),
    filler: MissingModalityFiller | None = None,
) -> Dict[int, LOOKArtifact]:
    def pairs():
        return iter_feature_pairs(
            graph, loader, node_name, missing_pattern, factor, device, upstream_artifacts, filler
        )

    if pca.node_name != node_name or pca.factor != factor:
        raise ValueError('LOOK node must use its matching shared complete PCA')
    mean, std, components, pca_mean = pca.mean, pca.std, pca.components, pca.pca_mean
    feature_shape, down_shape = pca.feature_shape, pca.downsample_shape
    rank = min(max_rank, len(components))
    components = components[:rank]
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
            pca_explained_variance=float(
                pca.explained_variance_ratio[:dimension].sum()
            ),
            pca_fit_seconds=pca.fit_seconds,
            pca_peak_rss_bytes=pca.peak_rss_bytes,
            pca_source_id=pca.source_id,
        )
    return artifacts


def validate_global_factor_bank(
    artifacts: Sequence[LOOKArtifact],
    correction_nodes: Sequence[str],
    factor: int,
    *,
    allow_prefix: bool = False,
) -> None:
    if not allow_prefix and len(artifacts) != len(correction_nodes):
        raise ValueError(
            f"LOOK bank x{factor} has {len(artifacts)} artifacts; "
            f"expected {len(correction_nodes)}"
        )
    if len(artifacts) > len(correction_nodes):
        raise ValueError(f"LOOK bank x{factor} contains too many artifacts")
    for artifact, expected_node in zip(artifacts, correction_nodes):
        if artifact.node_name != expected_node:
            raise ValueError(
                f"LOOK bank x{factor} expected node {expected_node!r}, "
                f"found {artifact.node_name!r}"
            )
        expected_factor = 1 if len(artifact.feature_shape) == 1 else factor
        if artifact.factor != expected_factor:
            raise ValueError(
                f"LOOK bank x{factor} node {artifact.node_name!r} uses "
                f"factor {artifact.factor}; expected {expected_factor}"
            )


def _read_search_history(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Invalid LOOK search history: {path}")
    return payload


def _fit_factor_bank(
    graph,
    train_loader,
    validation_loader,
    missing_pattern: str,
    correction_nodes: Sequence[str],
    factor: int,
    latent_dims: Sequence[int],
    max_rank: int,
    device: torch.device,
    output_dir: Path,
    pca_bank: Mapping[Tuple[str, int], FullFeaturePCA],
    filler: MissingModalityFiller | None = None,
    primary_metric: str = "macro_f1",
    resume: bool = True,
) -> Tuple[List[LOOKArtifact], List[Dict[str, object]], Dict[str, object]]:
    """Fit one node-wise LOOK bank with a single shared spatial factor."""
    from .evaluate import evaluate_missing
    from .matrix_analysis import analyze_look_bank

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = output_dir / "selected"
    completion_path = output_dir / "bank_complete.json"
    history_path = output_dir / "search_history.json"
    if resume and completion_path.is_file():
        selected = load_selected_bank(output_dir)
        validate_global_factor_bank(selected, correction_nodes, factor)
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        return selected, _read_search_history(history_path), completion

    selected = load_selected_bank(output_dir) if resume and selected_dir.exists() else []
    validate_global_factor_bank(
        selected, correction_nodes, factor, allow_prefix=True
    )
    search_history = _read_search_history(history_path) if resume else []
    for node_name in correction_nodes[len(selected):]:
        node_best = None
        node_best_score = -float("inf")
        node_records = []
        node_state = graph.get_node_by_name(node_name).feature_message.current_state
        node_factor = 1 if node_state.ndim == 2 else factor
        candidates = fit_look_node(
            graph=graph,
            loader=train_loader,
            node_name=node_name,
            missing_pattern=missing_pattern,
            factor=node_factor,
            latent_dims=latent_dims,
            max_rank=max_rank,
            device=device,
            pca=pca_bank[(node_name, node_factor)],
            upstream_artifacts=selected,
            filler=filler,
        )
        for latent_dim, candidate in candidates.items():
            candidate.save(
                output_dir / "candidates" /
                f"{node_name}_x{node_factor}_d{latent_dim}.pt"
            )
            result = evaluate_missing(
                graph,
                validation_loader,
                device,
                fixed_pattern=missing_pattern,
                artifact_banks={missing_pattern: [*selected, candidate]},
                filler=filler,
            )
            score = float(result["metrics"][primary_metric])
            record = {
                "bank_factor": factor,
                "node": node_name,
                "factor": node_factor,
                "latent_dim": latent_dim,
                "ridge_lambda": candidate.ridge_lambda,
                "primary_metric": primary_metric,
                "primary_score": score,
            }
            node_records.append(record)
            if score > node_best_score:
                node_best_score = score
                node_best = candidate
        if node_best is None:
            raise RuntimeError(f"No valid LOOK candidate for {node_name}")
        selected.append(node_best)
        node_best.save(output_dir / "selected" / f"{len(selected):02d}_{node_name}.pt")
        search_history.extend(node_records)
        write_json_atomic(search_history, history_path)

    validate_global_factor_bank(selected, correction_nodes, factor)
    validation = evaluate_missing(
        graph,
        validation_loader,
        device,
        fixed_pattern=missing_pattern,
        artifact_banks={missing_pattern: selected},
        filler=filler,
    )["metrics"]
    matrix_records = analyze_look_bank(selected, output_dir)
    selected_paths = sorted(selected_dir.glob("*.pt"))
    completion = {
        "status": "complete",
        "missing_pattern": missing_pattern,
        "factor": factor,
        "primary_metric": primary_metric,
        "primary_score": float(validation[primary_metric]),
        "validation_metrics": validation,
        "artifacts": len(selected),
        "artifact_bytes": sum(path.stat().st_size for path in selected_paths),
        "compression": [
            {
                "node": artifact.node_name,
                "factor": artifact.factor,
                "original_dimension": int(np.prod(artifact.feature_shape)),
                "compressed_dimension": int(np.prod(artifact.downsample_shape)),
                "compression_ratio": float(
                    np.prod(artifact.downsample_shape) / np.prod(artifact.feature_shape)
                ),
                "pca_explained_variance": artifact.pca_explained_variance,
                "pca_fit_seconds": artifact.pca_fit_seconds,
                "pca_peak_rss_bytes": artifact.pca_peak_rss_bytes,
            }
            for artifact in selected
        ],
        "matrix_diagnostics": matrix_records,
    }
    write_json_atomic(completion, completion_path)
    return selected, search_history, completion


def greedy_fit_look(
    graph,
    train_loader,
    validation_loader,
    missing_pattern: str,
    correction_nodes: Sequence[str],
    factors: Sequence[int],
    latent_dims: Sequence[int],
    max_rank: int,
    device: torch.device,
    output_dir: Path,
    pca_bank: Mapping[Tuple[str, int], FullFeaturePCA],
    filler: MissingModalityFiller | None = None,
    primary_metric: str = "macro_f1",
    resume: bool = True,
) -> Tuple[List[LOOKArtifact], List[Dict[str, object]]]:
    """Select one shared spatial factor for the complete LOOK artifact bank."""
    unique_factors = list(dict.fromkeys(int(value) for value in factors))
    if not unique_factors or any(value < 1 for value in unique_factors):
        raise ValueError("LOOK factors must be a non-empty sequence of positive integers")
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_records: List[Dict[str, object]] = []
    all_history: List[Dict[str, object]] = []
    best_artifacts: List[LOOKArtifact] | None = None
    best_score = -float("inf")
    best_factor = -1
    for factor in unique_factors:
        factor_dir = output_dir / "factors" / f"x{factor}"
        artifacts, history, completion = _fit_factor_bank(
            graph,
            train_loader,
            validation_loader,
            missing_pattern,
            correction_nodes,
            factor,
            latent_dims,
            max_rank,
            device,
            factor_dir,
            pca_bank=pca_bank,
            filler=filler,
            primary_metric=primary_metric,
            resume=resume,
        )
        score = float(completion["primary_score"])
        factor_records.append(completion)
        all_history.extend(history)
        if (score, factor) > (best_score, best_factor):
            best_score, best_factor = score, factor
            best_artifacts = artifacts

    if best_artifacts is None:
        raise RuntimeError("No complete LOOK factor bank was produced")
    selected_dir = output_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for stale in selected_dir.glob("*.pt"):
        stale.unlink()
    for index, artifact in enumerate(best_artifacts, start=1):
        artifact.save(selected_dir / f"{index:02d}_{artifact.node_name}.pt")
    selection = {
        "status": "selected",
        "selection_scope": "one_global_spatial_factor_per_artifact_bank",
        "vector_node_factor": 1,
        "primary_metric": primary_metric,
        "tie_break": "larger_factor_for_lower_spatial_cost",
        "selected_factor": best_factor,
        "selected_primary_score": best_score,
        "factor_banks": factor_records,
    }
    write_json_atomic(all_history, output_dir / "search_history.json")
    write_json_atomic(selection, output_dir / "factor_selection.json")
    return best_artifacts, all_history
