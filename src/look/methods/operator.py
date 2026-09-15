from __future__ import annotations

from look.runtime.state import durable_replace

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

from look.methods.imputation import MissingModalityFiller, NormalizedMeanFiller
from look.runtime.provenance import write_json_atomic
from look.runtime.state import PipelineState, file_sha256, quarantine, stable_hash
from look.methods.joint import PROTOCOL, members, member_shapes, read_site, write_site, site_level


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
    member_names: tuple = ()
    member_shapes: tuple = ()
    protocol: str = PROTOCOL
    split_rule: str = "channel_split_and_restore_member_shapes_v1"
    spatial_method: str = "interpolate"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.partial')
        torch.save(asdict(self), temporary)
        durable_replace(temporary, path)

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
    member_names: tuple = ()
    member_shapes: tuple = ()
    protocol: str = PROTOCOL
    split_rule: str = "channel_split_and_restore_member_shapes_v1"
    spatial_method: str = "interpolate"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(asdict(self), temporary)
        durable_replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> "LOOKArtifact":
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


def load_selected_bank(output_dir: Path) -> List[LOOKArtifact]:
    root = Path(output_dir)
    manifest = json.loads((root / 'selected_manifest.json').read_text())
    if manifest['protocol'] != PROTOCOL:
        raise ValueError('LOOK protocol mismatch')
    result = []
    for record in manifest['artifacts']:
        path = root / record['path']
        if path.parent != root / 'selected' or path.is_symlink() or file_sha256(path) != record['sha256']:
            raise ValueError('Invalid selected artifact')
        result.append(LOOKArtifact.load(path))
    return result


def save_selected_bank(artifacts, output_dir):
    root = Path(output_dir)
    (root / 'selected').mkdir(parents=True, exist_ok=True)
    records = []
    for i, artifact in enumerate(artifacts):
        path = root / 'selected' / f'{i + 1:02d}_{artifact.node_name}.pt'
        artifact.save(path)
        records.append(dict(path=str(path.relative_to(root)), sha256=file_sha256(path)))
    write_json_atomic(dict(protocol=PROTOCOL, artifacts=records), root / 'selected_manifest.json')
    return records


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


def downsample_flatten(feature: torch.Tensor, factor: int, spatial_method="interpolate") -> Tuple[torch.Tensor, Tuple[int, ...]]:
    from look.methods.spatial import reduce_spatial
    reduced = reduce_spatial(feature, factor, spatial_method)
    return reduced.flatten(1), tuple(reduced.shape[1:])


def _prepare_inputs(
    batch,
    device: torch.device,
    missing_pattern: str,
    filler: MissingModalityFiller,
):
    oct_tensor = batch["oct"].to(device, non_blocking=True)
    cfp_tensor = batch["cfp"].to(device, non_blocking=True)
    return filler.fill(oct_tensor, cfp_tensor, missing_pattern)


def _reset_inputs(graph, oct_tensor: torch.Tensor, cfp_tensor: torch.Tensor, counts=None) -> None:
    for node in graph.nodes:
        node.reset()
    from look.models.native_host import set_observed_counts
    set_observed_counts(graph, counts, len(oct_tensor))
    for name, tensor in (("oct_input", oct_tensor), ("cfp_input", cfp_tensor)):
        node = graph.get_node_by_name(name)
        if node is not None:
            node.feature_message.current_state = tensor


def apply_artifact(feature: torch.Tensor, artifact: LOOKArtifact) -> torch.Tensor:
    # Versioned supplementary artifacts implement their own fixed writeback;
    # ordinary LOOKArtifact and all historical checkpoint fields are unchanged.
    if hasattr(artifact, 'apply_feature'):
        return artifact.apply_feature(feature)
    device = feature.device
    flat, down_shape = downsample_flatten(feature, artifact.factor, artifact.spatial_method)
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
    counts=None,
) -> torch.Tensor:
    _reset_inputs(graph, oct_tensor, cfp_tensor, counts)
    by_node = {artifact.node_name: artifact for artifact in artifacts}
    if len(by_node) != len(artifacts):
        raise ValueError('Duplicate LOOK sites')
    target = stop_node or "fusion_logits"
    stop_level = site_level(graph, target)
    for level in range(-1, stop_level + 1):
        if level >= 0:
            graph.forward(levels=[level])
        for node_name, artifact in by_node.items():
            if site_level(graph, node_name) == level:
                if artifact.protocol != PROTOCOL or artifact.split_rule != "channel_split_and_restore_member_shapes_v1":
                    raise ValueError("LOOK protocol or split rule mismatch")
                if artifact.member_names and tuple(artifact.member_names) != members(node_name):
                    raise ValueError('LOOK member order mismatch')
                if artifact.member_shapes and tuple(map(tuple, artifact.member_shapes)) != member_shapes(graph, node_name):
                    raise ValueError('LOOK member shape mismatch')
                write_site(graph, node_name, apply_artifact(read_site(graph, node_name), artifact))
    return read_site(graph, target)


@torch.no_grad()
def iter_complete_features(graph, loader, node_name, factor, device, spatial_method="interpolate"):
    """Only original complete inputs: no filler, correction or missing forward."""
    graph.eval()
    for batch in loader:
        feature = forward_with_look(
            graph, batch['oct'].to(device), batch['cfp'].to(device), stop_node=node_name, **({'counts': batch['counts']} if 'counts' in batch else {})
        ).detach()
        flat, shape = downsample_flatten(feature, factor, spatial_method)
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
    spatial_method: str = "interpolate",
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], Tuple[int, ...]]]:
    graph.eval()
    filler = filler or NormalizedMeanFiller()
    for batch in loader:
        full_oct, full_cfp = _prepare_inputs(batch, device, "complete", filler)
        full_feature = forward_with_look(graph, full_oct, full_cfp, stop_node=node_name, **({'counts': batch['counts']} if 'counts' in batch else {})).detach()
        missing_oct, missing_cfp = _prepare_inputs(batch, device, missing_pattern, filler)
        missing_feature = forward_with_look(
            graph, missing_oct, missing_cfp, upstream_artifacts, stop_node=node_name, **({'counts': batch['counts']} if 'counts' in batch else {})
        ).detach()
        full_flat, down_shape = downsample_flatten(full_feature, factor, spatial_method)
        missing_flat, observed_shape = downsample_flatten(missing_feature, factor, spatial_method)
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


def fit_complete_pca(graph, loader, node_name, factor, max_rank, device, source_id, spatial_method="interpolate", strict_rank=False, *, feature_factory=None):
    def features():
        return feature_factory() if feature_factory is not None else iter_complete_features(graph, loader, node_name, factor, device, **({"spatial_method": spatial_method} if spatial_method != "interpolate" else {}))

    moments = StreamingMoments()
    feature_shape = down_shape = None
    samples = 0
    for full, feature_shape, down_shape in tqdm(features(), desc=f'PCA stats {node_name} x{factor}'):
        samples += len(full)
        if len(down_shape) == 3:
            values = full.reshape(-1, *down_shape).permute(0, 2, 3, 1).reshape(-1, down_shape[0])
        else:
            values = full
        moments.update(values)
    mean, std = moments.finalize()
    if len(down_shape) == 3:
        # Persist the channel statistics broadcast to flattened PCA coordinates.
        spatial = int(np.prod(down_shape[1:]))
        mean, std = mean.repeat_interleave(spatial), std.repeat_interleave(spatial)
    rank = min(max_rank, samples - 1, mean.numel())
    if strict_rank and rank != max_rank:
        raise ValueError("Requested PCA rank is infeasible; no implicit rank reduction")
    started = time.perf_counter()
    pca = _fit_incremental_pca(features, mean, std, rank)
    if pca.n_samples_seen_ != samples:
        raise RuntimeError('PCA must consume every complete training feature')
    return FullFeaturePCA(
        node_name=node_name, factor=factor, feature_shape=feature_shape,
        downsample_shape=down_shape, mean=mean, std=std,
        pca_mean=torch.from_numpy(pca.mean_).float(),
        components=torch.from_numpy(pca.components_).float(),
        explained_variance_ratio=torch.from_numpy(pca.explained_variance_ratio_).float(),
        sample_count=samples, fit_seconds=time.perf_counter() - started,
        peak_rss_bytes=_process_peak_rss_bytes(), source_id=source_id,
        member_names=members(node_name), member_shapes=member_shapes(graph, node_name), spatial_method=spatial_method,
    )


def prepare_complete_pca_bank(
    graph, loader, correction_nodes, factors, max_rank, device,
    pca_root: Path, identity: Mapping[str, object], quarantine_root: Path,
    *, load_only: bool = False, spatial_method="interpolate", strict_rank=False,
) -> Dict[Tuple[str, int], FullFeaturePCA]:
    """Build/verify a complete-training-only bank shared by all missing scenarios."""
    if getattr(loader.dataset, 'split', 'train') != 'train' or getattr(loader.dataset, 'augment', False):
        raise ValueError('Shared PCA requires the unaugmented training split')
    if getattr(loader, 'drop_last', False):
        raise ValueError('Shared PCA cannot drop training samples')
    identity = dict(identity, max_rank=max_rank, protocol=PROTOCOL,
                    joint_code_sha256=file_sha256(Path(__file__).with_name('joint.py')),
                    pca_code_sha256=file_sha256(Path(__file__)))
    identity.update(spatial_method=spatial_method, strict_rank=strict_rank)
    bank_id = stable_hash(identity)[:16]
    output = Path(pca_root) / bank_id
    graph.eval()
    first = next(iter(loader))
    with torch.no_grad():
        forward_with_look(graph, first['oct'].to(device), first['cfp'].to(device), **({'counts': first['counts']} if 'counts' in first else {}))
    entries = []
    for node in correction_nodes:
        shape = read_site(graph, node).shape
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
                    if basis.source_id != source_id or basis.node_name != node or basis.factor != factor or basis.spatial_method != spatial_method:
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
                basis = fit_complete_pca(graph, loader, node, factor, max_rank, device, source_id, spatial_method, strict_rank)
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
    *, feature_pairs=None,
) -> Dict[int, LOOKArtifact]:
    def pairs():
        if feature_pairs is not None:
            return feature_pairs()
        return iter_feature_pairs(
            graph, loader, node_name, missing_pattern, factor, device, upstream_artifacts, filler,
            **({"spatial_method": pca.spatial_method} if pca.spatial_method != "interpolate" else {})
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
            member_names=pca.member_names, member_shapes=pca.member_shapes, spatial_method=pca.spatial_method,
        )
    return artifacts


def validate_global_factor_bank(
    artifacts: Sequence[LOOKArtifact],
    correction_nodes: Sequence[str],
    factor: int,
    *,
    allow_prefix: bool = False,
) -> None:
    names = [a.node_name for a in artifacts]
    if len(set(names)) != len(names) or names != [n for n in correction_nodes if n in names]:
        raise ValueError('LOOK bank must be an ordered subset of correction sites')
    for artifact in artifacts:
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
    force_enable: bool = False,
) -> Tuple[List[LOOKArtifact], List[Dict[str, object]], Dict[str, object]]:
    """Fit one node-wise LOOK bank with a single shared spatial factor."""
    from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle
    from look.analysis.matrices import analyze_look_bank, analyze_correction_matrix

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = output_dir / "selected"
    completion_path = output_dir / "bank_complete.json"
    history_path = output_dir / "search_history.json"
    selected, decisions, search_history = [], [], []
    decision_dir = output_dir / 'decisions'
    decision_dir.mkdir(exist_ok=True)
    for site_index, node_name in enumerate(correction_nodes):
        basis = next(p for (name, _), p in pca_bank.items() if name == node_name)
        node_factor = 1 if len(basis.feature_shape) == 1 else factor
        pca = pca_bank[(node_name, node_factor)]
        decision_path = decision_dir / f'{site_index + 1:02d}_{node_name}.json'
        decision_identity = dict(protocol=PROTOCOL, node=node_name, site_index=site_index,
            factor=factor, latent_dims=list(latent_dims), pca_source_id=pca.source_id,
            missing_pattern=missing_pattern, filling=(filler or NormalizedMeanFiller()).name,
            primary_metric=primary_metric, upstream_decisions_sha256=stable_hash(decisions))
        if force_enable: decision_identity['force_enable'] = True
        if resume and decision_path.exists():
            decision = json.loads(decision_path.read_text())
            if decision['identity'] != decision_identity:
                raise ValueError(f'Decision identity mismatch: {decision_path}')
            if decision['enabled']:
                path = output_dir / decision['artifact']['path']
                if path.parent != output_dir / 'candidates' or path.is_symlink() or file_sha256(path) != decision['artifact']['sha256']:
                    raise ValueError('Accepted decision artifact mismatch')
                selected.append(LOOKArtifact.load(path))
            decisions.append(decision)
            search_history.extend(decision['candidates'])
            continue
        baseline = evaluate_missing(graph, validation_loader, device,
            fixed_pattern=missing_pattern, artifact_banks={missing_pattern: selected}, filler=filler)
        baseline_path = output_dir / 'predictions' / f'{site_index + 1:02d}_{node_name}_off.npz'
        save_prediction_bundle(baseline, baseline_path)
        baseline_score = float(baseline['metrics'][primary_metric])
        if not np.isfinite(baseline_score):
            raise ValueError('Non-finite baseline selection score')
        node_best = None
        node_best_score = -float("inf")
        node_records = []
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
        for latent_dim, candidate in sorted(candidates.items()):
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
            prediction_path = output_dir / "predictions" / f"{site_index + 1:02d}_{node_name}_d{latent_dim}.npz"
            save_prediction_bundle(result, prediction_path)
            score = float(result["metrics"][primary_metric])
            if not np.isfinite(score):
                raise ValueError('Non-finite candidate selection score')
            record = {
                "bank_factor": factor,
                "node": node_name,
                "factor": node_factor,
                "latent_dim": latent_dim,
                "ridge_lambda": candidate.ridge_lambda,
                "primary_metric": primary_metric,
                "primary_score": score,
                "metrics": result["metrics"],
                "prediction_path": str(prediction_path),
                "prediction_sha256": file_sha256(prediction_path),
            }
            node_records.append(record)
            if score > node_best_score:
                node_best_score = score
                node_best = candidate
        if node_best is None:
            raise RuntimeError(f"No valid LOOK candidate for {node_name}")
        enabled = force_enable or node_best_score > baseline_score
        artifact_path = output_dir / 'candidates' / f'{node_name}_x{node_factor}_d{node_best.latent_dim}.pt'
        decision = dict(identity=decision_identity, baseline_score=baseline_score, baseline_metrics=baseline["metrics"],
            baseline_prediction=dict(path=str(baseline_path), sha256=file_sha256(baseline_path)),
            best_candidate_score=node_best_score, enabled=enabled,
            best_dimension=node_best.latent_dim,
            selected_score=node_best_score if enabled else baseline_score,
            reason='predeclared_all_on' if force_enable else 'strict_improvement' if enabled else 'no_strict_improvement',
            artifact=dict(path=str(artifact_path.relative_to(output_dir)), sha256=file_sha256(artifact_path)),
            candidates=node_records,
            best_candidate_matrix_diagnostics=analyze_correction_matrix(node_best))
        write_json_atomic(decision, decision_path)
        decisions.append(decision)
        if enabled:
            selected.append(node_best)
        search_history.extend(node_records)
        write_json_atomic(search_history, history_path)
        print(f'LOOK {node_name} x{factor}: baseline={baseline_score:.6f} '
              f'candidate={node_best_score:.6f} enabled={enabled}', flush=True)

    validate_global_factor_bank(selected, correction_nodes, factor)
    if resume and completion_path.exists():
        load_selected_bank(output_dir)
        return selected, search_history, json.loads(completion_path.read_text())
    validation = evaluate_missing(
        graph,
        validation_loader,
        device,
        fixed_pattern=missing_pattern,
        artifact_banks={missing_pattern: selected},
        filler=filler,
    )["metrics"]
    save_selected_bank(selected, output_dir)
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
        "decisions": decisions,
        "evaluated_sites": len(decisions),
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
    force_enable: bool = False,
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
    best_count = float('inf')
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
            resume=resume, force_enable=force_enable,
        )
        score = float(completion["primary_score"])
        factor_records.append(completion)
        all_history.extend(history)
        if (score, -len(artifacts), factor) > (best_score, -best_count, best_factor):
            best_score, best_factor = score, factor
            best_count = len(artifacts)
            best_artifacts = artifacts

    if best_artifacts is None:
        raise RuntimeError("No complete LOOK factor bank was produced")
    selected_dir = output_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for stale in selected_dir.glob("*.pt"):
        stale.unlink()
    save_selected_bank(best_artifacts, output_dir)
    selection = {
        "status": "selected",
        "selection_scope": "one_global_spatial_factor_per_artifact_bank",
        "vector_node_factor": 1,
        "primary_metric": primary_metric,
        "tie_break": "fewer_active_sites_then_larger_factor",
        "selected_factor": best_factor,
        "selected_primary_score": best_score,
        "factor_banks": factor_records,
    }
    write_json_atomic(all_history, output_dir / "search_history.json")
    write_json_atomic(selection, output_dir / "factor_selection.json")
    return best_artifacts, all_history
