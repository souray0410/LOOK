"""Exact EmbraceNet complete-reference PCA and explicitly selected LOOK trajectories."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from look.evaluation.embracenet import (
    availability_matrix,
    evaluate_single_missing,
    isolated_embracenet_sampling,
)
from look.methods.affine_family import FamilyArtifact, fit_map
from look.methods.family_greedy import BASIS_CONSTRAINED, validate_candidates
from look.methods.independent_greedy import SelectionPaused, fit_trajectory
from look.methods.joint import PROTOCOL, member_shapes, members, read_site, site_level, write_site
from look.methods.linear_operator import fingerprint
from look.methods.linear_vector import ResidualMoments, estimated_workspace_bytes
from look.methods.operator import (
    FullFeaturePCA,
    LOOKArtifact,
    _gcv_lambda,
    downsample_flatten,
    fit_complete_pca,
)
from look.models.embracenet import _get_embracenet_operation, reset_embracenet_inputs
from look.runtime.host_checkpoint import atomic_save
from look.runtime.provenance import write_json_atomic
from look.runtime.state import file_sha256, stable_hash
from look.studies.embracenet_adapter import forward_embracenet_with_look
from look.evaluation.evaluator import save_prediction_bundle


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


@torch.no_grad()
def _complete_vector_factory(graph, loader, node, factor, device, should_pause=lambda: False):
    for batch in loader:
        if should_pause():
            raise SelectionPaused()
        availability = availability_matrix(len(batch["label"]), "complete", device)
        result = forward_embracenet_with_look(
            graph, batch["oct"].to(device), batch["cfp"].to(device), batch["counts"],
            availability, stop_node=node,
        )
        feature = result["output"].detach()
        flat, shape = downsample_flatten(feature, factor)
        yield flat.cpu(), tuple(feature.shape[1:]), shape


@torch.no_grad()
def fit_stochastic_embraced_pca(graph, loader, max_rank, device, source_id, should_pause=lambda: False):
    """Exact PCA sufficient statistics for the author's random complete embracement."""
    graph.eval()
    count = 0
    total_mean = None
    total_second = None
    started = time.perf_counter()
    with isolated_embracenet_sampling(graph):
        for batch in loader:
            if should_pause():
                raise SelectionPaused()
            availability = availability_matrix(len(batch["label"]), "complete", device)
            forward_embracenet_with_look(
                graph, batch["oct"].to(device), batch["cfp"].to(device), batch["counts"],
                availability, stop_node="joint_participant_feature",
            )
            oct_feature = graph.get_node_by_name("oct_participant_feature").feature_message.current_state
            cfp_feature = graph.get_node_by_name("cfp_participant_feature").feature_message.current_state
            moments = _get_embracenet_operation(graph).analytical_moments(
                [oct_feature, cfp_feature], availability,
                torch.full_like(availability, 0.5),
            )
            mean = moments["mean"].detach().cpu().double()
            variance = moments["variance_diagonal"].detach().cpu().double()
            batch_mean = mean.sum(0)
            batch_second = mean.T @ mean + torch.diag(variance.sum(0))
            total_mean = batch_mean if total_mean is None else total_mean + batch_mean
            total_second = batch_second if total_second is None else total_second + batch_second
            count += len(mean)
    if count < 2 or total_mean is None or total_second is None:
        raise ValueError("Stochastic complete-reference PCA needs at least two participants")
    mean = total_mean / count
    covariance = (total_second / count - torch.outer(mean, mean))
    covariance = (covariance + covariance.T) * 0.5
    variance = covariance.diag().clamp_min(1e-12)
    std = variance.sqrt()
    standardized = covariance / torch.outer(std, std)
    standardized = (standardized + standardized.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(standardized)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    rank = min(max_rank, count - 1, len(mean))
    if rank != max_rank:
        raise ValueError("Requested stochastic PCA rank is infeasible")
    components = eigenvectors[:, order[:rank]].T.contiguous()
    total = eigenvalues.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("Stochastic complete-reference covariance is degenerate")
    ratio = eigenvalues[:rank] / total
    return FullFeaturePCA(
        node_name="embraced_feature", factor=1, feature_shape=(len(mean),),
        downsample_shape=(len(mean),), mean=mean.float(), std=std.float(),
        pca_mean=torch.zeros(len(mean), dtype=torch.float32),
        components=components.float(), explained_variance_ratio=ratio.float(),
        sample_count=count, fit_seconds=time.perf_counter() - started,
        peak_rss_bytes=_peak_rss_bytes(), source_id=source_id,
        member_names=("embraced_feature",), member_shapes=((len(mean),),),
        protocol=PROTOCOL, split_rule="channel_split_and_restore_member_shapes_v1",
        spatial_method="interpolate",
    )


def prepare_embracenet_pca_bank(
    graph, loader, sites, factor, max_rank, device, output, identity, should_pause=lambda: False
):
    if loader.dataset.split != "train" or loader.dataset.augment or loader.drop_last:
        raise ValueError("EmbraceNet PCA requires complete unaugmented train samples")
    if graph.training or any(parameter.requires_grad for parameter in graph.parameters()):
        raise ValueError("Frozen eval EmbraceNet graph required for PCA")
    identity = dict(
        identity,
        schema="look_embracenet_complete_reference_pca_v1",
        factor=factor, max_rank=max_rank,
        embraced_reference="analytic_author_stochastic_second_moment",
        code_sha256=file_sha256(__file__),
    )
    root = Path(output) / stable_hash(identity)[:16]
    root.mkdir(parents=True, exist_ok=True)
    bank = {}
    manifest_rows = []
    for index, node in enumerate(sites):
        if should_pause():
            raise SelectionPaused()
        node_factor = 1 if node in ("joint_features", "joint_participant_feature", "embraced_feature") else factor
        path = root / f"{index:02d}_{node}_x{node_factor}.pt"
        meta = path.with_suffix(".json")
        source_id = f"{root.name}/{node}_x{node_factor}"
        basis = None
        if path.exists() and meta.exists():
            record = json.loads(meta.read_text())
            if record.get("sha256") != file_sha256(path) or record.get("source_id") != source_id:
                raise ValueError("EmbraceNet PCA cache identity changed")
            basis = FullFeaturePCA.load(path)
        elif path.exists() or meta.exists():
            raise ValueError("Partial EmbraceNet PCA cache entry exists")
        if basis is None:
            if node == "embraced_feature":
                basis = fit_stochastic_embraced_pca(
                    graph, loader, max_rank, device, source_id, should_pause
                )
            else:
                basis = fit_complete_pca(
                    graph, loader, node, node_factor, max_rank, device, source_id,
                    strict_rank=True,
                    feature_factory=lambda n=node, f=node_factor: _complete_vector_factory(
                        graph, loader, n, f, device, should_pause
                    ),
                )
            basis.save(path)
            write_json_atomic({
                "source_id": source_id, "sha256": file_sha256(path),
                "samples": basis.sample_count, "reference": (
                    "analytic_author_stochastic_second_moment" if node == "embraced_feature"
                    else "deterministic_complete_feature"
                ),
            }, meta)
        if basis.node_name != node or basis.factor != node_factor or basis.source_id != source_id:
            raise ValueError("EmbraceNet PCA basis metadata changed")
        bank[(node, node_factor)] = basis
        manifest_rows.append({
            "node": node, "factor": node_factor, "path": str(path),
            "sha256": file_sha256(path), "source_id": source_id,
        })
    manifest = {
        "schema": identity["schema"], "identity": identity, "entries": manifest_rows,
        "test_access": False,
    }
    write_json_atomic(manifest, root / "bank_manifest.json")
    return bank


@torch.no_grad()
def capture_sites(graph, batch, sites, bases, device, pattern="complete", upstream=()):
    if graph.training or any(parameter.requires_grad for parameter in graph.parameters()):
        raise ValueError("Frozen eval graph required")
    if pattern not in ("complete", "oct_missing", "cfp_missing"):
        raise ValueError("Unknown EmbraceNet capture pattern")
    by_level = {}
    for site in sites:
        by_level.setdefault(site_level(graph, site), []).append(site)
    by_node = {artifact.node_name: artifact for artifact in upstream}
    if len(by_node) != len(upstream):
        raise ValueError("Duplicate upstream correction")
    if pattern == "complete" and upstream:
        raise ValueError("Complete reference cannot contain missing-side LOOK corrections")

    availability = availability_matrix(len(batch["label"]), pattern, device)
    reset_embracenet_inputs(
        graph, batch["oct"].to(device), batch["cfp"].to(device), batch["counts"], availability
    )
    embrace_level = site_level(graph, "embraced_feature")
    result = {}
    with isolated_embracenet_sampling(graph):
        for level in range(-1, max(by_level) + 1):
            analytic = pattern == "complete" and level == embrace_level
            moments = None
            if analytic:
                oct_feature = graph.get_node_by_name("oct_participant_feature").feature_message.current_state
                cfp_feature = graph.get_node_by_name("cfp_participant_feature").feature_message.current_state
                moments = _get_embracenet_operation(graph).analytical_moments(
                    [oct_feature, cfp_feature], availability, torch.full_like(availability, 0.5)
                )
            elif level >= 0:
                graph.forward(levels=[level])
            if pattern != "complete":
                for name, artifact in by_node.items():
                    if site_level(graph, name) == level:
                        if artifact.protocol != PROTOCOL or artifact.split_rule != "channel_split_and_restore_member_shapes_v1":
                            raise ValueError("Upstream LOOK protocol changed")
                        write_site(graph, name, artifact.apply_feature(read_site(graph, name)))
            for name in by_level.get(level, ()):
                basis = bases[name]
                if analytic and name == "embraced_feature":
                    feature = moments["mean"]
                    conditional_variance = moments["variance_diagonal"]
                else:
                    feature = read_site(graph, name).detach()
                    conditional_variance = torch.zeros_like(feature)
                flat, shape = downsample_flatten(feature, basis.factor, basis.spatial_method)
                if tuple(feature.shape[1:]) != tuple(basis.feature_shape) or tuple(shape) != tuple(basis.downsample_shape):
                    raise ValueError("EmbraceNet feature shape changed")
                if conditional_variance.numel() != feature.numel():
                    raise ValueError("Conditional variance shape changed")
                variance_flat = conditional_variance.reshape_as(feature)
                if feature.ndim == 4 and basis.factor != 1:
                    if torch.count_nonzero(variance_flat):
                        raise ValueError("Spatial stochastic variance propagation is not registered")
                    variance_flat = torch.zeros_like(flat)
                else:
                    variance_flat = variance_flat.reshape(flat.shape)
                result[name] = (flat.detach().cpu().clone(), variance_flat.detach().cpu().clone())
    return result


def _update_with_variance(stats, x, residual_mean, conditional_variance):
    stats.update(x, residual_mean)
    extra = conditional_variance.detach().to(dtype=torch.float64, device="cpu").sum()
    stats.syy += extra


class EmbraceNetFamilyStatistics:
    def __init__(self, graph, loader, bases, device, root, identity, workspace_bytes, rank,
                 should_pause=lambda: False, projected_ranks=()):
        if loader.dataset.split != "train" or loader.dataset.augment or loader.drop_last:
            raise ValueError("Complete unaugmented train only")
        if not identity:
            raise ValueError("Immutable identity required")
        self.graph, self.loader, self.bases, self.device = graph, loader, bases, device
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.identity = fingerprint({
            "version": "embracenet_family_full_moments_v1", "source": identity,
            "bases": {name: asdict(basis) for name, basis in bases.items()},
            "stochastic_reference": "analytic_second_moment_at_embraced_feature",
        })
        self.budget, self.rank, self.check = workspace_bytes, rank, should_pause
        self.projected_ranks = tuple(projected_ranks)
        self.projected = {}
        self.metrics = {"full_forwards": 0, "missing_forwards": 0, "completed_hits": 0, "skipped_batches": 0}

    def statistics(self, pattern, upstream, sites):
        if pattern not in ("oct_missing", "cfp_missing") or not sites or len(set(sites)) != len(sites):
            raise ValueError("Invalid sites or missing pattern")
        dims = [self.bases[name].std.numel() for name in sites]
        resident = sum(8 * (2*d*d + 2*d + 1) for d in dims)
        projected_resident = len(sites) * sum(8 * (2*q*q + 2*q + 1) for q in self.projected_ranks)
        required = resident + projected_resident + max(estimated_workspace_bytes(d, self.rank) for d in dims)
        if required > self.budget:
            raise MemoryError("EmbraceNet family dense fitting exceeds admitted workspace")
        sid = fingerprint({
            "reference": self.identity, "pattern": pattern, "sites": list(sites),
            "upstream": [artifact.record() for artifact in upstream],
            "projected_ranks": self.projected_ranks,
        })
        path = self.root / (sid + ".pt")
        lock = self.root / (sid + ".lock")
        import fcntl
        with lock.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._collect(path, sid, pattern, upstream, sites)

    def _collect(self, path, sid, pattern, upstream, sites):
        projected = {name: {q: ResidualMoments.empty(q) for q in self.projected_ranks} for name in sites}
        stats = {name: ResidualMoments.empty(self.bases[name].std.numel()) for name in sites}
        stamps = []
        if path.exists():
            record = torch.load(path, map_location="cpu", weights_only=False)
            payload = record["payload"]
            if fingerprint(payload) != record["sha256"] or payload["identity"] != sid or payload["sites"] != list(sites):
                raise ValueError("Changed EmbraceNet family statistics identity")
            stamps = payload["batch_stamps"]
            stats = {name: ResidualMoments(**payload["statistics"][name]) for name in sites}
            projected = {
                name: {int(q): ResidualMoments(**value) for q, value in payload["projected"][name].items()}
                for name in sites
            }
            if payload["complete"]:
                self.projected = projected; self.metrics["completed_hits"] += 1
                return stats
        self.projected = projected
        cursor = len(stamps)

        def save_state(complete=False):
            payload = {
                "identity": sid, "sites": list(sites), "batch_stamps": stamps,
                "complete": complete,
                "statistics": {name: vars(value).copy() for name, value in stats.items()},
                "projected": {name: {q: vars(v).copy() for q, v in values.items()} for name, values in projected.items()},
            }
            atomic_save(path, {"payload": payload, "sha256": fingerprint(payload)})

        count = 0
        for index, batch in enumerate(self.loader):
            count = index + 1
            stamp = fingerprint({
                "ids": batch["participant_id"], "labels": batch["label"], "counts": batch["counts"],
                "oct_shape": list(batch["oct"].shape), "cfp_shape": list(batch["cfp"].shape),
            })
            if index < cursor:
                if stamp != stamps[index]:
                    raise ValueError("Committed EmbraceNet participant data/order changed")
                self.metrics["skipped_batches"] += 1
                continue
            if self.check():
                save_state(); raise SelectionPaused()
            complete = capture_sites(self.graph, batch, sites, self.bases, self.device, "complete")
            missing = capture_sites(self.graph, batch, sites, self.bases, self.device, pattern, upstream)
            self.metrics["full_forwards"] += 1; self.metrics["missing_forwards"] += 1
            for name in sites:
                basis = self.bases[name]
                full_mean, full_variance = complete[name]
                missing_mean, _ = missing[name]
                std = basis.std
                x = (missing_mean - basis.mean) / std
                residual_mean = (full_mean - missing_mean) / std
                variance = full_variance / std.square()
                _update_with_variance(stats[name], x, residual_mean, variance)
                for q, projected_stats in projected[name].items():
                    q_basis = basis.components[:q].to(dtype=torch.float64, device="cpu")
                    _update_with_variance(
                        projected_stats,
                        x.double() @ q_basis.T,
                        residual_mean.double() @ q_basis.T,
                        variance.double() * q_basis.square().sum(dim=0),
                    )
            stamps.append(stamp)
            if count % 32 == 0:
                save_state()
        if count < cursor:
            raise ValueError("EmbraceNet fitting stream shortened")
        save_state(True)
        return stats


def fit_embracenet_family_trajectory(
    graph, train_loader, dev_loader, *, arm, pattern, sites, factor, candidates,
    pca_bank, identity, output, device, workspace_bytes, should_pause=lambda: False,
    penalty_policy="prefix_train_pca_gcv", search="positive_forward_tree",
):
    if search not in ("positive_forward_tree", "best_forward"):
        raise ValueError("Explicit registered trajectory required")
    validate_candidates(arm, candidates, penalty_policy)
    if graph.training or any(parameter.requires_grad for parameter in graph.parameters()):
        raise ValueError("Frozen eval EmbraceNet graph required")
    if pattern not in ("oct_missing", "cfp_missing"):
        raise ValueError("Invalid EmbraceNet missing pattern")
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    bases = {}
    for site in sites:
        matches = [basis for (name, _), basis in pca_bank.items() if name == site]
        if len(matches) != 1:
            raise ValueError("Every EmbraceNet site requires one complete-reference PCA basis")
        bases[site] = matches[0]
    scientific_identity = {
        "identity": identity, "arm": arm, "pattern": pattern, "factor": factor, "search": search,
        "candidates": candidates, "penalty_policy": penalty_policy,
        "data_role": dev_loader.dataset.split,
        "bases": {name: fingerprint(asdict(basis)) for name, basis in bases.items()},
        "complete_reference": "author_stochastic_analytic_second_moment",
    }
    collector = EmbraceNetFamilyStatistics(
        graph, train_loader, bases, device, root / "family_moments", scientific_identity,
        workspace_bytes, max(candidate["rank"] or 1 for candidate in candidates), should_pause,
        projected_ranks=sorted({candidate["rank"] for candidate in candidates}),
    )
    ready = {}

    def fit(site, upstream, folder):
        basis = bases[site]; dimension = basis.std.numel(); folder.mkdir(parents=True, exist_ok=True)
        prefix = fingerprint([artifact.record() for artifact in upstream])
        key = (prefix, None)
        if key not in ready:
            start = max((sites.index(artifact.node_name) for artifact in upstream), default=-1) + 1
            targets = list(sites[start:])
            ready.clear(); ready[key] = collector.statistics(pattern, upstream, targets)
            write_json_atomic(collector.metrics, root / "feature_costs.json")
        stats = ready[key][site]
        for candidate in candidates:
            q = candidate["rank"]; q_basis = basis.components[:q].to(dtype=torch.float64, device="cpu")
            projected = collector.projected[site][q]
            ridge = _gcv_lambda(
                q_basis @ stats.cxx @ q_basis.T,
                q_basis @ stats.cxy @ q_basis.T,
                float(projected.syy), stats.count, q,
            )
            template = LOOKArtifact(
                site, pattern, "availability_zero_mask", basis.factor, q,
                basis.feature_shape, basis.downsample_shape, basis.mean, basis.std,
                basis.pca_mean, basis.components[:q], torch.zeros(q, q), torch.zeros(q),
                ridge, 0.0, 0.0, float(basis.explained_variance_ratio[:q].sum()),
                basis.fit_seconds, basis.peak_rss_bytes, basis.source_id,
                basis.member_names, basis.member_shapes, basis.protocol,
                basis.split_rule, basis.spatial_method,
            )
            mapping = fit_map(stats, q, ridge, arm=arm, basis=basis.components[:q])
            mapping.diagnostics.update(
                selection=search, penalty_policy=penalty_policy,
                complete_reference="author_stochastic_analytic_second_moment",
                conditional_variance_in_syy=True,
            )
            yield json.dumps(candidate, sort_keys=True), FamilyArtifact(template, mapping)

    def evaluate(bank):
        if should_pause():
            raise SelectionPaused()
        result = evaluate_single_missing(graph, dev_loader, device, pattern, bank)
        values = hashlib.sha256()
        for name in ("participant_ids", "labels", "logits"):
            array = np.ascontiguousarray(result[name]); values.update(name.encode())
            values.update(str(array.dtype).encode()); values.update(str(array.shape).encode()); values.update(array.tobytes())
        values_sha = values.hexdigest()
        key = fingerprint([artifact.record() for artifact in bank])
        prediction = root / "predictions" / f"{key}_{values_sha}.npz"
        if prediction.exists():
            with np.load(prediction, allow_pickle=False) as saved:
                if any(not np.array_equal(saved[name], result[name]) for name in ("participant_ids", "labels", "logits")):
                    raise ValueError("Stored EmbraceNet prediction evidence changed")
        else:
            save_prediction_bundle(result, prediction)
        return {
            "role": "development", "data_role": dev_loader.dataset.split,
            "score": float(result["metrics"]["macro_f1"]), "prediction": str(prediction),
            "sha256": file_sha256(prediction), "values_sha256": values_sha,
            "metrics": result["metrics"],
        }

    def save_artifact(artifact, path):
        atomic_save(path, artifact.record())

    def load_artifact(path):
        return FamilyArtifact.from_record(torch.load(path, map_location="cpu", weights_only=False))

    bank, result = fit_trajectory(
        identity=scientific_identity, sites=sites, mode=search,
        output=root, fit_candidates=fit, evaluate=evaluate,
        save_artifact=save_artifact, load_artifact=load_artifact,
        should_pause=should_pause,
    )
    records = [artifact.record() for artifact in bank]
    atomic_save(root / "bank.pt", {
        "identity": scientific_identity, "bank": records, "sha256": fingerprint(records)
    })
    replay = evaluate([FamilyArtifact.from_record(record) for record in records])
    if replay["values_sha256"] != result["final"]["values_sha256"]:
        raise ValueError("Reloaded EmbraceNet full-MHD predictions changed")
    write_json_atomic({
        "full_mhd_replay": True, "scientific_acceptance": False, "test_access": False
    }, root / "replay.json")
    return bank, result


def load_embracenet_bank(path):
    record = torch.load(Path(path) / "bank.pt", map_location="cpu", weights_only=False)
    if fingerprint(record["bank"]) != record["sha256"]:
        raise ValueError("EmbraceNet family bank changed")
    return [FamilyArtifact.from_record(value) for value in record["bank"]]


def fit_embracenet_search_suite(
    graph, train_loader, dev_loader, *, sites, candidates_by_arm, output, identity,
    **kwargs,
):
    """Run each approved fit family independently on the same frozen host.

    Enumerate single sites, then best-forward and the positive-forward tree.
    Single-site outcomes never filter another search. Callers bind data, host,
    source, seed and resource qualification into identity before GPU dispatch.
    Each trajectory retains its own resumable contract and prediction evidence.
    """
    arms = ('shared_pca_ridge', 'pca_free_mean', 'rrr_shared_intercept', 'residual_rrr')
    if set(candidates_by_arm) != set(arms):
        raise ValueError('All four registered fit families required')
    if not sites or len(set(sites)) != len(sites):
        raise ValueError('Ordered unique sites required')
    root = Path(output)
    records = []
    for arm in arms:
        for pattern in ('oct_missing', 'cfp_missing'):
            configurations = [('single_site', [site], f'single_{i:03d}')
                              for i, site in enumerate(sites)]
            configurations += [(mode, list(sites), mode)
                               for mode in ('best_forward', 'positive_forward_tree')]
            for mode, selected_sites, directory in configurations:
                run_identity = dict(parent=identity, coverage='full_fit_search_suite_v1',
                                    search=mode, sites=selected_sites)
                target = root / arm / pattern / directory
                completed = target / 'suite_completed.json'
                request_id = stable_hash(dict(identity=run_identity, arm=arm, pattern=pattern,
                                              candidates=candidates_by_arm[arm]))
                if completed.exists():
                    receipt = json.loads(completed.read_text())
                    if receipt['request_id'] != request_id:
                        raise ValueError('Completed suite trajectory identity changed')
                    for path, digest in receipt['files'].items():
                        if file_sha256(path) != digest:
                            raise ValueError('Completed suite trajectory artifact changed')
                    records.append(receipt['record'])
                    continue
                _, result = fit_embracenet_family_trajectory(
                    graph, train_loader, dev_loader, arm=arm, pattern=pattern,
                    sites=selected_sites, candidates=candidates_by_arm[arm],
                    identity=run_identity, output=target,
                    search='positive_forward_tree' if mode == 'single_site' else mode,
                    **kwargs,
                )
                record = dict(arm=arm, pattern=pattern, search=mode,
                              sites=selected_sites, output=str(target),
                              selection_sha256=file_sha256(target / 'selection.json'),
                              final=result['final'])
                paths = [target / name for name in ('selection.json', 'bank.pt', 'replay.json')]
                paths.append(Path(result['final']['prediction']))
                write_json_atomic(dict(request_id=request_id, record=record,
                                       files={str(p): file_sha256(p) for p in paths}), completed)
                records.append(record)
    write_json_atomic(dict(identity=identity, records=records, test_access=False,
                           scientific_acceptance=False), root / 'coverage.json')
    return records
