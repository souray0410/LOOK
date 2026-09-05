"""Isolated mechanism controls. Original LOOK kernels and artifacts are immutable.

Ridge algebra, PCA, dimensions, strict node acceptance and factor tie rules match
look.py. Only training upstream conditioning / joint writeback is varied.
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import torch

from .joint import members, member_shapes, read_site, write_site, site_level
from .look import (PROTOCOL, LOOKArtifact, LatentSufficientStatistics, _gcv_lambda,
    _reset_inputs, apply_artifact, downsample_flatten, save_selected_bank, load_selected_bank)
from .evaluate import save_prediction_bundle
from .filling import NormalizedMeanFiller
from .missingness import missingness_plan
from .stable_metrics import logit_metrics, probabilities_from_logits
from .state import atomic_write_json, file_sha256, stable_hash

POLICIES = ('joint', 'independent_fit', 'missing_only', 'bias_only', 'self_input_missing_only')


def forward_control(graph, oct_tensor, cfp_tensor, artifacts=(), *, policy='joint',
                    pattern='complete', stop_node=None):
    if policy not in POLICIES:
        raise ValueError(policy)
    if pattern not in ('complete', 'oct_missing', 'cfp_missing'):
        raise ValueError(pattern)
    _reset_inputs(graph, oct_tensor, cfp_tensor)
    if len({a.node_name for a in artifacts}) != len(artifacts):
        raise ValueError('Duplicate correction sites')
    target = stop_node or 'fusion_logits'
    for level in range(-1, site_level(graph, target) + 1):
        if level >= 0:
            graph.forward(levels=[level])
        for a in artifacts:
            if site_level(graph, a.node_name) != level:
                continue
            if a.protocol != PROTOCOL or a.split_rule != 'channel_split_and_restore_member_shapes_v1':
                raise ValueError('Correction protocol mismatch')
            if a.member_names and tuple(a.member_names) != members(a.node_name):
                raise ValueError('Correction member order mismatch')
            if a.member_shapes and tuple(map(tuple, a.member_shapes)) != member_shapes(graph, a.node_name):
                raise ValueError('Correction member shape mismatch')
            before = read_site(graph, a.node_name)
            if policy == 'self_input_missing_only':
                from .self_input import apply_self_input
                if a.missing_pattern != pattern:
                    raise ValueError('Self-input artifact direction mismatch')
                after = apply_self_input(before, a)
            else:
                after = apply_artifact(before, a)
            if policy in ('missing_only', 'self_input_missing_only') and a.node_name.startswith('joint_'):
                if pattern == 'complete':
                    raise ValueError('Missing-only correction requires a missing direction')
                first = graph.get_node_by_name(members(a.node_name)[0]).feature_message.current_state
                channels = first.shape[2] if a.node_name == 'joint_input' else first.shape[1]
                # Both policies write only the missing branch. self_input also masks the predictor input.
                after = (torch.cat((after[:, :channels], before[:, channels:]), 1)
                         if pattern == 'oct_missing' else
                         torch.cat((before[:, :channels], after[:, channels:]), 1))
            write_site(graph, a.node_name, after)
    return read_site(graph, target)


@torch.no_grad()
def evaluate_control(graph, loader, device, banks=None, *, policy='joint', filler,
                     fixed_pattern=None, random_ratio=None, mask_seed=3407, predict=None):
    if (fixed_pattern is None) == (random_ratio is None):
        raise ValueError('Specify one fixed pattern or random ratio')
    graph.eval()
    banks = banks or {}
    plan, metadata = ({}, None) if random_ratio is None else missingness_plan(
        loader.dataset.participant_ids, random_ratio, mask_seed)
    ys, zs, ids, patterns_all = [], [], [], []
    for batch in loader:
        oct_t, cfp_t = batch['oct'].to(device), batch['cfp'].to(device)
        ids_batch = list(batch['participant_id'])
        patterns = ([fixed_pattern] * len(ids_batch) if fixed_pattern is not None
                    else [plan[str(pid)] for pid in ids_batch])
        logits = torch.empty((len(ids_batch), graph.num_classes), device=device)
        for pattern in sorted(set(patterns)):
            idx = torch.tensor([i for i, p in enumerate(patterns) if p == pattern], device=device)
            o, c = filler.fill(oct_t[idx], cfp_t[idx], pattern)
            logits[idx] = (predict(o, c, pattern) if predict else forward_control(
                graph, o, c, banks.get(pattern, ()), policy=policy, pattern=pattern))
        ys.append(batch['label'].numpy()); zs.append(logits.cpu().numpy())
        ids.extend(ids_batch); patterns_all.extend(patterns)
    labels, logits = np.concatenate(ys), np.concatenate(zs).astype(np.float64)
    return dict(labels=labels, logits=logits, probabilities=probabilities_from_logits(logits),
                scores=logits[:, 1]-logits[:, 0], participant_ids=np.asarray(ids),
                patterns=np.asarray(patterns_all), metrics=logit_metrics(labels, logits),
                **({'missingness': metadata} if metadata is not None else {}))


@torch.no_grad()
def fit_candidates(graph, loader, node, pattern, factor, dims, max_rank, device,
                   pca, upstream, filler, policy):
    if pca.node_name != node or pca.factor != factor:
        raise ValueError('Mismatched shared PCA')
    graph.eval()
    rank = min(max_rank, len(pca.components))
    components = pca.components[:rank]
    stats = LatentSufficientStatistics(rank)
    fit_upstream = () if policy == 'independent_fit' else upstream
    for batch in loader:
        o, c = batch['oct'].to(device), batch['cfp'].to(device)
        full = forward_control(graph, o, c, stop_node=node).detach()
        o, c = filler.fill(o, c, pattern)
        missing = forward_control(graph, o, c, fit_upstream, policy=policy,
                                  pattern=pattern, stop_node=node).detach()
        full_flat, shape = downsample_flatten(full, factor)
        missing_flat, shape2 = downsample_flatten(missing, factor)
        if shape != shape2 or shape != tuple(pca.downsample_shape):
            raise ValueError('PCA/pair geometry mismatch')
        if policy == 'self_input_missing_only' and node.startswith('joint_'):
            from .self_input import neutralize_retained
            full_flat = neutralize_retained(full_flat, pca, pattern)
            missing_flat = neutralize_retained(missing_flat, pca, pattern)
        full_z = (((full_flat.cpu()-pca.mean)/pca.std)-pca.pca_mean) @ components.T
        missing_z = (((missing_flat.cpu()-pca.mean)/pca.std)-pca.pca_mean) @ components.T
        stats.update(missing_z, full_z)
    result = {}
    for d in sorted(set(min(x, rank) for x in dims if x > 0)):
        cxx, cxy, tss, mx, my, n = stats.centered(d)
        ridge = 0. if policy == 'bias_only' else _gcv_lambda(cxx, cxy, tss, n, d)
        w = (torch.zeros_like(cxy) if policy == 'bias_only' else
             torch.linalg.solve(cxx + ridge * torch.eye(d, dtype=cxx.dtype), cxy))
        b = my-mx@w
        rss = max(0., tss - 2*float(torch.trace(w.T@cxy)) + float(torch.trace(w.T@cxx@w)))
        result[d] = LOOKArtifact(node_name=node, missing_pattern=pattern, filling_strategy=filler.name,
            factor=factor, latent_dim=d, feature_shape=tuple(pca.feature_shape),
            downsample_shape=tuple(pca.downsample_shape), mean=pca.mean, std=pca.std,
            pca_mean=pca.pca_mean, components=components[:d], weight=w.float(), bias=b.float(),
            ridge_lambda=ridge, train_r2=1-rss/tss if tss > 0 else 0., train_mse=rss/max(1,n*d),
            pca_explained_variance=float(pca.explained_variance_ratio[:d].sum()),
            pca_fit_seconds=pca.fit_seconds, pca_peak_rss_bytes=pca.peak_rss_bytes,
            pca_source_id=pca.source_id, member_names=pca.member_names, member_shapes=pca.member_shapes)
    return result


def fit_control(graph, train, validation, device, pcas, config, pattern, policy, output, source_hash):
    """Resume decisions only within the same method, source and accepted upstream."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    factors, all_decisions = [], []
    best, best_key = None, (-float('inf'), -float('inf'), -float('inf'))
    for factor in config.downsample_factors:
        selected, decisions = [], []
        root = output/'factors'/f'x{factor}'
        for i, node in enumerate(config.correction_nodes):
            basis = next(v for (n, f), v in pcas.items() if n == node)
            nf = 1 if len(basis.feature_shape) == 1 else factor
            pca = pcas[(node, nf)]
            identity = dict(policy=policy, source=source_hash, node=node, pattern=pattern, factor=factor,
                dims=config.latent_dims, max_rank=config.max_pca_rank, pca=pca.source_id,
                accepted_upstream=stable_hash(decisions),
                fitting_upstream=[] if policy == 'independent_fit' else [a.node_name for a in selected])
            path = root/'decisions'/f'{i+1:02d}_{node}.json'
            if path.exists():
                decision = __import__('json').loads(path.read_text())
                if decision['identity'] != identity:
                    raise ValueError('Cross-method/source/upstream resume rejected')
                for rec in [decision['off_prediction'], *decision['candidates']]:
                    if file_sha256(Path(rec['prediction_path'])) != rec['prediction_sha256']:
                        raise ValueError('Candidate prediction changed')
                if decision['enabled']:
                    rec = decision['accepted_artifact']
                    if file_sha256(Path(rec['path'])) != rec['sha256']:
                        raise ValueError('Accepted correction changed')
                    selected.append(LOOKArtifact.load(Path(rec['path'])))
                decisions.append(decision)
                continue
            off = evaluate_control(graph, validation, device, {pattern:selected}, policy=policy,
                                   filler=NormalizedMeanFiller(), fixed_pattern=pattern)
            off_path = root/'predictions'/f'{i+1:02d}_{node}_off.npz'
            save_prediction_bundle(off, off_path)
            start = time.perf_counter()
            candidates = fit_candidates(graph, train, node, pattern, nf, config.latent_dims,
                config.max_pca_rank, device, pca, selected, NormalizedMeanFiller(), policy)
            fit_seconds = time.perf_counter()-start
            records, winner, score = [], None, -float('inf')
            eval_start = time.perf_counter()
            for d, candidate in sorted(candidates.items()):
                prediction = evaluate_control(graph, validation, device, {pattern:[*selected,candidate]},
                    policy=policy, filler=NormalizedMeanFiller(), fixed_pattern=pattern)
                value = float(prediction['metrics']['macro_f1'])
                if not np.isfinite(value): raise ValueError('Non-finite selection score')
                pred_path = root/'predictions'/f'{i+1:02d}_{node}_d{d}.npz'
                save_prediction_bundle(prediction, pred_path)
                records.append(dict(dimension=d, factor=nf, macro_f1=value,
                    prediction_path=str(pred_path), prediction_sha256=file_sha256(pred_path)))
                if value > score: score, winner = value, candidate
            if winner is None: raise ValueError('Empty candidate grid')
            enabled = score > off['metrics']['macro_f1']
            artifact = root/'candidates'/f'{node}_d{winner.latent_dim}.pt'
            winner.save(artifact)
            decision = dict(identity=identity, node=node, factor=nf, enabled=enabled,
                baseline_score=off['metrics']['macro_f1'], best_candidate_score=score,
                selected_score=score if enabled else off['metrics']['macro_f1'],
                chosen_dimension=winner.latent_dim if enabled else None, candidates=records,
                off_prediction=dict(prediction_path=str(off_path),prediction_sha256=file_sha256(off_path)),
                accepted_artifact=dict(path=str(artifact),sha256=file_sha256(artifact)),
                wb_fit_seconds=fit_seconds, candidate_validation_seconds=time.perf_counter()-eval_start)
            atomic_write_json(decision, path); decisions.append(decision)
            if enabled: selected.append(winner)
            print(f'{policy} {pattern} {node} x{factor}: {decision["baseline_score"]:.4f} -> {decision["selected_score"]:.4f}',flush=True)
        save_selected_bank(selected, root)
        value = decisions[-1]['selected_score']
        record = dict(factor=factor, macro_f1=value, enabled_count=len(selected), decisions=decisions,
                      selected_root=str(root))
        atomic_write_json(record, root/'bank_complete.json')
        factors.append(record); all_decisions.extend(decisions)
        key = (value, -len(selected), factor)
        if key > best_key: best, best_key = selected, key
    save_selected_bank(best, output)
    result = dict(policy=policy, selected_factor=best_key[2], macro_f1=best_key[0],
        factor_banks=factors, tie_rule='fewer_active_sites_then_larger_factor',
        wb_fit_seconds=sum(d['wb_fit_seconds'] for d in all_decisions),
        candidate_validation_seconds=sum(d['candidate_validation_seconds'] for d in all_decisions))
    atomic_write_json(result, output/'factor_selection.json')
    return best, result
