"""Full-MHD adapter for independently fitted affine-family trajectories.

Unlike fit_bank, this consumes all eligible nodes and a declared candidate table,
never the nodes, ranks or gates selected by another method. No defaults silently
define a research search space; callers must lock it in their protocol.
"""
from dataclasses import asdict
import json
import hashlib
import math
from pathlib import Path
import torch

from look.methods.affine_family import ARMS, FamilyArtifact, fit_map
from look.methods.independent_greedy import fit_trajectory, SelectionPaused
from look.methods.linear_operator import fingerprint
from look.methods.operator import LOOKArtifact, _gcv_lambda
from look.runtime.host_checkpoint import atomic_save
from look.runtime.state import atomic_write_json
from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle


RANKED = ('shared_pca_ridge', 'rrr_shared_intercept', 'residual_rrr',
          'pca_free_mean', 'pls_svd_ridge')
BASIS_CONSTRAINED = ('shared_pca_ridge', 'rrr_shared_intercept', 'pca_free_mean')


def validate_candidates(arm, candidates, penalty_policy='fixed'):
    if penalty_policy not in ('fixed', 'prefix_train_pca_gcv'):
        raise ValueError('Unknown penalty policy')
    if penalty_policy == 'prefix_train_pca_gcv':
        if arm not in RANKED or any(c.get('ridge_lambda') is not None for c in candidates):
            raise ValueError('Train PCA GCV requires ranked candidates with null ridge_lambda')
        candidates = [dict(c, ridge_lambda=1.) for c in candidates]
    if arm not in ARMS or not candidates:
        raise ValueError('Registered arm and explicit nonempty candidate table required')
    keys = []
    for c in candidates:
        if set(c) != {'rank', 'ridge_lambda'}:
            raise ValueError('Candidate fields differ from declared contract')
        rank, penalty = c['rank'], c['ridge_lambda']
        if arm in RANKED:
            if type(rank) is not int or rank < 1:
                raise ValueError('Ranked methods require a positive integer rank')
        elif rank is not None:
            raise ValueError('Unranked methods must declare rank not applicable')
        if arm == 'orthogonal_alignment':
            if penalty is not None:
                raise ValueError('Orthogonal alignment has no ridge penalty')
        elif type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty <= 0:
            raise ValueError('Positive finite penalty required')
        keys.append(json.dumps(c, sort_keys=True))
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate candidate configurations')


def fit_family_trajectory(graph, train_loader, dev_loader, *, arm, pattern, sites,
                          factor, candidates, pca_bank, identity, output, device,
                          workspace_bytes, mode='best_forward', should_pause=lambda:False,
                          penalty_policy='fixed'):
    validate_candidates(arm, candidates, penalty_policy)
    if (getattr(train_loader.dataset, 'split', None) != 'train'
        or getattr(train_loader.dataset, 'augment', None) is not False
        or train_loader.drop_last):
        raise ValueError('All unaugmented train samples required')
    if getattr(dev_loader.dataset, 'split', None) not in ('development', 'train_probe'):
        raise ValueError('Development or explicitly marked train-only probe required')
    if graph.training or any(p.requires_grad for p in graph.parameters()):
        raise ValueError('Frozen eval-mode graph required')
    if pattern not in ('oct_missing', 'cfp_missing') or type(factor) is not int or factor < 1:
        raise ValueError('Invalid pattern or spatial factor')
    if not identity:
        raise ValueError('Immutable scientific identity required')
    root=Path(output); root.mkdir(parents=True,exist_ok=True)
    bases={}
    for site in sites:
        matches=[b for (n,f),b in pca_bank.items() if n==site and
                 f==(1 if len(b.feature_shape)==1 else factor)]
        if len(matches)!=1:
            raise ValueError('Every eligible node requires a unique frozen basis')
        bases[site]=matches[0]
    scientific_identity=dict(identity=identity, arm=arm, pattern=pattern,
        factor=factor, candidates=candidates, penalty_policy=penalty_policy, data_role=dev_loader.dataset.split,
        bases={k:fingerprint(asdict(v)) for k,v in bases.items()})

    from look.methods.family_statistics import FamilyStatistics
    collector = FamilyStatistics(graph, train_loader, bases, device, root/'family_moments',
        scientific_identity, workspace_bytes, max(c['rank'] or 1 for c in candidates), should_pause,
        projected_ranks=sorted({c['rank'] for c in candidates}) if penalty_policy == 'prefix_train_pca_gcv' else ())
    ready = {}

    def fit(site, upstream, folder):
        b=bases[site]; d=b.std.numel(); folder.mkdir(parents=True,exist_ok=True)
        sid=fingerprint(dict(identity=scientific_identity,site=site,
                             upstream=[a.record() for a in upstream]))
        prefix = fingerprint([a.record() for a in upstream])
        multi = mode in ('best_forward', 'positive_forward_tree')
        key = (prefix, None if multi else site)
        if key not in ready:
            start = max((sites.index(a.node_name) for a in upstream), default=-1)+1
            targets = list(sites[start:]) if multi else [site]
            ready.clear()
            ready[key] = collector.statistics(pattern, upstream, targets)
            atomic_write_json(collector.metrics, root/'feature_costs.json')
        stats = ready[key][site]
        if should_pause():raise SelectionPaused()
        for c in candidates:
            # None stays visible in candidate provenance; dummy values below only
            # satisfy fit_map's legacy signature for mathematically unused fields.
            q=c['rank'] or 1; lam=c['ridge_lambda'] or 1.0
            if q>min(d,stats.count-1) or (arm in BASIS_CONSTRAINED and q>len(b.components)):
                raise ValueError('Locked rank infeasible; no clipping or silent skipping')
            basis=b.components[:q]
            if penalty_policy == 'prefix_train_pca_gcv':
                projected = collector.projected[site][q]
                q_basis = basis.to(dtype=torch.float64, device='cpu')
                lam = _gcv_lambda(q_basis @ stats.cxx @ q_basis.T,
                    q_basis @ stats.cxy @ q_basis.T, float(projected.syy), stats.count, q)
            a=LOOKArtifact(site,pattern,'normalized_mean',b.factor,q,b.feature_shape,
                b.downsample_shape,b.mean,b.std,b.pca_mean,basis,
                torch.zeros(q,q),torch.zeros(q),lam,0.,0.,
                float(b.explained_variance_ratio[:q].sum()),b.fit_seconds,b.peak_rss_bytes,
                b.source_id,b.member_names,b.member_shapes,b.protocol,b.split_rule,b.spatial_method)
            mapping=fit_map(stats,q,lam,arm=arm,basis=basis)
            mapping.diagnostics.update(selection='independent_per_node',declared_candidate=c,penalty_policy=penalty_policy,
                upstream_identity=sid, reference_configuration_inherited=False,
                note='Independently selected within the declared candidate table; not a conditional PCA-selected control.')
            yield json.dumps(c,sort_keys=True),FamilyArtifact(a,mapping)

    def evaluate(bank):
        if should_pause():raise SelectionPaused()
        r=evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern,
                           artifact_banks={pattern:bank})
        import numpy as np
        values=hashlib.sha256()
        for name in ('participant_ids','labels','logits'):
            arr=np.ascontiguousarray(r[name])
            values.update(name.encode());values.update(str(arr.dtype).encode())
            values.update(str(arr.shape).encode());values.update(arr.tobytes())
        values_sha=values.hexdigest()
        key=fingerprint([a.record() for a in bank])
        p=root/'predictions'/f'{key}_{values_sha}.npz'
        if p.exists():
            with np.load(p,allow_pickle=False) as saved:
                if any(not np.array_equal(saved[k],r[k]) for k in ('participant_ids','labels','logits')):
                    raise ValueError('Stored prediction evidence changed')
        else:save_prediction_bundle(r,p)
        from look.runtime.state import file_sha256
        return dict(role='development', data_role=dev_loader.dataset.split,
                    score=float(r['metrics']['macro_f1']),
                    prediction=str(p),sha256=file_sha256(p),values_sha256=values_sha,metrics=r['metrics'])
    def save(a,p):atomic_save(p,a.record())
    def load(p):return FamilyArtifact.from_record(torch.load(p,map_location='cpu',weights_only=False))
    bank,result=fit_trajectory(identity=scientific_identity,sites=sites,mode=mode,output=root,
        fit_candidates=fit,evaluate=evaluate,save_artifact=save,load_artifact=load,
        should_pause=should_pause)
    # Full-MHD replay after serialization, independently of the selection cache.
    records=[a.record() for a in bank]
    atomic_save(root/'bank.pt',dict(identity=scientific_identity,bank=records,sha256=fingerprint(records)))
    replay=evaluate([FamilyArtifact.from_record(r) for r in records])
    if replay['values_sha256']!=result['final']['values_sha256']:
        raise ValueError('Reloaded MHD predictions changed')
    atomic_write_json(dict(full_mhd_replay=True,scientific_acceptance=False,
                          test_access=False),root/'replay.json')
    return bank,result
