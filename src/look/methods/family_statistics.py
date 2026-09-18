"""Full-dimensional, train-only moments shared by every site of one prefix.

No participant feature cache is written: only centered sufficient statistics and
batch identity hashes. A new prefix needs a new scan; complete statistics need
neither loader iteration nor network execution. Partial resume reads/validates
committed batches without forwarding them.
"""
from dataclasses import asdict
import fcntl
from pathlib import Path
import torch
from look.methods import operator as op
from look.methods.joint import site_level, read_site, write_site, members, member_shapes, PROTOCOL
from look.methods.linear_operator import fingerprint
from look.methods.linear_vector import ResidualMoments, estimated_workspace_bytes
from look.methods.independent_greedy import SelectionPaused
from look.runtime.host_checkpoint import atomic_save


@torch.no_grad()
def capture_sites(graph, batch, sites, bases, device, pattern='complete', upstream=()):
    if graph.training or any(p.requires_grad for p in graph.parameters()):
        raise ValueError('Frozen eval graph required')
    by_level = {}
    for name in sites:
        by_level.setdefault(site_level(graph, name), []).append(name)
    by_node = {a.node_name: a for a in upstream}
    if len(by_node) != len(upstream):
        raise ValueError('Duplicate upstream correction')
    x, y = op._prepare_inputs(batch, device, pattern, op.NormalizedMeanFiller())
    op._reset_inputs(graph, x, y, batch.get('counts'))
    result = {}
    for level in range(-1, max(by_level) + 1):
        if level >= 0:
            graph.forward(levels=[level])
        for name, a in by_node.items():
            if site_level(graph, name) == level:
                if a.protocol != PROTOCOL or a.split_rule != 'channel_split_and_restore_member_shapes_v1':
                    raise ValueError('Upstream protocol changed')
                if a.member_names and tuple(a.member_names) != members(name):
                    raise ValueError('Upstream members changed')
                if a.member_shapes and tuple(map(tuple, a.member_shapes)) != member_shapes(graph, name):
                    raise ValueError('Upstream shapes changed')
                write_site(graph, name, op.apply_artifact(read_site(graph, name), a))
        for name in by_level.get(level, ()):
            b = bases[name]
            feature = read_site(graph, name).detach()
            flat, shape = op.downsample_flatten(feature, b.factor, b.spatial_method)
            if tuple(feature.shape[1:]) != tuple(b.feature_shape) or tuple(shape) != tuple(b.downsample_shape):
                raise ValueError('Feature shape changed')
            # Clone protects against graph reset/in-place downstream operations on CPU.
            result[name] = flat.cpu().clone()
    return result


class FamilyStatistics:
    def __init__(self, graph, loader, bases, device, root, identity, workspace_bytes,
                 rank, should_pause=lambda: False, projected_ranks=()):
        if loader.dataset.split != 'train' or loader.dataset.augment or loader.drop_last:
            raise ValueError('Complete unaugmented train only')
        if not identity:
            raise ValueError('Immutable identity required')
        self.graph, self.loader, self.bases, self.device = graph, loader, bases, device
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.identity = fingerprint(dict(version='family_full_moments_v1', source=identity,
                                         bases={n:asdict(b) for n,b in bases.items()}))
        self.budget, self.rank, self.check = workspace_bytes, rank, should_pause
        self.projected_ranks = tuple(projected_ranks)
        self.projected = {}
        self.metrics = dict(full_forwards=0, missing_forwards=0, completed_hits=0, skipped_batches=0)

    def statistics(self, pattern, upstream, sites):
        if pattern not in ('oct_missing', 'cfp_missing') or not sites or len(set(sites)) != len(sites):
            raise ValueError('Invalid sites or missing pattern')
        # Conservative aggregate bound, not the old unsafe per-site admission.
        dims = [self.bases[n].std.numel() for n in sites]
        resident = sum(8*(2*d*d + 2*d + 1) for d in dims)
        projected_resident = len(sites)*sum(8*(2*q*q + 2*q + 1) for q in self.projected_ranks)
        # Keep other sites resident while one solve uses its admitted workspace.
        required = resident + projected_resident + max(estimated_workspace_bytes(d, self.rank) for d in dims)
        if required > self.budget:
            raise MemoryError('Shared dense fitting exceeds admitted workspace; no silent compression')
        for n in sites:
            b = self.bases[n]
            if not torch.isfinite(b.mean).all() or not torch.isfinite(b.std).all() or not (b.std > 0).all():
                raise ValueError('Finite train normalization and positive scales required')
        sid = fingerprint(dict(reference=self.identity, pattern=pattern, sites=list(sites),
                               upstream=[a.record() for a in upstream], projected_ranks=self.projected_ranks))
        path = self.root / (sid + '.pt')
        with (self.root / (sid + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._collect(path, sid, pattern, upstream, sites)

    def _collect(self, path, sid, pattern, upstream, sites):
        stats = None
        projected = {n:{q:ResidualMoments.empty(q) for q in self.projected_ranks} for n in sites}
        for n in sites:
            if any(q > len(self.bases[n].components) for q in self.projected_ranks):
                raise ValueError('GCV reference rank infeasible')
        self.projected = projected
        stamps = []
        if path.exists():
            record = torch.load(path, map_location='cpu', weights_only=False)
            saved = record['payload']
            if fingerprint(saved) != record['sha256'] or saved['identity'] != sid or saved['sites'] != list(sites):
                raise ValueError('Changed statistics identity or corrupt statistics')
            stamps = saved['batch_stamps']
            stats = {n:ResidualMoments(**saved['statistics'][n]) for n in sites}
            projected = {n:{q:ResidualMoments(**v) for q,v in saved['projected'][n].items()} for n in sites}
            self.projected = projected
            if saved['complete']:
                self.metrics['completed_hits'] += 1
                return stats
            del record, saved
        else:
            stats = {n:ResidualMoments.empty(self.bases[n].std.numel()) for n in sites}
        cursor = len(stamps)
        def save(complete=False):
            payload = dict(identity=sid, sites=list(sites), batch_stamps=stamps,
                           # Synchronous save under the collection lock: tensor
                           # references cannot mutate until atomic_save returns.
                           # asdict would deep-copy every dense statistic.
                           complete=complete, statistics={n:vars(s).copy() for n,s in stats.items()},
                           projected={n:{q:vars(v).copy() for q,v in qs.items()} for n,qs in projected.items()})
            atomic_save(path, dict(payload=payload, sha256=fingerprint(payload)))
        count = 0
        for index, batch in enumerate(self.loader):
            count = index + 1
            if 'participant_id' not in batch:
                raise ValueError('Participant order required for resumable statistics')
            stamp = fingerprint(dict(ids=batch['participant_id'], labels=batch['label'],
                counts=batch.get('counts'), oct_shape=list(batch['oct'].shape), cfp_shape=list(batch['cfp'].shape)))
            if index < cursor:
                if stamp != stamps[index]:
                    raise ValueError('Committed participant data/order changed')
                self.metrics['skipped_batches'] += 1
                continue
            if self.check():
                save(); raise SelectionPaused()
            full = capture_sites(self.graph, batch, sites, self.bases, self.device)
            missing = capture_sites(self.graph, batch, sites, self.bases, self.device, pattern, upstream)
            self.metrics['full_forwards'] += 1; self.metrics['missing_forwards'] += 1
            for n in sites:
                b = self.bases[n]
                x = (missing[n]-b.mean)/b.std
                y = (full[n]-missing[n])/b.std
                stats[n].update(x, y)
                for q, ps in projected[n].items():
                    basis = b.components[:q].to(dtype=torch.float64, device='cpu')
                    ps.update(x.double() @ basis.T, y.double() @ basis.T)
            stamps.append(stamp)
            # Commit only after every site consumed the same batch.
            if count % 32 == 0:
                save()
        if count < cursor:
            raise ValueError('Fitting stream shortened')
        save(True)
        return stats
