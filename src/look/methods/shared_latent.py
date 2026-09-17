"""Frozen multi-site extraction, exact minibatch moments and durable train caches.

Only the computation schedule changes. Projection remains on CPU in the original
dtype/order. Each missing stream is keyed by the entire accepted upstream bank.
No dev scores or test inputs participate in a fitting cache.
"""
from dataclasses import asdict
import fcntl
from pathlib import Path
import time
import torch

from look.methods import operator as op
from look.methods.joint import site_level, read_site, write_site, members, member_shapes, PROTOCOL
from look.methods.linear_operator import fingerprint
from look.methods.independent_greedy import SelectionPaused
from look.runtime.host_checkpoint import atomic_save


VERSION = 'look_shared_latent_v1'


@torch.no_grad()
def project_sites(graph, batch, sites, bases, max_rank, device, pattern='complete', upstream=()):
    """Capture each site immediately at its level, before descendants can overwrite it."""
    if graph.training or any(p.requires_grad for p in graph.parameters()):
        raise ValueError('Frozen eval graph required')
    if not sites or len(set(sites)) != len(sites):
        raise ValueError('Unique nonempty sites required')
    by_level = {}
    for name in sites:
        by_level.setdefault(site_level(graph, name), []).append(name)
    by_node = {a.node_name: a for a in upstream}
    if len(by_node) != len(upstream):
        raise ValueError('Duplicate upstream correction')
    oct_x, cfp_x = op._prepare_inputs(batch, device, pattern, op.NormalizedMeanFiller())
    op._reset_inputs(graph, oct_x, cfp_x, batch.get('counts'))
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
                if a.member_shapes and tuple(map(tuple,a.member_shapes)) != member_shapes(graph,name):
                    raise ValueError('Upstream shapes changed')
                write_site(graph,name,op.apply_artifact(read_site(graph,name),a))
        for name in by_level.get(level, ()):
            b = bases[name]
            x = read_site(graph,name).detach()
            flat, shape = op.downsample_flatten(x,b.factor,b.spatial_method)
            if tuple(x.shape[1:]) != tuple(b.feature_shape) or tuple(shape) != tuple(b.downsample_shape):
                raise ValueError('Basis shape changed')
            # Same CPU projection as iter_feature_pairs -> fit_look_node.
            flat = flat.cpu()
            result[name] = (((flat-b.mean)/b.std)-b.pca_mean) @ b.components[:max_rank].T
    return result


def _save(path, payload):
    atomic_save(path,dict(payload=payload,sha256=fingerprint(payload)))


def _read(path):
    record = torch.load(path,map_location='cpu',weights_only=False)
    if fingerprint(record['payload']) != record['sha256']:
        raise ValueError('Corrupt latent cache')
    return record['payload']


class SharedLatentFitter:
    def __init__(self, graph, loader, bases, max_rank, device, root, reference_root,
                 identity, should_pause=lambda:False):
        ds = loader.dataset
        if ds.split != 'train' or ds.augment or loader.drop_last:
            raise ValueError('Complete unaugmented train only')
        if not identity:
            raise ValueError('Explicit frozen data identity required')
        self.graph,self.loader,self.bases,self.rank,self.device = graph,loader,bases,max_rank,device
        self.root,self.reference_root = Path(root),Path(reference_root)
        self.check = should_pause
        self.identity = fingerprint(dict(version=VERSION,source=identity,
            bases={n:asdict(b) for n,b in bases.items()},max_rank=max_rank))
        self.references = self.reference_root/self.identity
        self.references.mkdir(parents=True,exist_ok=True)
        self.root.mkdir(parents=True,exist_ok=True)
        self.metrics = dict(full_forwards=0,missing_forwards=0,reference_hits=0,
            reference_seconds=0.,missing_seconds=0.,statistics_seconds=0.,loader_seconds=0.)

    def statistics(self, pattern, upstream, sites):
        if pattern not in ('oct_missing','cfp_missing') or not sites or len(set(sites)) != len(sites):
            raise ValueError('Invalid missing stream')
        state_id = fingerprint(dict(reference=self.identity,pattern=pattern,
            upstream=[asdict(a) for a in upstream],sites=list(sites)))
        path = self.root/(state_id+'.pt')
        with (self.root/(state_id+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return self._statistics(path,state_id,pattern,upstream,sites)

    def _statistics(self,path,state_id,pattern,upstream,sites):
        stats={n:op.LatentSufficientStatistics(min(self.rank,len(self.bases[n].components))) for n in sites}
        cursor=0
        if path.exists():
            saved=_read(path)
            if saved['identity']!=state_id or saved['sites']!=list(sites):
                raise ValueError('Changed statistics identity')
            cursor=saved['batches']
            if type(cursor) is not int or cursor<0:
                raise ValueError('Invalid statistics cursor')
            for n in sites:
                stats[n].__dict__.update(saved['statistics'][n])
            if saved['complete']:
                return stats
        def save(complete=False):
            _save(path,dict(identity=state_id,sites=list(sites),batches=cursor,complete=complete,
                statistics={n:s.__dict__ for n,s in stats.items()},metrics=dict(self.metrics)))
        iterator=iter(self.loader);index=0
        try:
            while True:
                if self.check():
                    save();raise SelectionPaused()
                t=time.perf_counter()
                try: batch=next(iterator)
                except StopIteration: break
                self.metrics['loader_seconds']+=time.perf_counter()-t
                stamp=fingerprint(dict(ids=batch.get('participant_id'),labels=batch['label'],
                    counts=batch.get('counts'),oct_shape=list(batch['oct'].shape),cfp_shape=list(batch['cfp'].shape)))
                ref=self.references/f'{index:08d}.pt'
                if index<cursor:
                    committed=_read(ref)
                    if committed['identity']!=self.identity or committed['batch']!=stamp:
                        raise ValueError('Committed reference data/order changed')
                    index+=1;continue
                # Shared across missing patterns and strategies; coordinate writers.
                with (self.references/'writer.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    t=time.perf_counter()
                    if ref.exists():
                        full=_read(ref)
                        if full['identity']!=self.identity or full['batch']!=stamp:
                            raise ValueError('Reference data/order changed')
                        self.metrics['reference_hits']+=1
                    else:
                        z=project_sites(self.graph,batch,list(self.bases),self.bases,self.rank,self.device)
                        full=dict(identity=self.identity,batch=stamp,values=z)
                        _save(ref,full);self.metrics['full_forwards']+=1
                    self.metrics['reference_seconds']+=time.perf_counter()-t
                t=time.perf_counter()
                missing=project_sites(self.graph,batch,sites,self.bases,self.rank,self.device,pattern,upstream)
                self.metrics['missing_forwards']+=1
                self.metrics['missing_seconds']+=time.perf_counter()-t
                t=time.perf_counter()
                for n in sites:
                    stats[n].update(missing[n],full['values'][n])
                self.metrics['statistics_seconds']+=time.perf_counter()-t
                cursor=index+1;index+=1
                if cursor%32==0:save()
        except BaseException:
            # Keep the last committed boundary; a failed batch may have partly updated moments.
            raise
        if index<cursor:
            raise ValueError('Fitting stream shortened')
        save(True)
        return stats
