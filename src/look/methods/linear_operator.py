"""Versioned MHD writeback and resumable train-only linear-vector fitting."""
from dataclasses import asdict, dataclass
import hashlib
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from look.methods.linear_vector import ResidualMoments, AffineResidual, solve, estimated_workspace_bytes
from look.methods.operator import LOOKArtifact, downsample_flatten, iter_feature_pairs
from look.runtime.host_checkpoint import atomic_save
from look.runtime.state import stable_hash

VERSION = 'look_linear_vector_v1'
ARMS = ('shared_pca_ridge', 'rrr_shared_intercept', 'residual_rrr')


class LinearFitPaused(Exception):
    pass


def fingerprint(value):
    def encode(x):
        if isinstance(x, torch.Tensor):
            y=x.detach().cpu().contiguous()
            # Hash the contiguous CPU buffer directly; tobytes duplicates large
            # covariance matrices even though hashing is synchronous/read-only.
            return dict(shape=list(y.shape),dtype=str(y.dtype),sha256=hashlib.sha256(memoryview(y.numpy())).hexdigest())
        if isinstance(x, dict):return {k:encode(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [encode(v) for v in x]
        return x
    return stable_hash(encode(value))


@dataclass
class LinearVectorArtifact:
    base: LOOKArtifact
    mapping: AffineResidual

    @property
    def node_name(self): return self.base.node_name

    @property
    def protocol(self): return self.base.protocol

    @property
    def split_rule(self): return self.base.split_rule

    @property
    def member_names(self): return self.base.member_names

    @property
    def member_shapes(self): return self.base.member_shapes

    def apply_feature(self, feature):
        a=self.base
        if tuple(feature.shape[1:]) != tuple(a.feature_shape):
            raise ValueError('Original feature shape changed')
        flat,shape=downsample_flatten(feature,a.factor,a.spatial_method)
        if tuple(shape)!=tuple(a.downsample_shape):raise ValueError('Reduced feature shape changed')
        std=a.std.to(device=feature.device,dtype=feature.dtype)
        mean=a.mean.to(device=feature.device,dtype=feature.dtype)
        residual=self.mapping.predict((flat-mean)/std)*std
        residual=residual.reshape(len(feature),*shape)
        if feature.ndim==4 and residual.shape[-2:]!=feature.shape[-2:]:
            residual=F.interpolate(residual,size=feature.shape[-2:],mode='bilinear',align_corners=False)
        return feature+residual

    def record(self):
        return dict(schema=VERSION,base=asdict(self.base),mapping=asdict(self.mapping))

    @classmethod
    def from_record(cls,record):
        if record['schema']!=VERSION:raise ValueError('Unknown linear artifact schema')
        return cls(LOOKArtifact(**record['base']),AffineResidual(**record['mapping']))


def fit_bank(graph, loader, templates, arm, device, output, *, identity, workspace_bytes,
             should_pause=lambda:False, family=False):
    """Fixed LOOK-selected sites/factors/ranks/penalties, sequential per arm.

    This is a conditional mechanism comparison, NOT independently optimized methods.
    It must not be called on development/test or augmented training data.
    """
    artifact_type=LinearVectorArtifact; version=VERSION; allowed=ARMS
    if family:
        from look.methods.affine_family import FamilyArtifact, ARMS as allowed, VERSION as version
        artifact_type=FamilyArtifact
    if arm not in allowed:raise ValueError('Unregistered linear arm')
    if getattr(loader.dataset,'split',None)!='train' or getattr(loader.dataset,'augment',None) is not False:
        raise ValueError('Only unaugmented training features may fit corrections')
    if loader.drop_last:raise ValueError('Cannot discard fitting participants')
    if graph.training or any(p.requires_grad for p in graph.parameters()):
        raise ValueError('Host must already be frozen and in eval mode')
    if not identity:raise ValueError('Immutable source/data identity required')
    out=Path(output);out.mkdir(parents=True,exist_ok=True);selected=[];diagnostics=[]
    for index,a in enumerate(templates):
        dim=a.std.numel();rank=a.latent_dim
        # Budget applies to additional dense fitting workspace, not total allocation.
        estimate=estimated_workspace_bytes(dim,rank)
        if estimate>workspace_bytes:raise MemoryError(f'Linear dense workspace {estimate} > {workspace_bytes}; no silent compression')
        node_id=fingerprint(dict(protocol=version,identity=identity,arm=arm,template=asdict(a),
                                 upstream=[x.record() for x in selected]))
        path=out/f'{index:03d}.pt';resume=out/f'{index:03d}_resume.pt'
        if path.exists():
            r=torch.load(path,map_location='cpu',weights_only=False)
            if r['identity']!=node_id or fingerprint(r['artifact'])!=r['artifact_sha256']:
                raise ValueError('Completed linear fit identity or artifact changed')
            fitted=artifact_type.from_record(r['artifact'])
            selected.append(fitted);diagnostics.append(fitted.mapping.diagnostics);continue
        stats=ResidualMoments.empty(dim);cursor=0
        if resume.exists():
            r=torch.load(resume,map_location='cpu',weights_only=False)
            if r['identity']!=node_id or fingerprint(r['statistics'])!=r['statistics_sha256']:
                raise ValueError('Linear fit resume changed')
            stats=ResidualMoments(**r['statistics']);cursor=r['batches']
        def checkpoint(batches):
            state=asdict(stats)
            atomic_save(resume,dict(identity=node_id,batches=batches,statistics=state,statistics_sha256=fingerprint(state)))
        batches=0;last_checkpoint=time.monotonic()
        for full,missing,_,_ in iter_feature_pairs(graph,loader,a.node_name,a.missing_pattern,a.factor,
                                                  device,selected,spatial_method=a.spatial_method):
            batches+=1
            if batches<=cursor:continue
            if should_pause():checkpoint(batches-1);raise LinearFitPaused()
            stats.update((missing-a.mean)/a.std,(full-missing)/a.std)
            if time.monotonic()-last_checkpoint>=300:
                checkpoint(batches);last_checkpoint=time.monotonic()
        if batches<cursor:raise ValueError('Resume cursor exceeds fitting stream')
        if should_pause():checkpoint(batches);raise LinearFitPaused()
        kwargs={'basis':a.components} if arm=='shared_pca_ridge' else (
            {'intercept_basis':a.components} if arm=='rrr_shared_intercept' else {})
        if family:
            from look.methods.affine_family import fit_map
            mapping=fit_map(stats,rank,a.ridge_lambda,arm=arm,basis=a.components)
        else:
            mapping=solve(stats,rank,a.ridge_lambda,**kwargs)
        mapping.diagnostics.update(reference_pca_explained_variance=a.pca_explained_variance,
            reference_pca_source=a.pca_source_id,
            fraction_comparison='PCA variance and regularized residual gain are different quantities')
        fitted=artifact_type(a,mapping);record=fitted.record()
        atomic_save(path,dict(identity=node_id,artifact=record,artifact_sha256=fingerprint(record)))
        selected.append(fitted);diagnostics.append(mapping.diagnostics)
    return selected,diagnostics
