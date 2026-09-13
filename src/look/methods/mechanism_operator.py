"""Frozen-host supplementary operators; no task labels enter their fitting."""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from look.methods.operator import (LOOKArtifact, apply_artifact, downsample_flatten,
    iter_feature_pairs, LatentSufficientStatistics, _gcv_lambda)
from look.runtime.host_checkpoint import atomic_save, capture_rng, restore_rng
from look.runtime.state import stable_hash
from look.data.mechanism_samples import ordered_ids, target_permutation
from look.studies.mechanism_protocol import MLP


class FitPaused(Exception): pass


@dataclass
class MechanismArtifact:
    base: LOOKArtifact
    writeback: str = 'both'
    mlp_state: dict | None = None
    hidden: int | None = None
    direct_affine: bool = False
    def __getattr__(self, name): return getattr(self.base,name)
    def apply_feature(self, feature):
        a=self.base
        if self.mlp_state is None and not self.direct_affine:
            residual=apply_artifact(feature,a)-feature
        else:
            flat,_=downsample_flatten(feature,a.factor)
            q=a.components.to(feature.device);std=a.std.to(feature.device)
            z=((flat-a.mean.to(feature.device))/std-a.pca_mean.to(feature.device))@q.T
            if self.direct_affine:
                # Avoid cancellation when the direct map is close to identity.
                direct=a.weight.to(device=feature.device,dtype=torch.float64)+torch.eye(a.latent_dim,device=feature.device,dtype=torch.float64)
                dz=(z.double()@direct+a.bias.to(feature.device).double()-z.double()).to(z.dtype)
            else:
                s={k:v.to(feature.device) for k,v in self.mlp_state.items()}
                dz=F.linear(F.relu(F.linear(z,s['0.weight'],s['0.bias'])),s['2.weight'],s['2.bias'])
            residual=((dz@q)*std).reshape(len(feature),*a.downsample_shape)
            if feature.ndim==4 and residual.shape[-2:]!=feature.shape[-2:]:
                residual=F.interpolate(residual,size=feature.shape[-2:],mode='bilinear',align_corners=False)
        if self.writeback!='both' and len(a.member_names)==2:
            missing='oct' if a.missing_pattern=='oct_missing' else 'cfp'
            chosen=missing if self.writeback=='missing' else ('cfp' if missing=='oct' else 'oct')
            start=0; mask=torch.zeros_like(residual)
            for name,shape in zip(a.member_names,a.member_shapes):
                width=shape[0]
                if name.startswith(chosen+'_'):mask[:,start:start+width]=residual[:,start:start+width]
                start+=width
            residual=mask
        return feature+residual
    def record(self):
        return dict(base=asdict(self.base),writeback=self.writeback,mlp_state=self.mlp_state,
                    hidden=self.hidden,direct_affine=self.direct_affine)
    @classmethod
    def from_record(cls,r): return cls(**dict(r,base=LOOKArtifact(**r['base'])))


def hidden_width(d):
    target=d*d+d
    # Two affine layers: 2*d*h+h+d. Global integer minimizer.
    center=(target-d)/(2*d+1)
    return min({max(1,int(center)),max(1,int(center)+1)},key=lambda h:(abs(2*d*h+h+d-target),h))


def latent_pairs(graph,loader,artifact,upstream,device):
    a=artifact; fulls=[];misses=[]
    for full,missing,_,_ in iter_feature_pairs(graph,loader,a.node_name,a.missing_pattern,a.factor,device,upstream):
        fulls.append((((full-a.mean)/a.std)-a.pca_mean)@a.components.T)
        misses.append((((missing-a.mean)/a.std)-a.pca_mean)@a.components.T)
    # Eye-level nodes and the participant pool have different sample ownership.
    ids=list(map(str,loader.dataset.participant_ids))
    dataset=loader.dataset
    if hasattr(dataset,'counts'):counts=list(dataset.counts)
    else:counts=[len(dataset[i]['cfp']) for i in range(len(dataset))]
    x=torch.cat(misses);y=torch.cat(fulls)
    if len(x)==len(ids): counts=[1]*len(ids)
    if len(x)!=sum(counts):raise ValueError('Feature ownership mismatch')
    return x,y,ids,counts


def ridge(x,full,a, penalty=None):
    """Residual ridge; penalty can match a restricted physical writeback metric."""
    stats=LatentSufficientStatistics(x.shape[1]);stats.update(x,full)
    cxx,cxy,tss,mx,my,n=stats.centered(x.shape[1])
    lam=_gcv_lambda(cxx,cxy,tss,n,x.shape[1]) if penalty is None else float(a.ridge_lambda)
    if penalty is None:
        w=torch.linalg.solve(cxx+lam*torch.eye(len(mx),dtype=cxx.dtype),cxy)
    else:
        # min ||(XW-Y)D||_F^2 + lambda||W||_F^2; D is the
        # PCA reconstruction, scaling and destination mask. Shared lambda is
        # fixed from LOOK; no development search for the restricted arm.
        ex,u=torch.linalg.eigh(cxx);eg,v=torch.linalg.eigh(penalty.double())
        eg=eg.clamp_min(0);ex=ex.clamp_min(0)
        right=u.T@cxy@v
        w=u@((right*eg)/(ex[:,None]*eg[None,:]+lam))@v.T
    b=my-mx@w
    return replace(a,weight=w.float(),bias=b.float(),ridge_lambda=lam)


def writeback_gram(a,mode):
    if len(a.member_names)!=2: return None
    missing='oct' if a.missing_pattern=='oct_missing' else 'cfp'
    chosen=missing if mode=='missing' else ('cfp' if missing=='oct' else 'oct')
    mask=torch.zeros(a.components.shape[1]);offset=0
    spatial=int(np.prod(a.downsample_shape[1:])) if len(a.downsample_shape)>1 else 1
    for name,shape in zip(a.member_names,a.member_shapes):
        length=shape[0]*spatial
        if name.startswith(chosen+'_'):mask[offset:offset+length]=1
        offset+=length
    d=a.components*(a.std*mask)[None,:]
    return d.double()@d.double().T


def fit_mlp(x,y,ids,counts,seed,path,should_pause=lambda:False):
    """Participant-held-out epoch choice, then exact-epoch full-data refit.

    Saves optimizer, RNG, phase, epoch and offset at a pause. Validation is only
    reconstruction MSE on train-held-out people, never disease labels or dev.
    """
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    outer=capture_rng()
    try:
        torch.manual_seed(seed)
        d=x.shape[1];h=hidden_width(d)
        def create():
            m=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,d))
            nn.init.zeros_(m[2].weight);nn.init.zeros_(m[2].bias)
            return m
        model=create();initial={k:v.clone() for k,v in model.state_dict().items()}
        opt=torch.optim.AdamW(model.parameters(),lr=MLP['lr'],weight_decay=MLP['weight_decay'])
        held=set(ordered_ids(ids,seed,'look_mlp_holdout_v1')[max(1,int(len(ids)*.9)):])
        owner=np.repeat(np.asarray(ids),counts)
        val=torch.tensor(np.flatnonzero(np.isin(owner,list(held))),dtype=torch.long)
        fit=torch.tensor(np.flatnonzero(~np.isin(owner,list(held))),dtype=torch.long)
        if not len(val) or not len(fit):raise ValueError('Insufficient participant-held-out MLP data')
        identity=stable_hash(dict(ids=ids,counts=counts,seed=seed,shape=list(x.shape),config=MLP))
        state=dict(phase='select',epoch=1,offset=0,best=float('inf'),best_epoch=0,stall=0)
        if path.exists():
            r=torch.load(path,map_location='cpu',weights_only=False)
            if r['identity']!=identity:raise ValueError('MLP resume identity mismatch')
            state=r['progress'];model.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);restore_rng(r['rng'])
        def checkpoint():
            atomic_save(path,dict(identity=identity,progress=state,model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng()))
        target=y-x
        while True:
            indices=fit if state['phase']=='select' else torch.arange(len(x))
            order=indices[torch.randperm(len(indices),generator=torch.Generator().manual_seed(seed+state['epoch']))]
            while state['offset']<len(order):
                index=order[state['offset']:state['offset']+MLP['batch']]
                opt.zero_grad(set_to_none=True);loss=F.mse_loss(model(x[index]),target[index])
                if not torch.isfinite(loss):raise ValueError('Nonfinite MLP fitting loss')
                loss.backward();opt.step();opt.zero_grad(set_to_none=True);state['offset']+=len(index)
                if should_pause():checkpoint();raise FitPaused()
            if state['phase']=='select':
                with torch.no_grad():
                    error=sum(float(F.mse_loss(model(x[i]),target[i],reduction='sum')) for i in val.split(MLP['batch']))/(len(val)*d)
                if error<state['best']:
                    substantial=state['best']==float('inf') or (state['best']-error)/max(state['best'],1e-30)>=MLP['relative_improvement']
                    state['best']=error;state['best_epoch']=state['epoch']
                else:substantial=False
                state['stall']=0 if substantial else state['stall']+1
                if state['stall']>=MLP['patience'] or state['epoch']>=MLP['epochs']:
                    model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=MLP['lr'],weight_decay=MLP['weight_decay'])
                    state.update(phase='refit',epoch=1,offset=0);checkpoint();continue
            elif state['epoch']>=state['best_epoch']:
                return {k:v.detach().clone() for k,v in model.state_dict().items()},h,dict(
                    selected_epochs=state['best_epoch'],heldout_participants=len(held),
                    heldout_sha256=stable_hash(sorted(held)),stored_parameters=sum(p.numel() for p in model.parameters()))
            state.update(epoch=state['epoch']+1,offset=0);checkpoint()
    finally:restore_rng(outer)


def fit_variant(graph,loader,templates,arm,device,seed,output,should_pause=lambda:False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);selected=[];records=[]
    for index,a in enumerate(templates):
        p=out/f'{index:02d}.pt'
        if p.exists():
            r=torch.load(p,map_location='cpu',weights_only=False)
            selected.append(MechanismArtifact.from_record(r['artifact']));records.append(r['diagnostics']);continue
        if should_pause(): raise FitPaused()
        mode='missing' if arm.startswith('missing_') else 'available' if arm.startswith('available_') else 'both'
        diagnostics=dict(node=a.node_name,arm=arm,branch_specific=len(a.member_names)==2,
                         interpretation='fixed_selected_LOOK_configuration')
        artifact=MechanismArtifact(a,writeback=mode,direct_affine=arm=='affine_equivalence')
        if arm not in ('missing_readout','available_readout','affine_equivalence'):
            upstream=[] if arm=='independent' else selected
            x,y,ids,counts=latent_pairs(graph,loader,a,upstream,device)
            if arm=='shuffle':
                permutation=target_permutation(ids,counts,seed,getattr(loader.dataset,'counts',None));y=y[permutation]
                diagnostics.update(permutation_sha256=stable_hash(permutation.tolist()),stratification='equal_observed_eye_count')
            if arm=='mlp':
                state,h,info=fit_mlp(x,y,ids,counts,seed,out/f'{index:02d}_resume.pt',should_pause)
                artifact.mlp_state=state;artifact.hidden=h;diagnostics.update(info)
            else:
                gram=writeback_gram(a,mode) if arm.endswith('_refit') else None
                artifact.base=ridge(x,y,a,gram)
        atomic_save(p,dict(artifact=artifact.record(),diagnostics=diagnostics))
        selected.append(artifact);records.append(diagnostics)
    return selected,records
