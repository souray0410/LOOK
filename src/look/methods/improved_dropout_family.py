"""LOOK correction for the frozen improved-modality-dropout target-task adapter.

The missing modality is substituted by the learned author token *before* any
LOOK-visible site. This prevents a correction from reading the unavailable raw
modality. Two logical vector sites are exposed:

1. imd_fusion_input: normalized token-substituted modality features, concatenated.
2. fusion_feature: output of the author-style TNF MLP before the classifier.

The accepted A checkpoint is never mutated by fitting.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA

from look.evaluation.stability import logit_metrics, probabilities_from_logits
from look.methods.affine_family import FamilyArtifact, fit_map
from look.methods.independent_greedy import fit_trajectory, SelectionPaused
from look.methods.linear_operator import fingerprint
from look.methods.linear_vector import ResidualMoments, estimated_workspace_bytes
from look.methods.operator import FullFeaturePCA, LOOKArtifact, _gcv_lambda
from look.models.native_host import set_observed_counts
from look.runtime.host_checkpoint import atomic_save
from look.runtime.state import atomic_write_json, file_sha256

SITES=("imd_fusion_input","fusion_feature")
PATTERNS=("oct_missing","cfp_missing")
ARMS=("pca_free_mean","residual_rrr")
PROTOCOL="look_imd_logical_vector_v1"
SPLIT_RULE="token_substituted_logical_vector_v1"


def _fusion_model(graph):
    edge=graph.get_edge_by_name("improved_dropout_logits_edge")
    if edge is None or len(edge.edge_operations)!=1:
        raise ValueError("Improved-dropout fusion edge changed")
    operation=edge.edge_operations[0].function
    model=getattr(operation,"fusion",None)
    if model is None or not all(hasattr(model,name) for name in ("fusor","classifier_norm","classifier_dropout","classifier")):
        raise ValueError("Improved-dropout fusion operation changed")
    return model


def _participant_features(graph,batch,device):
    from look.methods.operator import _reset_inputs
    oct_tensor=batch["oct"].to(device)
    cfp_tensor=batch["cfp"].to(device)
    _reset_inputs(graph,oct_tensor,cfp_tensor,batch["counts"])
    target=max(
        graph.node_level_map["oct_participant_feature"],
        graph.node_level_map["cfp_participant_feature"],
    )
    graph.forward(levels=list(range(target+1)))
    return (
        graph.get_node_by_name("oct_participant_feature").feature_message.current_state,
        graph.get_node_by_name("cfp_participant_feature").feature_message.current_state,
    )


def _apply_bank(feature,site,bank):
    matches=[a for a in bank if a.node_name==site]
    if len(matches)>1:
        raise ValueError("Duplicate IMD LOOK logical site")
    return matches[0].apply_feature(feature) if matches else feature


def forward_from_participants(graph,oct_feature,cfp_feature,state,bank=()):
    if state not in ("complete",*PATTERNS):
        raise ValueError("Unknown IMD state")
    model=_fusion_model(graph)
    if state=="oct_missing":
        oct_value=model.empty_oct(len(cfp_feature))
        cfp_value=cfp_feature
    elif state=="cfp_missing":
        oct_value=oct_feature
        cfp_value=model.empty_cfp(len(oct_feature))
    else:
        oct_value,cfp_value=oct_feature,cfp_feature
    joined=torch.cat((model.norm_oct(oct_value),model.norm_cfp(cfp_value)),dim=1)
    joined=_apply_bank(joined,"imd_fusion_input",bank)
    fused=model.fusor(joined)
    fused=_apply_bank(fused,"fusion_feature",bank)
    logits=model.classifier(model.classifier_dropout(model.classifier_norm(fused)))
    return {"imd_fusion_input":joined,"fusion_feature":fused,"logits":logits}


@torch.no_grad()
def capture_batch(graph,batch,device,state,bank=()):
    if graph.training or any(p.requires_grad for p in graph.parameters()):
        raise ValueError("Frozen eval-mode IMD A required")
    oct_feature,cfp_feature=_participant_features(graph,batch,device)
    return forward_from_participants(graph,oct_feature,cfp_feature,state,bank)


@torch.no_grad()
def evaluate(graph,loader,device,state,bank=()):
    if state not in ("complete",*PATTERNS):
        raise ValueError("Unknown IMD evaluation state")
    ids=[];labels=[];logits=[]
    for batch in loader:
        out=capture_batch(graph,batch,device,state,bank)
        ids.extend(map(str,batch["participant_id"]))
        labels.append(batch["label"].numpy())
        logits.append(out["logits"].detach().cpu().numpy())
    labels=np.concatenate(labels).astype(np.int64)
    logits=np.concatenate(logits).astype(np.float64)
    probabilities=probabilities_from_logits(logits)
    return {
        "participant_ids":np.asarray(ids),
        "labels":labels,
        "logits":logits,
        "probabilities":probabilities,
        "scores":logits[:,1]-logits[:,0],
        "patterns":np.asarray([state]*len(labels)),
        "metrics":logit_metrics(labels,logits),
    }


@torch.no_grad()
def fit_pca_bank(graph,loader,device,output,identity,rank=32):
    if getattr(loader.dataset,"split",None)!="train" or getattr(loader.dataset,"augment",None):
        raise ValueError("IMD PCA requires unaugmented train")
    if graph.training or any(p.requires_grad for p in graph.parameters()):
        raise ValueError("Frozen eval A required")
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    manifest=root/"accepted.json"
    if manifest.exists():
        record=json.loads(manifest.read_text())
        if record.get("identity")!=identity or record.get("rank")!=rank:
            raise ValueError("IMD PCA identity changed")
        bank={}
        for site,row in record["sites"].items():
            path=root/row["file"]
            if file_sha256(path)!=row["sha256"]:
                raise ValueError("IMD PCA artifact changed")
            bank[site]=FullFeaturePCA.load(path)
        return bank
    arrays={site:[] for site in SITES};ids=[]
    for batch in loader:
        out=capture_batch(graph,batch,device,"complete")
        ids.extend(map(str,batch["participant_id"]))
        for site in SITES:
            arrays[site].append(out[site].detach().cpu().float())
    if len(ids)!=len(set(ids)):
        raise ValueError("Duplicate train participant during IMD PCA")
    bank={};rows={}
    for site in SITES:
        x=torch.cat(arrays[site],0)
        if x.ndim!=2 or len(x)!=len(ids) or rank>min(len(x)-1,x.shape[1]):
            raise ValueError("IMD PCA rank/data shape infeasible")
        mean=x.double().mean(0).float()
        std=((x.double().square().mean(0)-mean.double().square()).clamp_min(1e-12).sqrt()).float()
        z=((x-mean)/std).numpy().astype(np.float32,copy=False)
        pca=IncrementalPCA(n_components=rank,batch_size=max(512,rank))
        pca.fit(z)
        basis=FullFeaturePCA(
            node_name=site,factor=1,feature_shape=(x.shape[1],),downsample_shape=(x.shape[1],),
            mean=mean,std=std,pca_mean=torch.from_numpy(pca.mean_).float(),
            components=torch.from_numpy(pca.components_).float(),
            explained_variance_ratio=torch.from_numpy(pca.explained_variance_ratio_).float(),
            sample_count=len(x),fit_seconds=0.0,peak_rss_bytes=0,
            source_id=f"{identity}/{site}/rank{rank}",
            member_names=(site,),member_shapes=((x.shape[1],),),
            protocol=PROTOCOL,split_rule=SPLIT_RULE,spatial_method="interpolate",
        )
        path=root/f"{site}.pt";basis.save(path);bank[site]=basis
        rows[site]={"file":path.name,"sha256":file_sha256(path),"samples":len(x),"dimension":x.shape[1]}
    atomic_write_json({"schema":"look_imd_pca_v1","state":"accepted","identity":identity,"rank":rank,"sites":rows,"test_access":False},manifest)
    return bank


def _collect_statistics(graph,loader,device,pattern,bank,bases,sites,workspace_bytes,rank,should_pause):
    dims=[bases[s].std.numel() for s in sites]
    required=sum(8*(2*d*d+2*d+1) for d in dims)+max(estimated_workspace_bytes(d,rank) for d in dims)
    if required>workspace_bytes:
        raise MemoryError("IMD family dense workspace exceeds admission")
    stats={s:ResidualMoments.empty(d) for s,d in zip(sites,dims)}
    projected={s:ResidualMoments.empty(rank) for s in sites}
    stamps=[]
    for batch in loader:
        if should_pause(): raise SelectionPaused()
        if "participant_id" not in batch: raise ValueError("Participant identity required")
        stamp=fingerprint({"ids":list(map(str,batch["participant_id"])),"labels":batch["label"],"counts":batch["counts"]})
        oct_feature,cfp_feature=_participant_features(graph,batch,device)
        full=forward_from_participants(graph,oct_feature,cfp_feature,"complete",())
        missing=forward_from_participants(graph,oct_feature,cfp_feature,pattern,bank)
        for site in sites:
            b=bases[site]
            x=(missing[site].detach().cpu()-b.mean)/b.std
            y=(full[site].detach().cpu()-missing[site].detach().cpu())/b.std
            stats[site].update(x,y)
            q=b.components[:rank].double()
            projected[site].update(x.double()@q.T,y.double()@q.T)
        stamps.append(stamp)
    return stats,projected,stamps


def fit_family_trajectory(graph,train_loader,dev_loader,*,arm,pattern,bases,identity,output,device,workspace_bytes,rank=32,should_pause=lambda:False):
    if arm not in ARMS or pattern not in PATTERNS:
        raise ValueError("Unknown IMD LOOK family")
    if getattr(train_loader.dataset,"split",None)!="train" or getattr(train_loader.dataset,"augment",None):
        raise ValueError("IMD family fit requires unaugmented train")
    if getattr(dev_loader.dataset,"split",None)!="development":
        raise ValueError("IMD family selection requires development")
    if graph.training or any(p.requires_grad for p in graph.parameters()):
        raise ValueError("Frozen eval A required")
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    scientific_identity={"identity":identity,"arm":arm,"pattern":pattern,"rank":rank,
        "sites":list(SITES),"pca":{s:fingerprint(asdict(bases[s])) for s in SITES},
        "mode":"positive_forward_tree","penalty_policy":"prefix_train_pca_gcv",
        "missing_semantics":"learned_token_before_LOOK_visible_sites"}
    ready={}
    def fit(site,upstream,folder):
        prefix=fingerprint([a.record() for a in upstream])
        if prefix not in ready:
            start=max((SITES.index(a.node_name) for a in upstream),default=-1)+1
            targets=list(SITES[start:])
            stats,projected,stamps=_collect_statistics(
                graph,train_loader,device,pattern,upstream,bases,targets,workspace_bytes,rank,should_pause)
            ready.clear();ready[prefix]=(stats,projected,stamps)
        stats,projected,_=ready[prefix];s=stats[site];b=bases[site];q=b.components[:rank].double()
        ps=projected[site]
        ridge=_gcv_lambda(q@s.cxx@q.T,q@s.cxy@q.T,float(ps.syy),s.count,rank)
        template=LOOKArtifact(
            site,pattern,"learned_token_missingness",1,rank,b.feature_shape,b.downsample_shape,
            b.mean,b.std,b.pca_mean,b.components[:rank],torch.zeros(rank,rank),torch.zeros(rank),
            ridge,0.0,0.0,float(b.explained_variance_ratio[:rank].sum()),0.0,0,b.source_id,
            b.member_names,b.member_shapes,PROTOCOL,SPLIT_RULE,"interpolate")
        mapping=fit_map(s,rank,ridge,arm=arm,basis=b.components[:rank])
        mapping.diagnostics.update(selection="positive_forward_tree",penalty_policy="prefix_train_pca_gcv",
            missing_semantics="learned_token_before_LOOK_visible_sites",source_A_frozen=True)
        yield json.dumps({"rank":rank,"ridge_lambda":ridge},sort_keys=True),FamilyArtifact(template,mapping)
    def assess(bank):
        if should_pause(): raise SelectionPaused()
        result=evaluate(graph,dev_loader,device,pattern,bank)
        values=hashlib.sha256()
        for name in ("participant_ids","labels","logits"):
            a=np.ascontiguousarray(result[name]);values.update(name.encode());values.update(str(a.dtype).encode());values.update(str(a.shape).encode());values.update(a.tobytes())
        digest=values.hexdigest();key=fingerprint([a.record() for a in bank]);path=root/"predictions"/f"{key}_{digest}.npz";path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                if any(not np.array_equal(saved[n],result[n]) for n in ("participant_ids","labels","logits")):
                    raise ValueError("Stored IMD prediction evidence changed")
        else:
            np.savez_compressed(path,**{k:result[k] for k in ("participant_ids","labels","logits","probabilities","scores","patterns")})
        return {"role":"development","data_role":"development","score":float(result["metrics"]["macro_f1"]),
            "prediction":str(path),"sha256":file_sha256(path),"values_sha256":digest,"metrics":result["metrics"]}
    def save(a,p): atomic_save(p,a.record())
    def load(p): return FamilyArtifact.from_record(torch.load(p,map_location="cpu",weights_only=False))
    selected,result=fit_trajectory(identity=scientific_identity,sites=list(SITES),mode="positive_forward_tree",output=root,
        fit_candidates=fit,evaluate=assess,save_artifact=save,load_artifact=load,should_pause=should_pause)
    records=[a.record() for a in selected]
    atomic_save(root/"bank.pt",{"identity":scientific_identity,"bank":records,"sha256":fingerprint(records)})
    replay=assess([FamilyArtifact.from_record(r) for r in records])
    if replay["values_sha256"]!=result["final"]["values_sha256"]:
        raise ValueError("Reloaded IMD LOOK predictions changed")
    atomic_write_json({"full_replay":True,"scientific_acceptance":False,"test_access":False},root/"replay.json")
    return selected,result


def load_bank(path):
    record=torch.load(Path(path)/"bank.pt",map_location="cpu",weights_only=False)
    if fingerprint(record["bank"])!=record["sha256"]:
        raise ValueError("IMD family bank changed")
    return [FamilyArtifact.from_record(v) for v in record["bank"]]
