"""Resumable balanced modality-dropout fine-tuning for a deterministic native LOOK host."""
from __future__ import annotations
import copy, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from look.data.observed_pair import collate_observed
from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle
from look.models.graph import optimizer_parameter_groups
from look.models.native_host import forward_host
from look.runtime.host_checkpoint import atomic_save, cpu_tree, load, save
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.training.observed_host import HostSchedule, validate_config

MASK_SEED_OFFSET=910019
STATE_NAMES=("complete","oct_missing","cfp_missing")


def _new_mask_generator(seed):
    g=torch.Generator(device="cpu");g.manual_seed(int(seed)+MASK_SEED_OFFSET);return g


def _state_counts(states):
    return {name:int((states==i).sum()) for i,name in enumerate(STATE_NAMES)}


def _schedule_file(out, epoch): return Path(out)/"mask_schedules"/f"epoch_{epoch:03d}.json"


def _participant_order_fingerprint(participant_ids):
    payload=json.dumps([str(x) for x in participant_ids],ensure_ascii=False,separators=(",",":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mask_schedule(progress,generator,epoch,participant_ids,out):
    n=len(participant_ids)
    if progress.get("mask_epoch")==epoch:
        states=progress.get("mask_states")
        if not isinstance(states,torch.Tensor) or states.dtype!=torch.int64 or tuple(states.shape)!=(n,):
            raise ValueError("Saved balanced-dropout mask schedule invalid")
        path=_schedule_file(out,epoch)
        if not path.exists() or file_sha256(path)!=progress.get("mask_schedule_sha256"):
            raise ValueError("Saved balanced-dropout mask schedule file changed")
        payload=json.loads(path.read_text())
        if (payload.get("participant_count")!=n
                or payload.get("participant_order_sha256")!=_participant_order_fingerprint(participant_ids)):
            raise ValueError("Saved balanced-dropout participant order changed")
        return states
    if progress["offset"]!=0 or progress.get("mask_epoch") is not None:
        raise ValueError("Cannot replace in-progress balanced-dropout mask schedule")
    states=torch.randint(3,(n,),generator=generator,dtype=torch.int64)
    payload={"schema":"look_balanced_dropout_mask_schedule_v1","epoch":epoch,
        "mask_seed":int(progress["mask_seed"]),"participant_count":n,
        "participant_order_sha256":_participant_order_fingerprint(participant_ids),
        "states":[STATE_NAMES[int(i)] for i in states.tolist()],"counts":_state_counts(states)}
    path=_schedule_file(out,epoch);path.parent.mkdir(parents=True,exist_ok=True);atomic_write_json(payload,path)
    progress["mask_epoch"]=epoch;progress["mask_states"]=states.clone();progress["mask_rng_state"]=generator.get_state().clone()
    progress["mask_schedule_path"]=str(path);progress["mask_schedule_sha256"]=file_sha256(path)
    return states


def _mask_inputs(batch,participant_states,device):
    if participant_states.ndim!=1 or participant_states.dtype!=torch.int64 or torch.any((participant_states<0)|(participant_states>2)):
        raise ValueError("Invalid participant missing-state vector")
    counts=list(map(int,batch["counts"]))
    if len(counts)!=len(participant_states) or any(n not in (1,2) for n in counts):
        raise ValueError("Participant/eye ownership changed")
    eye_states=torch.repeat_interleave(participant_states,torch.tensor(counts,dtype=torch.long)).to(device)
    oct_tensor=batch["oct"].to(device).clone();cfp_tensor=batch["cfp"].to(device).clone()
    if len(eye_states)!=len(oct_tensor) or len(oct_tensor)!=len(cfp_tensor):
        raise ValueError("Eye-level mask expansion changed")
    oct_tensor[eye_states==1]=0;cfp_tensor[eye_states==2]=0
    return oct_tensor,cfp_tensor


def _save_boundary(path,*,graph,optimizer,scheduler,identity,progress,node_ids,mask_generator):
    progress["mask_rng_state"]=mask_generator.get_state().clone()
    return save(path,model=graph,optimizer=optimizer,scheduler=scheduler,identity=identity,progress=progress,node_ids=node_ids)


def _load_boundary(path,*,graph,optimizer,scheduler,identity,node_ids,mask_generator):
    progress=load(path,model=graph,optimizer=optimizer,scheduler=scheduler,identity=identity,node_ids=node_ids)
    state=progress.get("mask_rng_state")
    if not isinstance(state,torch.Tensor): raise ValueError("Checkpoint lacks balanced-dropout mask RNG state")
    mask_generator.set_state(state.cpu())
    return progress


def train_balanced_dropout_host(graph,train,development,config,seed,output,identity,device,should_pause=lambda:False,*,preflight_target_updates=None):
    validate_config(config);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if train.split!="train" or development.split!="development" or set(train.participant_ids)&set(development.participant_ids):
        raise ValueError("Balanced-dropout split identity changed")
    if seed!=3416: raise ValueError("Balanced-dropout finite package locked to seed3416")
    optimizer=torch.optim.AdamW(optimizer_parameter_groups(graph,config["pretrained_lr"],config["new_layer_lr"]),weight_decay=config["weight_decay"])
    scheduler=HostSchedule(optimizer,config);optimizer.zero_grad(set_to_none=True)
    node_ids=[(n.id,n.name) for n in sorted(graph.nodes,key=lambda n:n.id)]
    mask_generator=_new_mask_generator(seed)
    progress=dict(epoch=1,offset=0,updates=0,history=[],epoch_loss=0.0,epoch_seen=0,seconds=0.0,
        mask_seed=seed+MASK_SEED_OFFSET,mask_epoch=None,mask_states=None,mask_rng_state=mask_generator.get_state().clone(),
        mask_schedule_path=None,mask_schedule_sha256=None)
    if (out/"last.pt").exists():
        progress=_load_boundary(out/"last.pt",graph=graph,optimizer=optimizer,scheduler=scheduler,identity=identity,node_ids=node_ids,mask_generator=mask_generator)
    started=time.monotonic()
    def checkpoint():
        nonlocal started
        now=time.monotonic();progress["seconds"]+=now-started;started=now
        _save_boundary(out/"last.pt",graph=graph,optimizer=optimizer,scheduler=scheduler,identity=identity,progress=progress,node_ids=node_ids,mask_generator=mask_generator)
    def status(state):
        atomic_write_json({"state":state,"identity":identity,"epoch":progress["epoch"],"offset":progress["offset"],
            "updates":progress["updates"],"mask_epoch":progress.get("mask_epoch"),"mask_schedule_sha256":progress.get("mask_schedule_sha256"),
            "updated_at":time.time(),"test_access":False},out/"status.json")
    dev_loader=DataLoader(development,batch_size=config["microbatch"],collate_fn=collate_observed,shuffle=False,num_workers=config["num_workers"],
        generator=torch.Generator().manual_seed(seed))
    if preflight_target_updates is not None and progress["updates"]>=preflight_target_updates:
        checkpoint();status("paused");return {"state":"paused","updates":0,"total_updates":progress["updates"],"reason":"preflight_target_already_reached"}
    if not (out/"best.pt").exists():
        result=evaluate_missing(graph,dev_loader,device,fixed_pattern="complete")
        scheduler.step(result["metrics"]["macro_f1"],0)
        atomic_save(out/"best.pt",{"identity":identity,"epoch":0,"model":cpu_tree(graph.state_dict()),"node_ids":node_ids,
            "selection":"complete_development_macro_f1"})
        save_prediction_bundle(result,out/"development_predictions.npz");checkpoint()
    launch_updates=0
    while progress["epoch"]<=config["epochs"]:
        if scheduler.stall>=config["patience"] and progress["epoch"]>config["minimum_epochs"]: break
        epoch=progress["epoch"];train.set_epoch(epoch);scheduler.begin(epoch);graph.train()
        states=_mask_schedule(progress,mask_generator,epoch,train.participant_ids,out)
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        while progress["offset"]<len(order):
            if should_pause(): checkpoint();status("paused");return {"state":"paused"}
            block=order[progress["offset"]:progress["offset"]+config["effective_batch"]]
            loader=DataLoader(Subset(train,block),batch_size=config["microbatch"],collate_fn=collate_observed,shuffle=False,
                num_workers=config["num_workers"],generator=torch.Generator().manual_seed(seed+epoch))
            cursor=0
            for batch in loader:
                n=len(batch["label"]);indices=block[cursor:cursor+n];cursor+=n
                expected=[str(train.participant_ids[i]) for i in indices]
                if list(map(str,batch["participant_id"]))!=expected: raise ValueError("Training participant order changed")
                participant_states=states[torch.as_tensor(indices,dtype=torch.long)]
                oct_tensor,cfp_tensor=_mask_inputs(batch,participant_states,device)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=config["precision"]=="bf16"):
                    forward_host(graph,oct_tensor,cfp_tensor,batch["counts"],batch["label"].to(device),loss_scale=n/len(block))
                    loss=graph.get_node_by_name("loss").feature_message.current_state
                if not torch.isfinite(loss): raise ValueError("Non-finite balanced-dropout loss")
                graph.backward(levels=graph.backward_levels)
                progress["epoch_loss"]+=float(loss.detach())*len(block);progress["epoch_seen"]+=n
            if cursor!=len(block): raise ValueError("Effective-batch accounting changed")
            torch.nn.utils.clip_grad_norm_(graph.parameters(),config["clip"],error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            progress["offset"]+=len(block);progress["updates"]+=1;launch_updates+=1;status("training")
            if should_pause() or (preflight_target_updates is not None and progress["updates"]>=preflight_target_updates):
                checkpoint();status("paused");return {"state":"paused","updates":launch_updates,"total_updates":progress["updates"],"reason":"preflight_or_resource"}
        checkpoint();status("validating")
        result=evaluate_missing(graph,dev_loader,device,fixed_pattern="complete");score=result["metrics"]["macro_f1"];improved=scheduler.step(score,epoch)
        counts=_state_counts(states);schedule_sha=progress["mask_schedule_sha256"]
        if improved:
            atomic_save(out/"best.pt",{"identity":identity,"epoch":epoch,"model":cpu_tree(graph.state_dict()),"node_ids":node_ids,
                "selection":"complete_development_macro_f1"})
            save_prediction_bundle(result,out/"development_predictions.npz")
        progress["history"].append({"epoch":epoch,"loss":progress["epoch_loss"]/max(1,progress["epoch_seen"]),"metrics":result["metrics"],
            "mask_counts":counts,"mask_schedule_sha256":schedule_sha})
        progress.update(epoch=epoch+1,offset=0,epoch_loss=0.0,epoch_seen=0,mask_epoch=None,mask_states=None,
            mask_schedule_path=None,mask_schedule_sha256=None)
        atomic_write_json(progress["history"],out/"history.json");checkpoint()
    if scheduler.stall<config["patience"]:
        status("needs_review_epoch_cap");return {"state":"needs_review_epoch_cap"}
    selected=torch.load(out/"best.pt",map_location="cpu",weights_only=False)
    if selected["identity"]!=identity or selected["node_ids"]!=node_ids: raise ValueError("Selected balanced-dropout host identity changed")
    graph.load_state_dict(selected["model"],strict=True);graph.eval();replay=evaluate_missing(graph,dev_loader,device,fixed_pattern="complete")
    saved=np.load(out/"development_predictions.npz",allow_pickle=False)
    if not np.array_equal(saved["participant_ids"].astype(str),replay["participant_ids"].astype(str)) or not np.array_equal(saved["labels"],replay["labels"]):
        raise ValueError("Selected balanced-dropout development identity changed")
    if not np.array_equal(saved["logits"],replay["logits"]): raise ValueError("Selected balanced-dropout logits not exact")
    receipt={"schema":"look_balanced_dropout_host_v1","state":"accepted","identity":identity,"best_epoch":scheduler.best_epoch,
        "stop_epoch":progress["epoch"]-1,"plateau":True,"selection":"complete_development_macro_f1","train_missing_probabilities":
        {"complete":1/3,"oct_missing":1/3,"cfp_missing":1/3},"mask_seed":seed+MASK_SEED_OFFSET,"test_access":False,
        "metrics":replay["metrics"],"seconds":progress["seconds"],"source_nodes":graph.native_host_provenance,
        "files":{name:file_sha256(out/name) for name in ("best.pt","last.pt","history.json","development_predictions.npz")}}
    atomic_write_json(receipt,out/"accepted.json");status("completed");return receipt


def _equal(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and set(a)==set(b) and all(_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return isinstance(b,type(a)) and len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    return a==b


def profile_resume(make_graph,train,development,config,seed,output,identity,device,should_pause=lambda:False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/"accepted.json").exists(): return json.loads((out/"accepted.json").read_text())
    import random
    def reset():
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    reset();g=make_graph();train_balanced_dropout_host(g,train,development,config,seed,out/"continuous",identity,device,should_pause,preflight_target_updates=2)
    del g;torch.cuda.empty_cache();reset();g=make_graph();train_balanced_dropout_host(g,train,development,config,seed,out/"resumed",identity,device,should_pause,preflight_target_updates=1)
    del g;torch.cuda.empty_cache();g=make_graph();train_balanced_dropout_host(g,train,development,config,seed,out/"resumed",identity,device,should_pause,preflight_target_updates=2)
    c=torch.load(out/"continuous/last.pt",map_location="cpu",weights_only=False);r=torch.load(out/"resumed/last.pt",map_location="cpu",weights_only=False)
    cp=copy.deepcopy(c);rp=copy.deepcopy(r);cp["progress"]["seconds"]=0.0;rp["progress"]["seconds"]=0.0
    exact={key:_equal(cp[key],rp[key]) for key in ("model","optimizer","scheduler","progress","rng","node_ids")}
    if not all(exact.values()): raise ValueError("Balanced-dropout uninterrupted/resumed update differs: "+repr(exact))
    def eval_state(state):
        graph=make_graph();graph.load_state_dict(state["model"],strict=True);graph.eval()
        loader=DataLoader(development,batch_size=config["microbatch"],collate_fn=collate_observed,shuffle=False,num_workers=0)
        value=evaluate_missing(graph,loader,device,fixed_pattern="complete");del graph;torch.cuda.empty_cache();return value
    a=eval_state(c);b=eval_state(r)
    dev_exact=np.array_equal(a["logits"],b["logits"]) and np.array_equal(a["participant_ids"].astype(str),b["participant_ids"].astype(str))
    if not dev_exact: raise ValueError("Balanced-dropout resume development replay differs")
    receipt={"schema":"look_balanced_dropout_profile_v1","state":"accepted","identity":identity,"updates":2,
        "uninterrupted_vs_resumed_next_update_exact":True,"components_exact":exact,"development_logits_exact":True,
        "peak_reserved_bytes":int(torch.cuda.max_memory_reserved()),"test_access":False}
    atomic_write_json(receipt,out/"accepted.json");return receipt
