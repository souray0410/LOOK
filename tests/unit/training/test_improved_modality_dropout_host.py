import copy
import shutil

import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader, Dataset
from mhd_framework.models import create_model

from look.data.observed_pair import collate_observed
from look.models.improved_modality_dropout import build_improved_dropout_host
from look.models.observed_participant import ObservedParticipantModel
from look.training.improved_modality_dropout_host import DEFAULTS, evaluate_imd_state, train_improved_dropout_host
from look.runtime.host_checkpoint import capture_rng, restore_rng


class TinyPair(Dataset):
    def __init__(self, split, size, seed=51):
        self.split=split; self.augment=split=="train"; self.seed=seed; self.epoch=0
        self.participant_ids=[f"{split}-{i}" for i in range(size)]; self.counts=[1]*size
        g=torch.Generator().manual_seed(seed+(0 if split=="train" else 1000))
        self.oct=[torch.randn(1,3,224,224,generator=g) for _ in range(size)]
        self.cfp=[torch.randn(1,3,224,224,generator=g) for _ in range(size)]
        self.labels=[i%2 for i in range(size)]
    def set_epoch(self,epoch): self.epoch=epoch
    def __len__(self): return len(self.participant_ids)
    def __getitem__(self,i):
        return dict(oct=self.oct[i].clone(),cfp=self.cfp[i].clone(),label=self.labels[i],participant_id=self.participant_ids[i])


def _parents():
    config=dict(name="resnet18",num_classes=2,views=1)
    return [ObservedParticipantModel(create_model(config)) for _ in range(2)]


def _graph():
    return build_improved_dropout_host(*_parents(), hidden_dropout=0, classifier_dropout=0)


def _eq(a,b,path="root"):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0); return
    if isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b); return
    if isinstance(a,dict):
        assert set(a)==set(b),path
        for k in a:
            if path.endswith(".progress") and k=="seconds": continue
            _eq(a[k],b[k],path+"."+str(k))
        return
    if isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)): _eq(x,y,path+f"[{i}]")
        return
    assert a==b,path


def test_imd_training_preflight_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(2)
    config=dict(DEFAULTS); config.update(epochs=3, minimum_epochs=1, patience=2, microbatch=2, effective_batch=4, warmup_epochs=1)
    train=TinyPair("train",4); dev=TinyPair("development",2)
    identity="tiny-imd-resume"
    torch.manual_seed(1207); np.random.seed(1207)
    template=_graph(); init_model=copy.deepcopy(template.state_dict()); init_rng=capture_rng()
    def fresh():
        g=_graph(); g.load_state_dict(init_model,strict=True); restore_rng(init_rng); return g
    continuous=fresh()
    result=train_improved_dropout_host(continuous,train,dev,config,3416,tmp_path/"continuous",identity,torch.device("cpu"),preflight_updates=2)
    assert result["state"]=="paused" and result["total_updates"]==2
    continuous_dev=evaluate_imd_state(continuous,DataLoader(dev,batch_size=2,collate_fn=collate_observed),torch.device("cpu"),"complete")
    base=fresh()
    result=train_improved_dropout_host(base,train,dev,config,3416,tmp_path/"base",identity,torch.device("cpu"),preflight_updates=1)
    assert result["state"]=="paused" and result["total_updates"]==1
    resumed=[]
    for name in ("resume_a","resume_b"):
        shutil.copytree(tmp_path/"base",tmp_path/name)
        g=_graph()
        result=train_improved_dropout_host(g,train,dev,config,3416,tmp_path/name,identity,torch.device("cpu"),preflight_updates=2)
        assert result["state"]=="paused" and result["total_updates"]==2
        dev_result=evaluate_imd_state(g,DataLoader(dev,batch_size=2,collate_fn=collate_observed),torch.device("cpu"),"complete")
        resumed.append((torch.load(tmp_path/name/"last.pt",map_location="cpu",weights_only=False),dev_result["logits"]))
    uninterrupted=torch.load(tmp_path/"continuous/last.pt",map_location="cpu",weights_only=False)
    _eq(uninterrupted,resumed[0][0])
    _eq(resumed[0][0],resumed[1][0])
    np.testing.assert_array_equal(continuous_dev["logits"],resumed[0][1])
    np.testing.assert_array_equal(resumed[0][1],resumed[1][1])
    best=torch.load(tmp_path/"base/best.pt",map_location="cpu",weights_only=False)
    assert best.pop("framework_api")=="V5"
    assert best["selection"]=="complete_state_development_macro_f1"
    torch.save(best,tmp_path/"base/best.pt")
    with pytest.raises(ValueError,match="Current V5 selected host"):
        train_improved_dropout_host(_graph(),train,dev,config,3416,tmp_path/"base",identity,torch.device("cpu"),preflight_updates=1)


def test_imd_evaluation_states_are_deterministic_and_distinct():
    torch.set_num_threads(2)
    graph=_graph().eval(); data=TinyPair("development",3); loader=DataLoader(data,batch_size=3,collate_fn=collate_observed)
    complete=evaluate_imd_state(graph,loader,torch.device("cpu"),"complete")
    oct_a=evaluate_imd_state(graph,loader,torch.device("cpu"),"oct_missing")
    oct_b=evaluate_imd_state(graph,loader,torch.device("cpu"),"oct_missing")
    np.testing.assert_array_equal(oct_a["logits"],oct_b["logits"])
    assert complete["logits"].shape==(3,2)
    assert oct_a["patterns"].tolist()==["oct_missing"]*3
