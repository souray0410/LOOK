import copy
import torch
import numpy as np
import pytest
from mhd_framework.models import create_model
from look.models.observed_participant import ObservedParticipantModel
from look.models.improved_modality_dropout import build_improved_dropout_host, forward_improved_dropout_host
from look.methods.improved_dropout_family import forward_from_participants, _fusion_model


def _graph():
    config=dict(name="resnet18",num_classes=2,views=1)
    parents=[ObservedParticipantModel(create_model(config)) for _ in range(2)]
    g=build_improved_dropout_host(*parents,hidden_dropout=0,classifier_dropout=0)
    g.eval()
    for p in g.parameters():p.requires_grad_(False)
    return g


def test_logical_forward_matches_graph_for_all_states():
    torch.manual_seed(11);g=_graph()
    oct_x=torch.randn(2,3,224,224);cfp_x=torch.randn(2,3,224,224);counts=[1,1]
    from look.methods.improved_dropout_family import _participant_features
    batch={"oct":oct_x,"cfp":cfp_x,"counts":counts}
    oct_f,cfp_f=_participant_features(g,batch,torch.device("cpu"))
    for state in ("complete","oct_missing","cfp_missing"):
        expected=forward_improved_dropout_host(g,oct_x,cfp_x,counts,state=state)
        got=forward_from_participants(g,oct_f,cfp_f,state)["logits"]
        torch.testing.assert_close(got,expected,rtol=0,atol=0)


def test_missing_logical_input_does_not_read_absent_participant_feature():
    torch.manual_seed(13);g=_graph();m=_fusion_model(g)
    oct_f=torch.randn(3,512);cfp_f=torch.randn(3,512)
    a=forward_from_participants(g,oct_f,cfp_f,"oct_missing")
    b=forward_from_participants(g,torch.full_like(oct_f,float("nan")),cfp_f,"oct_missing")
    torch.testing.assert_close(a["imd_fusion_input"],b["imd_fusion_input"],rtol=0,atol=0)
    torch.testing.assert_close(a["logits"],b["logits"],rtol=0,atol=0)
    c=forward_from_participants(g,oct_f,cfp_f,"cfp_missing")
    d=forward_from_participants(g,oct_f,torch.full_like(cfp_f,float("nan")),"cfp_missing")
    torch.testing.assert_close(c["imd_fusion_input"],d["imd_fusion_input"],rtol=0,atol=0)
    torch.testing.assert_close(c["logits"],d["logits"],rtol=0,atol=0)


def test_sites_are_after_missing_token_substitution_and_before_classifier():
    g=_graph()
    from look.methods.improved_dropout_family import SITES
    assert SITES==("imd_fusion_input","fusion_feature")
    # Native graph correction sites are all raw/pre-token states or final logits.
    # The continuation deliberately exposes only safe logical post-token states.
    assert "imd_fusion_input" not in tuple(g.correction_nodes)
    assert "fusion_feature" not in tuple(g.correction_nodes)
    assert "oct_participant_feature" in tuple(g.correction_nodes)
    assert "cfp_participant_feature" in tuple(g.correction_nodes)


class TinyPair(torch.utils.data.Dataset):
    def __init__(self,split,size,seed=23):
        self.split=split;self.augment=False;self.participant_ids=[f"{split}-{i}" for i in range(size)]
        g=torch.Generator().manual_seed(seed+(0 if split=="train" else 1000))
        self.oct=[torch.randn(1,3,224,224,generator=g) for _ in range(size)]
        self.cfp=[torch.randn(1,3,224,224,generator=g) for _ in range(size)]
        self.labels=[i%2 for i in range(size)]
    def __len__(self):return len(self.participant_ids)
    def __getitem__(self,i):
        return {"oct":self.oct[i].clone(),"cfp":self.cfp[i].clone(),"label":self.labels[i],"participant_id":self.participant_ids[i]}
    def set_epoch(self,epoch):pass


def _collate(rows):
    return {"oct":torch.cat([r["oct"] for r in rows]),"cfp":torch.cat([r["cfp"] for r in rows]),
        "counts":[1]*len(rows),"label":torch.tensor([r["label"] for r in rows]),"participant_id":[r["participant_id"] for r in rows]}


@pytest.mark.parametrize("arm",["pca_free_mean","residual_rrr"])
@pytest.mark.parametrize("pattern",["oct_missing","cfp_missing"])
def test_small_real_mhd_imd_family_tree_replays(tmp_path,arm,pattern):
    from torch.utils.data import DataLoader
    from look.methods.improved_dropout_family import fit_pca_bank,fit_family_trajectory,evaluate,load_bank
    torch.set_num_threads(2);torch.manual_seed(37)
    g=_graph()
    train=TinyPair("train",6);dev=TinyPair("development",4)
    train_loader=DataLoader(train,batch_size=3,shuffle=False,collate_fn=_collate)
    dev_loader=DataLoader(dev,batch_size=2,shuffle=False,collate_fn=_collate)
    pca=fit_pca_bank(g,train_loader,torch.device("cpu"),tmp_path/"pca","tiny-imd",rank=2)
    bank,result=fit_family_trajectory(g,train_loader,dev_loader,arm=arm,pattern=pattern,bases=pca,
        identity="tiny-imd",output=tmp_path/arm/pattern,device=torch.device("cpu"),workspace_bytes=256*1024**2,rank=2)
    assert result["mode"]=="positive_forward_tree" and result["test_access"] is False
    loaded=load_bank(tmp_path/arm/pattern)
    first=evaluate(g,dev_loader,torch.device("cpu"),pattern,bank)
    replay=evaluate(g,dev_loader,torch.device("cpu"),pattern,loaded)
    np.testing.assert_array_equal(first["participant_ids"],replay["participant_ids"])
    np.testing.assert_array_equal(first["labels"],replay["labels"])
    np.testing.assert_array_equal(first["logits"],replay["logits"])
