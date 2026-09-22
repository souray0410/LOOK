import json
from types import SimpleNamespace
import torch

from look.training.balanced_dropout_host import (
    _mask_inputs, _mask_schedule, _new_mask_generator, _participant_order_fingerprint,
    _profile_semantic_checkpoint, MASK_SEED_OFFSET,
)


def test_participant_mask_expands_to_all_owned_eyes():
    batch={
        "counts":[2,1,2],
        "oct":torch.arange(5*3*2*2,dtype=torch.float32).reshape(5,3,2,2)+1,
        "cfp":torch.arange(5*3*2*2,dtype=torch.float32).reshape(5,3,2,2)+101,
    }
    states=torch.tensor([1,2,0],dtype=torch.int64)
    oct_tensor,cfp_tensor=_mask_inputs(batch,states,torch.device("cpu"))
    assert torch.count_nonzero(oct_tensor[:2])==0
    assert torch.count_nonzero(cfp_tensor[2:3])==0
    assert torch.equal(cfp_tensor[:2],batch["cfp"][:2])
    assert torch.equal(oct_tensor[2:3],batch["oct"][2:3])
    assert torch.equal(oct_tensor[3:],batch["oct"][3:])
    assert torch.equal(cfp_tensor[3:],batch["cfp"][3:])


def test_epoch_mask_schedule_is_persisted_and_reused(tmp_path):
    ids=["a","b","c","d","e"]
    g=_new_mask_generator(3416)
    progress=dict(offset=0,mask_seed=3416+MASK_SEED_OFFSET,mask_epoch=None,mask_states=None,
        mask_rng_state=g.get_state().clone(),mask_schedule_path=None,mask_schedule_sha256=None)
    first=_mask_schedule(progress,g,1,ids,tmp_path)
    path=tmp_path/"mask_schedules/epoch_001.json"
    payload=json.loads(path.read_text())
    assert "participant_ids" not in payload
    assert payload["participant_count"]==len(ids)
    assert payload["participant_order_sha256"]==_participant_order_fingerprint(ids)
    assert sum(payload["counts"].values())==len(ids)
    saved_sha=progress["mask_schedule_sha256"]
    second=_mask_schedule(progress,g,1,ids,tmp_path)
    assert torch.equal(first,second)
    assert progress["mask_schedule_sha256"]==saved_sha


def test_epoch_mask_schedule_is_deterministic_for_same_seed(tmp_path):
    ids=[str(i) for i in range(20)]
    values=[]
    for name in ("a","b"):
        root=tmp_path/name;g=_new_mask_generator(3416)
        progress=dict(offset=0,mask_seed=3416+MASK_SEED_OFFSET,mask_epoch=None,mask_states=None,
            mask_rng_state=g.get_state().clone(),mask_schedule_path=None,mask_schedule_sha256=None)
        values.append(_mask_schedule(progress,g,3,ids,root))
    assert torch.equal(values[0],values[1])


def test_epoch_mask_schedule_rejects_participant_order_change(tmp_path):
    ids=["a","b","c","d","e"]
    g=_new_mask_generator(3416)
    progress=dict(offset=0,mask_seed=3416+MASK_SEED_OFFSET,mask_epoch=None,mask_states=None,
        mask_rng_state=g.get_state().clone(),mask_schedule_path=None,mask_schedule_sha256=None)
    _mask_schedule(progress,g,1,ids,tmp_path)
    try:
        _mask_schedule(progress,g,1,list(reversed(ids)),tmp_path)
    except ValueError as exc:
        assert "participant order changed" in str(exc)
    else:
        raise AssertionError("participant-order mismatch must fail closed")


def test_profile_semantic_checkpoint_normalizes_only_lane_bookkeeping():
    base={
        "progress":{
            "seconds":1.25,
            "mask_schedule_path":"/tmp/continuous/mask_schedules/epoch_001.json",
            "mask_schedule_sha256":"abc123",
            "updates":2,
            "offset":32,
            "mask_states":torch.tensor([0,1,2],dtype=torch.int64),
        },
        "model":{"x":torch.tensor([1.0])},
    }
    other={
        "progress":{
            "seconds":9.75,
            "mask_schedule_path":"/tmp/resumed/mask_schedules/epoch_001.json",
            "mask_schedule_sha256":"abc123",
            "updates":2,
            "offset":32,
            "mask_states":torch.tensor([0,1,2],dtype=torch.int64),
        },
        "model":{"x":torch.tensor([1.0])},
    }
    a=_profile_semantic_checkpoint(base);b=_profile_semantic_checkpoint(other)
    assert a["progress"]["seconds"]==b["progress"]["seconds"]==0.0
    assert a["progress"]["mask_schedule_path"]==b["progress"]["mask_schedule_path"]=="sha256:abc123"
    assert a["progress"]["updates"]==b["progress"]["updates"]==2
    assert torch.equal(a["progress"]["mask_states"],b["progress"]["mask_states"])


def test_profile_semantic_checkpoint_keeps_schedule_sha_identity():
    value={"progress":{"seconds":0.0,"mask_schedule_path":"/tmp/a.json","mask_schedule_sha256":"left"}}
    other={"progress":{"seconds":0.0,"mask_schedule_path":"/tmp/b.json","mask_schedule_sha256":"right"}}
    a=_profile_semantic_checkpoint(value);b=_profile_semantic_checkpoint(other)
    assert a["progress"]["mask_schedule_path"]!=b["progress"]["mask_schedule_path"]
    assert a["progress"]["mask_schedule_sha256"]!=b["progress"]["mask_schedule_sha256"]


def test_profile_semantic_checkpoint_rejects_unhashed_schedule_path():
    value={"progress":{"seconds":0.0,"mask_schedule_path":"/tmp/a.json","mask_schedule_sha256":None}}
    try:
        _profile_semantic_checkpoint(value)
    except ValueError as exc:
        assert "without SHA" in str(exc)
    else:
        raise AssertionError("unhashed profile schedule path must fail closed")


def test_profile_resume_uses_semantic_checkpoint_normalization():
    import inspect
    from look.training import balanced_dropout_host as module
    source=inspect.getsource(module.profile_resume)
    assert "cp=_profile_semantic_checkpoint(c);rp=_profile_semantic_checkpoint(r)" in source
    assert 'cp["progress"]["seconds"]=0.0' not in source
