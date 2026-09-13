import numpy as np
import pytest
import torch
from look.data.mechanism_samples import ordered_ids,target_permutation,mask_training_batch


def test_participant_permutation_and_global_rng():
    torch.manual_seed(42);before=torch.get_rng_state().clone()
    ids=['a','b','c','d'];counts=[1,2,1,2]
    order=target_permutation(ids,counts,6)
    assert list(order)==[3,4,5,0,1,2]
    assert sorted(order.tolist())==list(range(6))
    assert torch.equal(before,torch.get_rng_state())
    with pytest.raises(ValueError,match='singleton'):target_permutation(['a','b','c'],[1,2,2],3)
    first=ordered_ids(ids,3,'subset');assert first==ordered_ids(list(reversed(ids)),3,'subset')


def test_missing_rng_exact_recovery():
    gen=torch.Generator().manual_seed(73)
    b=dict(cfp=torch.ones(6,3,2,2),oct=torch.ones(6,3,2,2),counts=[1,2,1,2])
    before=torch.get_rng_state().clone();state=gen.get_state()
    x,s=mask_training_batch(b,gen);gen.set_state(state);y,t=mask_training_batch(b,gen)
    assert torch.equal(s,t) and torch.equal(x['cfp'],y['cfp'])
    assert torch.equal(before,torch.get_rng_state()) and b['cfp'].all()
    assert torch.equal(x['oct'][1],x['oct'][2]) and torch.equal(x['cfp'][4],x['cfp'][5])


def test_same_participant_permutation_before_and_after_eye_pooling():
    import numpy as np
    ids=['a','b','c','d'];counts=[1,2,1,2]
    from look.data.mechanism_samples import target_permutation
    eye=target_permutation(ids,counts,7,counts)
    participant=target_permutation(ids,[1]*4,7,counts)
    owners=np.repeat(np.arange(4),counts)
    np.testing.assert_array_equal(owners[eye],np.repeat(participant,counts))
