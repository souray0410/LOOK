import pytest
import torch
from look.data.observed_pair import ObservedPair,collate_observed


class Native:
    def __init__(self):
        self.rows=[dict(id='a',label=0,eyes=['left']),dict(id='b',label=1,eyes=['left','right'])];self.augment=False
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return torch.ones(len(self.rows[i]['eyes']),3,2,2),self.rows[i]['label'],i


def test_keeps_only_real_eyes():
    d=ObservedPair(Native(),Native(),'train')
    b=collate_observed([d[0],d[1]])
    assert b['counts']==[1,2] and b['cfp'].shape[0]==3 and b['label'].shape[0]==2
    with pytest.raises(ValueError,match='test'):ObservedPair(Native(),Native(),'test')


def test_reordered_eyes_rejected():
    n=Native();n.rows[1]['eyes'].reverse()
    with pytest.raises(ValueError):ObservedPair(Native(),n,'train')
