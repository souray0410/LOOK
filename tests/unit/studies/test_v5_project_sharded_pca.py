import json
import torch
import pytest
from look.runtime.state import file_sha256,stable_hash
from look.studies import v5_project_sharded_pca as subject

def write(p,x): p.write_text(json.dumps(x))
def cache(tmp_path,complete=True):
 root=tmp_path/'cache';(root/'chunks'/'000000').mkdir(parents=True)
 identity={'framework_commit':subject.FRAMEWORK,'models_commit':subject.MODELS,'test_access':False,'sites':['s']}
 write(root/'identity.json',identity)
 shard=root/'chunks'/'000000'/'00_s.pt';torch.save(torch.ones(2,3,2,2),shard)
 write(root/'chunks'/'000000'/'receipt.json',{'batch':0,'sites':['s'],'test_access':False,'participant_ids':['a','b'],'participants_sha256':subject.participant_sha(['a','b']),'files':{'s':{'path':'00_s.pt','sha256':file_sha256(shard)}}})
 write(root/'accepted.json',{'schema':subject.SCHEMA,'state':'accepted_complete' if complete else 'accepted_partial','completed_batches':1,'total_batches':1,'participants_total':2,'identity_sha256':stable_hash(identity),'test_access':False})
 return root

def test_complete_cache_and_factory_verify_shard(tmp_path):
 root=cache(tmp_path);_,_,rows=subject.validate_complete_cache(root)
 full,shape,down=next(subject.cached_feature_factory(rows,'s',1)())
 assert full.shape==(2,12) and shape==(3,2,2) and down==(3,2,2)

def test_partial_cache_fails_closed(tmp_path):
 with pytest.raises(ValueError,match='Accepted complete'): subject.validate_complete_cache(cache(tmp_path,False))

def test_modified_shard_fails_closed(tmp_path):
 root=cache(tmp_path);_,_,rows=subject.validate_complete_cache(root)
 torch.save(torch.zeros(2,3,2,2),rows[0]/'00_s.pt')
 with pytest.raises(ValueError,match='hash'): next(subject.cached_feature_factory(rows,'s',1)())


def test_participant_order_receipt_fails_closed(tmp_path):
 root=cache(tmp_path);p=root/'chunks'/'000000'/'receipt.json';row=json.loads(p.read_text());row['participant_ids'].reverse();write(p,row)
 with pytest.raises(ValueError,match='participant identity'): subject.validate_complete_cache(root)
