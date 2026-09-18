import json
import numpy as np
import pytest
from look.data.array_pair import ArrayPair
from look.runtime.state import file_sha256


def make(root):
    audit=dict(status='accepted',test_used=False,manifest_sha256={},tracks={})
    for track in ('cfp_2d','oct_bscan_2d'):
        folder=root/track;folder.mkdir();rows=[]
        for i in range(2):
            p=folder/f'{i}.npy';np.save(p,np.zeros((2,3,224,224),dtype=np.uint8))
            rows.append(dict(id=str(i),label=i,file=p.name,sha256=file_sha256(p)))
        p=folder/'train.json';p.write_text(json.dumps(dict(role='train',samples=rows)))
        audit['manifest_sha256'][track]={'train':file_sha256(p)}
        audit['tracks'][track]=dict(mean=[.5]*3,std=[.5]*3)
    (root/'accepted.json').write_text(json.dumps(audit))
    return audit


def test_pair_and_normalization(tmp_path):
    make(tmp_path);ds=ArrayPair(tmp_path,'train')
    assert ds[0]['cfp'].shape==(2,3,224,224)
    assert ds[0]['cfp'].min()==-1
    assert ds.participant_ids==['0','1']


def test_sealed_role_refused(tmp_path):
    with pytest.raises(ValueError):ArrayPair(tmp_path,'test')


def test_array_tamper_refused(tmp_path):
    make(tmp_path);ds=ArrayPair(tmp_path,'train')
    (tmp_path/'cfp_2d/0.npy').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Array changed'):ds[0]


def test_pair_label_mismatch_refused(tmp_path):
    audit=make(tmp_path);p=tmp_path/'oct_bscan_2d/train.json';obj=json.loads(p.read_text())
    obj['samples'][0]['label']=1;p.write_text(json.dumps(obj))
    audit['manifest_sha256']['oct_bscan_2d']['train']=file_sha256(p)
    (tmp_path/'accepted.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='Paired'):ArrayPair(tmp_path,'train')
