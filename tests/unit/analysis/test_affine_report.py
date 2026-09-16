import numpy as np
import pytest
from look.analysis.affine_report import report
from look.methods.affine_family import ARMS
from look.runtime.state import file_sha256


def test_report_preserves_negative_effects_zero_variance_and_pairing(tmp_path):
    records=[];y=np.array([0,1]*10);host=dict(disease='cataract',architecture='resnet50',position='deep',seed=3416)
    for method in (*ARMS,'host'):
        for pattern in ('oct_missing','cfp_missing'):
            p=tmp_path/f'{method}_{pattern}.npz'
            pred=1-y if method=='residual_ridge' else y
            np.savez(p,participant_ids=np.arange(20),labels=y,logits=np.eye(2)[pred])
            records.append(dict(method=method,scenario=pattern,path=str(p),sha256=file_sha256(p),metrics={'macro_f1':float(method!='residual_ridge')}))
    result=report([(host,'terminal',records)],tmp_path/'report',iterations=100)
    assert len(result['contrasts'])==51 and result['test_access'] is False
    assert any(x['difference']<0 for x in result['contrasts'])
    assert any(x['zero_variance'] for x in result['contrasts'])
    with pytest.raises(ValueError,match='Incomplete'):report([(host,'terminal',records[:-1])],tmp_path/'bad',iterations=10)
    with pytest.raises(ValueError,match='Duplicate seed'):report([(host,'terminal',records)]*2,tmp_path/'duplicate',iterations=10)
