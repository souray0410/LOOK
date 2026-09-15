import copy
import numpy as np
import pytest
from look.studies.linear_protocol import protocol,matrix,validate,VERSION
from look.studies.linear_case import verify_case
from look.runtime.state import atomic_write_json,stable_hash,file_sha256


def test_scope_replicates_without_positive_score_and_rejects_false_receipt(tmp_path):
    assert len(matrix())==81
    p=protocol();assert len(p['arms'])==3 and p['additional_neural_training'] is False
    spec=dict(schema=VERSION,protocol=p,host=matrix()[0],test_access=False)
    validate(spec);spec=copy.deepcopy(spec);spec['host']['seed']=3417
    with pytest.raises(ValueError,match='pilot'):validate(spec)
    spec['pilot']={'run_dir':'a','sha256':'b'};validate(spec)
    atomic_write_json(dict(schema=VERSION,identity=stable_hash(spec),state='accepted',files={}),tmp_path/'accepted.json')
    with pytest.raises(ValueError):verify_case(tmp_path,spec)
    spec['test_access']=True
    with pytest.raises(ValueError):validate(spec)


def test_comparisons_have_declared_directions_and_zero_variance(tmp_path):
    from look.analysis.linear_report import report
    rows=[]
    for m in protocol()['arms']:
        for pattern in protocol()['patterns']:
            path=tmp_path/f'{m}_{pattern}.npz'
            np.savez(path,participant_ids=np.array(['a','b','c','d']),labels=np.array([0,1,0,1]),logits=np.array([[1.,0],[0,1],[1,0],[0,1]]))
            rows.append(dict(method=m,scenario=pattern,path=str(path),sha256=file_sha256(path),metrics={'macro_f1':1.}))
    r=report(rows,tmp_path/'report')
    assert len(r['definitions'])==9
    assert [(d['first'],d['second']) for d in r['definitions'][::3]]==[tuple(c) for c in protocol()['comparisons']]
    assert all(c['zero_variance'] and c['simultaneous_95'] is None for c in r['contrasts'])
    with pytest.raises(ValueError,match='Incomplete'):report(rows[:-1],tmp_path/'partial')


def test_feed_preserves_identity_and_rejects_test(tmp_path):
    from look.runtime.project_dispatch import work
    spec=tmp_path/'spec.json';atomic_write_json({'version':'test'},spec)
    row=dict(spec=str(spec),spec_sha256=file_sha256(spec),run_dir=str(tmp_path/'run'),id='x')
    feed=tmp_path/'linear.json';atomic_write_json(dict(schema='look_linear_work_feed_v1',test_access=False,tasks=[row,row]),feed)
    config=dict(project_feed=str(tmp_path/'none'),native_feed=str(tmp_path/'none'),linear_feeds=[str(feed)])
    tasks=work(config);assert len(tasks)==1 and tasks[0]['execution']=='look_linear'
    atomic_write_json(dict(schema='look_linear_work_feed_v1',test_access=True,tasks=[row]),feed)
    with pytest.raises(ValueError,match='Unsealed'):work(config)
