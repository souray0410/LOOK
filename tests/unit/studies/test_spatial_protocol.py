import copy
import pytest
from look.studies.spatial_protocol import protocol,matrix,validate,VERSION
from look.studies.spatial_case import verify_case
from look.runtime.state import atomic_write_json,stable_hash


def test_scope_and_no_performance_gate(tmp_path):
    assert len(matrix())==81
    p=protocol();assert len(p['routes'])==3 and not p['additional_neural_training']
    s=dict(schema=VERSION,protocol=p,host=matrix()[0],test_access=False)
    validate(s)
    s=copy.deepcopy(s);s['host']['seed']=3417
    with pytest.raises(ValueError,match='pilot'):validate(s)
    s['pilot']={'run_dir':'x','sha256':'x'};validate(s)
    atomic_write_json(dict(schema=VERSION,identity=stable_hash(s),state='accepted',test_access=False,routes=p['routes'],files={}),tmp_path/'accepted.json')
    with pytest.raises(ValueError):verify_case(tmp_path,s)


def test_three_route_statistics_is_symmetric(tmp_path):
    import numpy as np
    from look.analysis.spatial_report import report
    from look.runtime.state import file_sha256
    rows=[]
    for m in protocol()['routes']:
        for pattern in ('oct_missing','cfp_missing'):
            path=tmp_path/f'{m}_{pattern}.npz'
            np.savez(path,labels=np.array([0,1,0,1]),participant_ids=np.array(['a','b','c','d']),logits=np.array([[1.,0],[0,1],[1,0],[0,1]]))
            rows.append(dict(method=m,scenario=pattern,path=str(path),sha256=file_sha256(path),metrics={'macro_f1':1.}))
    r=report(rows,tmp_path/'report')
    assert len(r['definitions'])==9
    assert all(c['zero_variance'] and c['simultaneous_95'] is None for c in r['contrasts'])
    assert all(c['difference']==0 for c in r['contrasts'])


def test_direct_pca_has_explicit_ram_class_without_changing_gpu_duration():
    from look.runtime.project_dispatch import allocation_command
    ordinary=allocation_command('/config','x','python')
    spatial=allocation_command('/config','x','python',512)
    assert '--mem=128G' in ordinary and '--mem=512G' in spatial
    assert '--time=48:00:00' in ordinary and '--time=48:00:00' in spatial
    assert '--gres=gpu:a100:1' in spatial
    with pytest.raises(ValueError):allocation_command('/config','x','python',999)
