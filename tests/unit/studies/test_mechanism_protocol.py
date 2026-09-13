import copy
from collections import Counter
import pytest
from look.studies.mechanism_protocol import matrix,protocol,validate_task,VERSION


def test_finite_counts_and_validation():
    tasks=matrix();counts=Counter(t['kind'] for t in tasks)
    assert counts['host_training']==162
    assert counts['student_training']==216
    assert counts['correction']==1620
    assert counts['sample_curve']==5184
    assert len(tasks)==7182
    assert len({t['id'] for t in tasks})==7182
    spec=dict(schema=VERSION,test_access=False,task=tasks[0],protocol=protocol())
    validate_task(spec)
    bad=copy.deepcopy(spec);bad['task']['host']['seed']=3420
    with pytest.raises(ValueError):validate_task(bad)
    bad=copy.deepcopy(spec);bad['test_access']=True
    with pytest.raises(ValueError):validate_task(bad)
    assert all(t.get('repeat')==0 for t in tasks if t.get('fraction')==1.)
