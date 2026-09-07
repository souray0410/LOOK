import pytest
from look_core.resumed_starts import split_scope
from look_core.dual_queue import validate_plan,ready_jobs


def scope():
    return [dict(case_id=f'context_{i}_start_{j}',context=f'context_{i}',start_ordinal=j)
            for i in range(27) for j in range(1,10)]


def test_exact_remaining_without_duplicate_reused_work():
    cases=scope();done={c['case_id']:{} for c in cases[:61]}
    pending=split_scope(cases,done)
    assert len(pending)==182
    assert not set(done)&{c['case_id'] for c in pending}


def test_missing_or_duplicate_start_is_rejected():
    cases=scope();cases[-1]['start_ordinal']=8
    with pytest.raises(ValueError):split_scope(cases,{})
    with pytest.raises(ValueError):split_scope(scope()[:-1],{})


def test_selection_waits_for_its_unfinished_starts():
    jobs=[dict(id='start8',kind='suffix',seed=3407,after=[],source={},case={}),
          dict(id='select',kind='selected',seed=3407,after=['start8'],source={},records=[])]
    validate_plan(dict(protocol='independent_single_gpu_cases_v1',test_access=False,jobs=jobs))
    assert [j['id'] for j in ready_jobs(jobs,{},set())]==['start8']
    assert [j['id'] for j in ready_jobs(jobs,{'start8':{}},set())]==['select']
