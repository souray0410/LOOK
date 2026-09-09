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


@pytest.mark.parametrize('flexible',[False,True])
def test_selected_worker_never_writes_its_input_result(tmp_path, monkeypatch, flexible):
    from look_core.dual_queue import atomic,read,record,digest,worker
    import look_core.dual_queue as queue
    import look_core.bounded_runtime as runtime
    import look_core.start_study as starts
    import torch
    root=tmp_path/'source';root.mkdir();out=tmp_path/'output';out.mkdir()
    original=tmp_path/'historical_result.json';atomic({'status':'complete','original':True},original)
    before=record(original)
    source={k:before for k in ('result','manifest','checkpoint','pca_manifest','labels','natural_labels')}
    source.update(selection={'seed':3407},pcas=[],generators={})
    source_path=tmp_path/'source.json';atomic(source,source_path)
    rows=[dict(case={'context':'ctx','start_ordinal':i},result=before) for i in range(1,10)]
    job=dict(id='select',kind='selected',seed=3407,context='ctx',source=record(source_path),records=rows,after=[])
    plan=dict(protocol='independent_single_gpu_cases_v1',test_access=False,jobs=[job]);pp=tmp_path/'plan.json';atomic(plan,pp)
    atomic(dict(plan_sha256=digest(plan),source_manifest_sha256='code'),out/'queue_identity.json')
    if flexible:
        import importlib.util
        from pathlib import Path
        path=Path(__file__).resolve().parents[1]/'operations/flexible_worker.py'
        spec=importlib.util.spec_from_file_location('tested_flexible_worker',path)
        queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
        worker=queue.worker
        monkeypatch.setattr(runtime,'install_allocator_limits',runtime.install_allocator_limits)
    monkeypatch.setattr(queue,'check_source',lambda _: 'code')
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(runtime,'install_runtime',lambda:None)
    monkeypatch.setattr(starts,'verify_source',lambda _:None)
    monkeypatch.setattr(starts,'audit_case',lambda c,p:c)
    def evaluate(context,rows,source,destination,device,devices):
        p=destination/'selected_starts'/context/'validation_result.json'
        atomic(dict(status='complete',phase='validation',test_access=False),p);return record(p)
    monkeypatch.setattr(starts,'evaluate_selected',evaluate)
    for key,value in dict(LOOK_PHYSICAL_GPU='3' if flexible else '0',LOOK_ASSIGNED_GPU_UUID='GPU-test',CUDA_VISIBLE_DEVICES='GPU-test',LOOK_EXECUTION_MICROBATCH='8').items():monkeypatch.setenv(key,value)
    worker(root,pp,out,'select')
    assert record(original)==before
    assert read(out/'cases/select/queue_complete.json')['result']['path'] != str(original)
