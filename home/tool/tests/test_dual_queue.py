import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import pytest
from look_core.dual_queue import (validate_plan, ready_jobs, worker_environment, atomic,
    record, digest, validate_receipt, gpu_snapshot, supervise)


def job(name, after=()):
    return dict(id=name, kind='acceptance', seed=3407, after=list(after), source={})


def plan(jobs):
    return dict(protocol='independent_single_gpu_cases_v1', test_access=False,
                acceptance_only=True, jobs=jobs)


def test_dependency_waves_and_no_duplicate_claim():
    jobs=[job('s1'),job('s2'),job('next', ['s1','s2'])]
    validate_plan(plan(jobs))
    assert [j['id'] for j in ready_jobs(jobs, {}, {})] == ['s1','s2']
    assert ready_jobs(jobs, {'s1':{}}, {'s2'}) == []
    assert [j['id'] for j in ready_jobs(jobs, {'s1':{},'s2':{}}, {})] == ['next']


@pytest.mark.parametrize('jobs', [[job('x'),job('x')], [job('../x')], [job('x',['missing'])]])
def test_invalid_queue_rejected(jobs):
    with pytest.raises(ValueError): validate_plan(plan(jobs))


def test_no_implicit_search_and_no_test():
    assert validate_plan(plan([])) == []
    p=plan([]);p['test_access']=True
    with pytest.raises(ValueError): validate_plan(p)


def test_worker_sees_one_physical_gpu_and_no_ddp():
    with patch.dict(os.environ, {'RANK':'1', 'LOCAL_RANK':'1', 'WORLD_SIZE':'2'}):
        for g in (0,1):
            env=worker_environment(g,Path('/source'),8)
            assert env['CUDA_VISIBLE_DEVICES']==str(g)
            assert not {'RANK','LOCAL_RANK','WORLD_SIZE'}.intersection(env)
    with pytest.raises(ValueError): worker_environment(2,Path('/source'),8)


def test_receipt_guards_identity_and_result_tampering(tmp_path):
    artifact=tmp_path/'result.json';artifact.write_text('{}')
    j=job('a');p=tmp_path/'receipt.json'
    atomic(dict(identity={'code':'1'},job_sha256=digest(j),status='complete',
                test_access=False,artifacts=[record(artifact)]),p)
    validate_receipt(p,{'code':'1'},j)
    with pytest.raises(ValueError):validate_receipt(p,{'code':'2'},j)
    artifact.write_text('{"changed":1}')
    with pytest.raises(ValueError):validate_receipt(p,{'code':'1'},j)


def test_all_look_processes_counted_per_card_and_other_project_ignored():
    devices='0, GPU-a, 20000\n1, GPU-b, 21000\n'
    rows='GPU-a, 10, 7000\nGPU-a, 11, 8000\nGPU-b, 12, 9000\n'
    def proc(p):
        if p.name=='environ': return b'LOOK_BOUNDED_WORKER=1' if p.parent.name in ('10','11') else b''
        return b'python\0-m\0radonbridge.experiment\0'
    with patch('subprocess.check_output',side_effect=[devices,rows]), patch.object(Path,'read_bytes',proc):
        s=gpu_snapshot()
    assert s[0]['look_mib']==15000 and s[1]['look_mib']==0


def test_process_exit_during_nvml_snapshot(tmp_path):
    with patch('subprocess.check_output',side_effect=['0, GPU-a, 20000\n1, GPU-b, 20000\n','GPU-a, 99, 5000\n']), patch.object(Path,'read_bytes',side_effect=FileNotFoundError):
        assert gpu_snapshot()[0]['look_mib']==0


def test_real_supervisor_two_children_resume_and_resource_gate(tmp_path):
    """CPU subprocesses exercise the real scheduling/receipt loop, without CUDA."""
    root=tmp_path/'source';(root/'pipeline').mkdir(parents=True)
    worker=root/'pipeline/46_run_dual_gpu_queue.py'
    worker.write_text('''import argparse,json,time,os,hashlib,pathlib
p=argparse.ArgumentParser()
for arg in ('project-root','plan','output','worker'):p.add_argument('--'+arg)
a=p.parse_args();out=pathlib.Path(a.output);j=next(j for j in json.loads(pathlib.Path(a.plan).read_text())['jobs'] if j['id']==a.worker)
d=out/'cases'/a.worker;d.mkdir(parents=True,exist_ok=True)
beg=time.time();time.sleep(.25)
r=d/'result.json';r.write_text(json.dumps(dict(gpu=os.environ['CUDA_VISIBLE_DEVICES'],start=beg,end=time.time())))
rec=dict(identity=json.loads((out/'queue_identity.json').read_text()),job_sha256=hashlib.sha256(json.dumps(j,sort_keys=True,separators=(',',':')).encode()).hexdigest(),status='complete',test_access=False,artifacts=[dict(path=str(r),bytes=r.stat().st_size,sha256=hashlib.sha256(r.read_bytes()).hexdigest())])
(d/'queue_complete.partial').write_text(json.dumps(rec));(d/'queue_complete.partial').replace(d/'queue_complete.json')
''')
    atomic(dict(files=[dict(path='pipeline/'+worker.name,bytes=worker.stat().st_size,
                           sha256=record(worker)['sha256'])]),root/'file_manifest.json')
    pp=tmp_path/'plan.json';atomic(plan([job('s1'),job('s2'),job('s3',['s1','s2'])]),pp)
    calls=[]
    def snapshot():
        calls.append(1)
        return {g:dict(free_mib=0 if len(calls)==1 else 20000,look_mib=0) for g in (0,1)}
    out=tmp_path/'out';lock=tmp_path/'global.lock'
    state=supervise(root,pp,out,lock,snapshot=snapshot,poll=.02)
    assert state['status']=='complete' and len(state['completed'])==3
    records=[json.loads((out/'cases'/f's{i}'/'result.json').read_text()) for i in (1,2,3)]
    assert records[0]['gpu']=='0' and records[1]['gpu']=='1'
    assert max(r['start'] for r in records[:2]) < min(r['end'] for r in records[:2])
    assert records[2]['start'] > max(r['end'] for r in records[:2])
    original=(out/'cases/s1/result.json').stat().st_mtime_ns
    supervise(root,pp,out,lock,snapshot=snapshot,poll=.02)
    assert (out/'cases/s1/result.json').stat().st_mtime_ns==original


def test_duplicate_scientific_case_rejected():
    j=dict(id='one',kind='method',seed=3407,source={},case={'method':'ssf'},spec={})
    p=plan([j,dict(j,id='two')]);p['acceptance_only']=False
    with pytest.raises(ValueError,match='Duplicate scientific'):validate_plan(p)


def test_budget_violation_stops_owned_worker_and_preserves_failure(tmp_path):
    root=tmp_path/'source';(root/'pipeline').mkdir(parents=True)
    script=root/'pipeline/46_run_dual_gpu_queue.py'
    script.write_text('import time\ntime.sleep(30)\n')
    atomic(dict(files=[dict(path='pipeline/'+script.name,bytes=script.stat().st_size,
                           sha256=record(script)['sha256'])]),root/'file_manifest.json')
    pp=tmp_path/'plan.json';atomic(plan([job('only')]),pp)
    calls=[]
    def snapshot():
        calls.append(1)
        return {0:dict(free_mib=20000,look_mib=15000 if len(calls)>1 else 0),
                1:dict(free_mib=0,look_mib=0)}
    out=tmp_path/'out'
    with pytest.raises(RuntimeError,match='memory budget/OOM'):
        supervise(root,pp,out,tmp_path/'lock',snapshot=snapshot,poll=.02)
    state=json.loads((out/'queue_status.json').read_text())
    assert state['status']=='blocked' and not state['completed'] and not state['active']
    assert state['peaks_mib']['0']==15000


def test_global_lock_rejects_second_supervisor(tmp_path):
    import fcntl
    root=tmp_path/'src';root.mkdir();atomic({'files':[]},root/'file_manifest.json')
    pp=tmp_path/'p.json';atomic(plan([]),pp)
    lp=tmp_path/'global.lock'
    with lp.open('a') as locked:
        fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):supervise(root,pp,tmp_path/'out',lp)


def test_own_worker_does_not_require_racy_environment_access():
    def proc(p):
        if p.name=='environ':raise PermissionError('exiting task')
        return b'python\0/home/mengh/LOOK/release/pipeline/46_run_dual_gpu_queue.py\0'
    with patch('subprocess.check_output',side_effect=['0, GPU-a, 20000\n1, GPU-b, 20000\n','GPU-a, 99, 5000\n']), patch.object(Path,'read_bytes',proc):
        assert gpu_snapshot()[0]['look_mib']==5000


def test_unknown_exiting_process_is_conservatively_counted():
    def proc(p):
        if p.name=='environ':raise PermissionError('exiting task')
        return b''
    with patch('subprocess.check_output',side_effect=['0, GPU-a, 20000\n1, GPU-b, 20000\n','GPU-a, 99, 5000\n']), patch.object(Path,'read_bytes',proc):
        assert gpu_snapshot()[0]['look_mib']==5000
