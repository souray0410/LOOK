"""Real CPU child processes verify runtime changes independently of CUDA math."""
import importlib.util
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import pytest
from look_core.dual_queue import atomic,read,record

path=Path(__file__).resolve().parents[1]/'operations/flexible_queue.py'
spec=importlib.util.spec_from_file_location('flexible_queue',path)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def devices(*ids):
    return {i:dict(uuid=f'GPU-{i}',free_mib=20000,look_mib=0,unknown_mib=0,other_mib=8000) for i in ids}


def test_any_noncontiguous_devices_and_pause(tmp_path):
    for ids in ([3],[7,2],[1,3,5],[]):
        p=q.set_devices(tmp_path/'control.json',ids,snapshot=lambda:devices(1,2,3,5,7))
        assert q.validate_policy(p,devices(1,2,3,5,7))==ids
    with pytest.raises(ValueError):q.set_devices(tmp_path/'c',[3,3],snapshot=lambda:devices(3))
    with pytest.raises(ValueError):q.set_devices(tmp_path/'c',[0],snapshot=lambda:devices(3))
    with pytest.raises(ValueError):q.set_devices(tmp_path/'c',[-1],snapshot=lambda:devices(3))
    with pytest.raises(ValueError):q.set_devices(tmp_path/'c',[True],snapshot=lambda:devices(1))


def test_device_uuid_mismatch_rejected(tmp_path):
    p=q.set_devices(tmp_path/'c',[3],snapshot=lambda:devices(3))
    s=devices(3);s[3]['uuid']='GPU-new'
    with pytest.raises(ValueError):q.validate_policy(p,s)


def test_shared_environment_is_not_project_ownership(tmp_path):
    def process(pid,args,env=None):
        d=tmp_path/str(pid);d.mkdir();(d/'cmdline').write_bytes(args)
        if env is not None:(d/'environ').write_bytes(env)
    exe=b'/home/mengh/LOOK/old/tool/environment/.venv/bin/python\0'
    process(1,exe+b'-m\0radonbridge.experiment\0',b'')
    process(2,exe+b'/home/mengh/LOOK/release/pipeline/run.py\0')
    process(3,exe+b'-m\0arbitrary\0',b'LOOK_BOUNDED_WORKER=1\0')
    process(4,b'')
    assert [q.classify_process(i,tmp_path) for i in range(1,5)]==['other','look','look','unknown']


def test_project_aggregate_separate_from_unknown_and_external():
    with patch('subprocess.check_output',side_effect=['3, GPU-3, 15000\n',
        'GPU-3, 1, 7000\nGPU-3, 2, 6000\nGPU-3, 3, 8000\nGPU-3, 4, 1000\n']), \
        patch.object(q,'classify_process',side_effect=['look','look','other','unknown']):
        s=q.gpu_snapshot()[3]
    assert s['look_mib']==13000 and s['other_mib']==8000 and s['unknown_mib']==1000


def test_environment_exposes_exactly_one_uuid():
    for gpu in (0,3,17):
        with patch.dict(os.environ,{'RANK':'2','WORLD_SIZE':'4'}):
            env=q.worker_environment(gpu,Path('/frozen'),8,f'GPU-{gpu}')
        assert env['CUDA_VISIBLE_DEVICES']==f'GPU-{gpu}' and env['LOOK_PHYSICAL_GPU']==str(gpu)
        assert 'RANK' not in env and 'WORLD_SIZE' not in env
        assert env['PYTHONPATH']=='/frozen/tool'


def setup_queue(tmp_path,n=4):
    root=tmp_path/'worker';root.mkdir();atomic({'files':[]},root/'file_manifest.json')
    entry=tmp_path/'fake.py'
    entry.write_text('''import argparse,json,time,os,hashlib,pathlib
p=argparse.ArgumentParser()
for arg in ('worker-root','plan','output','worker'):p.add_argument('--'+arg)
a=p.parse_args();out=pathlib.Path(a.output);j=next(j for j in json.loads(pathlib.Path(a.plan).read_text())['jobs'] if j['id']==a.worker)
d=out/'cases'/a.worker;d.mkdir(parents=True,exist_ok=True)
with (d/'attempts.jsonl').open('a') as f:f.write(json.dumps(dict(gpu=os.environ['LOOK_PHYSICAL_GPU'],pid=os.getpid()))+'\\n')
beg=time.time();time.sleep(.35)
r=d/'result.json';r.write_text(json.dumps(dict(gpu=os.environ['LOOK_PHYSICAL_GPU'],pid=os.getpid(),start=beg,end=time.time())))
rec=dict(identity=json.loads((out/'queue_identity.json').read_text()),job_sha256=hashlib.sha256(json.dumps(j,sort_keys=True,separators=(',',':')).encode()).hexdigest(),status='complete',test_access=False,artifacts=[dict(path=str(r),bytes=r.stat().st_size,sha256=hashlib.sha256(r.read_bytes()).hexdigest())])
(d/'queue_complete.partial').write_text(json.dumps(rec));(d/'queue_complete.partial').replace(d/'queue_complete.json')
''')
    plan=tmp_path/'plan.json'
    atomic(dict(protocol='independent_single_gpu_cases_v1',test_access=False,acceptance_only=True,
                jobs=[dict(id=f'j{i}',kind='acceptance',seed=3407+i,source={}) for i in range(n)]),plan)
    return root,plan,tmp_path/'out',tmp_path/'lock',tmp_path/'control.json',entry


def results(out):return [read(p) for p in sorted(out.glob('cases/*/result.json'))]


def test_actual_single_gpu_then_resume_receipts_without_rewriting(tmp_path):
    args=setup_queue(tmp_path,2);root,pp,out,lock,control,entry=args
    q.set_devices(control,[3],snapshot=lambda:devices(3))
    state=q.supervise(*args,snapshot=lambda:devices(3),poll=.02)
    assert state['status']=='complete' and len(state['completed'])==2
    r=results(out);assert all(x['gpu']=='3' for x in r) and r[1]['start']>r[0]['end']
    hashes={str(p):record(p) for p in out.glob('cases/*/queue_complete.json')}
    q.set_devices(control,[3,7],snapshot=lambda:devices(3,7))
    q.supervise(*args,snapshot=lambda:devices(3,7),poll=.02)
    assert hashes=={str(p):record(p) for p in out.glob('cases/*/queue_complete.json')}


@pytest.mark.parametrize('mode',['drain','interrupt'])
def test_dual_to_single_to_dual_does_not_interrupt_retained_case(tmp_path,mode):
    args=setup_queue(tmp_path,6);_,_,out,_,control,_=args
    q.set_devices(control,[3,7],snapshot=lambda:devices(3,7));changed=False;added=False
    def snapshot():
        nonlocal changed,added
        s=devices(3,7)
        attempts=list(out.glob('cases/*/attempts.jsonl'))
        if len(attempts)>=2 and not changed:
            q.set_devices(control,[3],mode,snapshot=lambda:s);changed=True
        if len(list(out.glob('cases/*/result.json')))>=2 and not added:
            q.set_devices(control,[3,7],snapshot=lambda:s);added=True
        return s
    state=q.supervise(*args,snapshot=snapshot,poll=.02)
    assert state['status']=='complete' and changed and added
    first=(out/'cases/j0/attempts.jsonl').read_text().splitlines()
    assert len(first)==1 # GPU 3 was retained throughout, with the same process.
    removed=(out/'cases/j1/attempts.jsonl').read_text().splitlines()
    assert len(removed)==(2 if mode=='interrupt' else 1)
    assert len(state['completed'])==6
    assert any(r['gpu']=='7' for r in results(out)[2:])


def test_three_devices_overlap_and_pause_resume(tmp_path):
    args=setup_queue(tmp_path,3);_,_,out,_,control,_=args
    q.set_devices(control,[],snapshot=lambda:devices(2,5,8));calls=0
    def snapshot():
        nonlocal calls
        calls+=1
        if calls==4:q.set_devices(control,[2,5,8],snapshot=lambda:devices(2,5,8))
        return devices(2,5,8)
    state=q.supervise(*args,snapshot=snapshot,poll=.02)
    r=results(out)
    assert {x['gpu'] for x in r}=={'2','5','8'}
    assert max(x['start'] for x in r)<min(x['end'] for x in r)
    assert state['max_concurrent_cases']==3


def test_invalid_control_stops_dispatch_without_killing(tmp_path):
    args=setup_queue(tmp_path,2);_,_,out,_,control,_=args
    good=q.set_devices(control,[3],snapshot=lambda:devices(3));invalid=False;restored=False
    def snapshot():
        nonlocal invalid,restored
        if (out/'cases/j0/attempts.jsonl').exists() and not invalid:
            control.write_text('{bad');invalid=True
        if (out/'cases/j0/result.json').exists() and not restored:
            atomic(good,control);restored=True
        return devices(3)
    q.supervise(*args,snapshot=snapshot,poll=.02)
    assert invalid and restored and len((out/'cases/j0/attempts.jsonl').read_text().splitlines())==1


def test_external_and_unknown_memory_do_not_kill_active_case(tmp_path):
    args=setup_queue(tmp_path,1);_,_,out,_,control,_=args
    q.set_devices(control,[3],snapshot=lambda:devices(3))
    def snapshot():
        s=devices(3)
        if (out/'cases/j0/attempts.jsonl').exists():s[3].update(other_mib=18000,unknown_mib=1500,look_mib=1000)
        return s
    assert q.supervise(*args,snapshot=snapshot,poll=.02)['status']=='complete'


def test_project_budget_stops_only_own_process(tmp_path):
    args=setup_queue(tmp_path,1);_,_,out,_,control,_=args
    q.set_devices(control,[3],snapshot=lambda:devices(3))
    def snapshot():
        s=devices(3)
        if (out/'cases/j0/attempts.jsonl').exists():s[3]['look_mib']=15000
        return s
    with pytest.raises(RuntimeError,match='LOOK memory budget/OOM'):q.supervise(*args,snapshot=snapshot,poll=.02)
    assert read(out/'queue_status.json')['status']=='blocked'
