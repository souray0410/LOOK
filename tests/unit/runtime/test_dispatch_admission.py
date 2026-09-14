import json
import fcntl
import pytest
import sys
import types
from look.runtime import project_dispatch as m
from look.runtime.state import file_sha256


@pytest.fixture(autouse=True)
def scheduler_boundary(monkeypatch):
    # Application tests do not depend on a sibling training checkout.
    package=types.ModuleType('scheduling');package.__path__=[]
    quota=types.ModuleType('scheduling.quota_guard');quota.snapshot=lambda:None
    policy=types.ModuleType('scheduling.policy');policy.Claims=lambda path:object()
    renewal=types.ModuleType('scheduling.renewal');renewal.job_from_log=lambda text:None
    for name,module in [('scheduling',package),('scheduling.quota_guard',quota),
                        ('scheduling.policy',policy),('scheduling.renewal',renewal)]:
        monkeypatch.setitem(sys.modules,name,module)


def test_busy_account_lock_waits_without_submission(tmp_path, monkeypatch):
    import scheduling.quota_guard as q
    monkeypatch.setattr(m,'work',lambda c:[{'id':'ready'}])
    monkeypatch.setattr(m,'eligible',lambda *a:True)
    monkeypatch.setattr(q,'snapshot',lambda:pytest.fail('Lock not acquired'))
    lock=tmp_path/'account.lock'
    with lock.open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert m.submit_one({'claims':str(tmp_path/'claims'),
            'account_submission_lock':str(lock)},'config',{'requests':[]})=='waiting_submission_lock'


def test_nonlock_resource_error_is_not_hidden(tmp_path,monkeypatch):
    import scheduling.quota_guard as q
    monkeypatch.setattr(m,'work',lambda c:[{'id':'ready'}])
    monkeypatch.setattr(m,'eligible',lambda *a:True)
    def fail():raise BlockingIOError(11,'fork unavailable')
    monkeypatch.setattr(q,'snapshot',fail)
    with pytest.raises(BlockingIOError):
        m.submit_one({'claims':str(tmp_path/'claims'),'account_submission_lock':str(tmp_path/'lock')},'config',{'requests':[]})


def test_additional_feed_reuses_run_without_double_claim(tmp_path):
    spec=tmp_path/'spec.json';spec.write_text(json.dumps({'test_used':False}))
    t={'id':'first','spec':str(spec),'spec_sha256':file_sha256(spec),'run_dir':str(tmp_path/'run')}
    feeds=[]
    for index in range(2):
        task=dict(t,id=str(index))
        queue=tmp_path/f'q{index}.json';queue.write_text(json.dumps({'tasks':[task]}))
        feed=tmp_path/f'f{index}.json';feed.write_text(json.dumps({'schema':"look_native_work_feed_v1",'test_access':False,
            'queues':[{'queue':str(queue),'queue_sha256':file_sha256(queue)}]}))
        feeds.append(str(feed))
    result=m.work({'project_feed':str(tmp_path/'no_project'),'native_feed':feeds[0],
                   'additional_native_feeds':[feeds[1]]})
    assert len(result)==1 and result[0]['id']=='0'


def test_short_worker_exit_and_unknown_liveness():
    seen=iter([True,None,False]);waits=[]
    assert m.exited_step('1','2',lambda *a:next(seen),sleep=waits.append)
    assert waits==[2,2]
    assert not m.exited_step('1',None,lambda *a:False)
    assert not m.exited_step('1','2',lambda *a:None,attempts=2,sleep=lambda _:None)


def test_nested_step_does_not_inherit_owner_cpu_request():
    original={'SLURM_JOB_ID':'1','SLURM_CPUS_PER_TASK':'1','SLURM_TRES_PER_TASK':'cpu=1','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2'}
    result=m.worker_environment(original)
    assert 'SLURM_CPUS_PER_TASK' not in result and 'SLURM_TRES_PER_TASK' not in result
    assert result['SLURM_JOB_ID']=='1' and result['CUDA_VISIBLE_DEVICES']=='0' and result['OMP_NUM_THREADS']=='2'
    assert original['SLURM_CPUS_PER_TASK']=='1'


def test_incompatible_probe_is_excluded_before_allocation(tmp_path,monkeypatch):
    tasks=[{'run_dir':'old','execution':'native'},{'run_dir':'ready','execution':'look'}]
    monkeypatch.setattr(m,'work',lambda _:tasks);monkeypatch.setattr(m,'eligible',lambda *a:True)
    def fail(*a):raise ValueError('unsupported pinned API')
    monkeypatch.setattr(m,'native_api_preflight',fail)
    assert m.admissible_work({'output':str(tmp_path)},object())==tasks[1:]
    assert json.loads((tmp_path/'api_admission.json').read_text())['rejected'][0]['run']=='old'


def test_incident_hold_prevents_repeated_submission(tmp_path):
    (tmp_path/'admission_hold.json').write_text('{}')
    assert m.submit_one({'output':str(tmp_path),'claims':str(tmp_path)},'config',{})=='waiting_incident_repair'
