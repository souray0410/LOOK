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
