import json
import sys
import types
import pytest
from look.runtime import project_dispatch as m


def test_priority_and_normal_dispatch_share_weekly_gate(tmp_path,monkeypatch):
    from look.runtime import weekly_delivery
    tasks=[dict(id='allowed',execution='look',run_dir='a'),
           dict(id='deferred',execution='look',run_dir='b')]
    monkeypatch.setattr(m,'work',lambda _:tasks)
    monkeypatch.setattr(m,'eligible',lambda *args:True)
    monkeypatch.setattr(weekly_delivery,'filter_tasks',lambda tasks,path:(tasks[:1],[dict(run='b',reason='weekly')]))
    cfg=dict(weekly_delivery_policy='policy')
    assert m.priority_work(cfg,object())==m.admissible_work(cfg,object())==tasks[:1]


def test_exclusive_claim_precedes_pause_and_failure_rolls_back(tmp_path,monkeypatch):
    events=[]
    package=types.ModuleType('mhd_models.scheduling');package.__path__=[]
    priority=types.ModuleType('mhd_models.scheduling.project_priority')
    priority.sha=lambda p:'digest'
    def pause(*args):
        events.append('pause')
        raise ValueError('parent completed before handover')
    priority.request_pause=pause
    monkeypatch.setitem(sys.modules,'mhd_models.scheduling',package)
    monkeypatch.setitem(sys.modules,'mhd_models.scheduling.project_priority',priority)
    class Claims:
        def acquire(self,*args):events.append('claim');return {'generation':1}
        def release(self,*args,**kwargs):events.append('release');assert kwargs['step_dead']
    target=dict(run_dir=str(tmp_path/'target'),spec_sha256='abc')
    qualified=(target,tmp_path/'profile','identity',tmp_path)
    with pytest.raises(ValueError,match='completed'):
        m.reserve_priority(qualified,Claims(),'owner','job',dict(run_dir=str(tmp_path/'native')))
    assert events==['claim','pause','release']


def test_lost_claim_does_not_pause_parent(tmp_path,monkeypatch):
    package=types.ModuleType('mhd_models.scheduling');package.__path__=[]
    priority=types.ModuleType('mhd_models.scheduling.project_priority')
    priority.sha=lambda p:'digest'
    priority.request_pause=lambda *args:pytest.fail('Claim not owned')
    monkeypatch.setitem(sys.modules,'mhd_models.scheduling',package)
    monkeypatch.setitem(sys.modules,'mhd_models.scheduling.project_priority',priority)
    class Claims:
        def acquire(self,*args):raise RuntimeError('already claimed')
    with pytest.raises(RuntimeError,match='already claimed'):
        m.reserve_priority((dict(run_dir='target',spec_sha256='x'),None,None,None),
                           Claims(),'owner','job',dict(run_dir='native'))


@pytest.mark.parametrize('status',['needs_review','failed','completed'])
def test_reserved_task_rechecks_status_before_execution(tmp_path,monkeypatch,status):
    run=tmp_path/'run';run.mkdir();(run/'status.json').write_text(json.dumps({'state':status}))
    token=dict(state='claimed',owner='same',generation=2,spec_sha256='abc',job_id='job')
    claim=tmp_path/'claim.json';claim.write_text(json.dumps(token))
    class Claims:
        def path(self,run):return claim
    task=dict(run_dir=str(run),execution='look',spec_sha256='abc')
    monkeypatch.setattr(m,'work',lambda _: [task])
    assert m.admissible_work({},Claims(),reservation=(task,token,{}))==[]
