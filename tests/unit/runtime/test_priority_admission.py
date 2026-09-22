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
    package=types.ModuleType('scheduling');package.__path__=[]
    priority=types.ModuleType('scheduling.project_priority')
    priority.sha=lambda p:'digest'
    def pause(*args):
        events.append('pause')
        raise ValueError('parent completed before handover')
    priority.request_pause=pause
    monkeypatch.setitem(sys.modules,'scheduling',package)
    monkeypatch.setitem(sys.modules,'scheduling.project_priority',priority)
    class Claims:
        def acquire(self,*args):events.append('claim');return {'generation':1}
        def release(self,*args,**kwargs):events.append('release');assert kwargs['step_dead']
    target=dict(run_dir=str(tmp_path/'target'),spec_sha256='abc')
    qualified=(target,tmp_path/'profile','identity',tmp_path)
    with pytest.raises(ValueError,match='completed'):
        m.reserve_priority(qualified,Claims(),'owner','job',dict(run_dir=str(tmp_path/'native')))
    assert events==['claim','pause','release']


def test_lost_claim_does_not_pause_parent(tmp_path,monkeypatch):
    package=types.ModuleType('scheduling');package.__path__=[]
    priority=types.ModuleType('scheduling.project_priority')
    priority.sha=lambda p:'digest'
    priority.request_pause=lambda *args:pytest.fail('Claim not owned')
    monkeypatch.setitem(sys.modules,'scheduling',package)
    monkeypatch.setitem(sys.modules,'scheduling.project_priority',priority)
    class Claims:
        def acquire(self,*args):raise RuntimeError('already claimed')
    with pytest.raises(RuntimeError,match='already claimed'):
        m.reserve_priority((dict(run_dir='target',spec_sha256='x'),None,None,None),
                           Claims(),'owner','job',dict(run_dir='native'))
