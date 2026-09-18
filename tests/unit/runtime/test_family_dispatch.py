import json
import sys
import types
import pytest
from pathlib import Path
from look.runtime import project_dispatch as m
from look.runtime.state import file_sha256


def test_family_feed_deduplicates_and_rejects_changes(tmp_path):
    spec=tmp_path/'spec.json';spec.write_text('{}')
    task=dict(id='family',spec=str(spec),spec_sha256=file_sha256(spec),run_dir=str(tmp_path/'run'))
    feed=tmp_path/'feed.json';feed.write_text(json.dumps(dict(schema='look_family_search_feed_v1',test_access=False,tasks=[task,task])))
    cfg=dict(project_feed=str(tmp_path/'none'),native_feed=str(tmp_path/'none2'),family_search_feeds=[str(feed)])
    assert m.work(cfg)==[dict(task,execution='look_family_search')]
    spec.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='specification changed'):m.work(cfg)


def test_family_verifier_not_pca_verifier(tmp_path,monkeypatch):
    spec=tmp_path/'spec.json';spec.write_text('{}');calls=[]
    module=types.ModuleType('look.studies.family_search_case')
    module.verify_case=lambda *a:calls.append(a)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    m.verify_delivery_dependency(dict(execution='look_family_search',spec=str(spec),run_dir='family'))
    assert calls==[('family',{})]


def test_family_profile_failure_never_executes_formal(tmp_path,monkeypatch):
    import look
    from look.runtime import profile_lifecycle
    source=Path(look.__file__).resolve().parent.parent
    sp=tmp_path/'spec.json';sp.write_text(json.dumps(dict(source_pins=[dict(path=str(source/'look/__init__.py'),sha256='pin')])) )
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(python=sys.executable)))
    monkeypatch.setenv('SLURM_JOB_ID','test');monkeypatch.setenv('SLURM_STEP_ID','0')
    observed=[]
    def profile(cmd,env):
        observed.append((cmd,env));raise RuntimeError('profile failed')
    monkeypatch.setattr(profile_lifecycle,'run_profile',profile)
    monkeypatch.setattr(m.os,'execvpe',lambda *a:pytest.fail('Must not launch formal'))
    with pytest.raises(RuntimeError,match='profile failed'):
        m.execute_work(config,sp,tmp_path/'run','look_family_search',tmp_path/'attempt/step.json')
    assert observed[0][0][2]=='look.studies.family_search_case'
    assert '--profile' in observed[0][0]
    assert observed[0][1]['PYTHONPATH'].split(':')[0]==str(source)


def test_deferred_search_does_not_remove_fitting_task(tmp_path,monkeypatch):
    tasks=[dict(execution='look_search',search_mode='best_forward',run_dir='old'),
           dict(execution='look_family_search',run_dir='family')]
    monkeypatch.setattr(m,'work',lambda c:tasks)
    monkeypatch.setattr(m,'active_search_modes',lambda *a:{})
    monkeypatch.setattr(m,'eligible',lambda *a:True)
    cfg=dict(deferred_search_modes=['best_forward'],output=str(tmp_path))
    assert m.admissible_work(cfg,object())==tasks[1:]
    assert json.loads((tmp_path/'api_admission.json').read_text())['rejected'][0]['reason']=='explicit_scientific_priority_amendment'
