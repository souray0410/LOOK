import json
from pathlib import Path
import pytest
from look.runtime.project_dispatch import work,eligible,allocation_command,source_binding
from look.runtime.state import file_sha256


def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x));return str(p)


def test_no_work_does_not_create_work_or_request(tmp_path):
    assert work(dict(project_feed=str(tmp_path/'project'),native_feed=str(tmp_path/'native')))==[]
    c=allocation_command('/fixed/config','look_auto_1','/fixed/python')
    assert '--gres=gpu:a100:1' in c and '--time=48:00:00' in c and '--mem=128G' in c
    assert 'scancel' not in c


def test_changed_feed_rejected_before_claim(tmp_path):
    spec=tmp_path/'spec';write(spec,{'test_used':False})
    feed=write(tmp_path/'project',dict(schema='look_project_work_feed_v1',test_access=False,
         tasks=[dict(id='x',spec=str(spec),spec_sha256='bad',run_dir=str(tmp_path/'run'))]))
    with pytest.raises(ValueError,match='specification'):
        work(dict(project_feed=feed,native_feed=str(tmp_path/'none')))


@pytest.mark.parametrize('state',['claimed','running','failed','completed','liveness_needs_review'])
def test_old_claim_cannot_be_stolen_even_when_stale(tmp_path,state):
    claim=tmp_path/'claim';write(claim,dict(state=state,updated_at=0))
    class Claims:
        def path(self,run):return claim
    assert not eligible(dict(run_dir=str(tmp_path/'run')),Claims())


def test_exact_native_source_binding(tmp_path):
    path=tmp_path/'source'/'workflows'/'native.py';path.parent.mkdir(parents=True);path.write_text('exact')
    framework=tmp_path/'framework'/'mhd_framework'/'core.py';framework.parent.mkdir(parents=True);framework.write_text('core')
    spec={'trainer_source_sha256':{'workflows/native.py':file_sha256(path)},'framework':{'api':'V5','commit':'fixed','source_sha256':{'core.py':file_sha256(framework)}}}
    config=dict(native_sources=[str(tmp_path/'source')],dependency_pythonpath='/dependencies',framework_pythonpaths={'fixed':str(tmp_path/'framework')})
    assert source_binding(spec,config)['source']==str(tmp_path/'source')
    path.write_text('changed')
    with pytest.raises(ValueError,match='No immutable'):source_binding(spec,config)


def test_proven_pre_submission_failure_is_not_ambiguous_request(tmp_path):
    from look.runtime.project_dispatch import failed_before_submission
    log=tmp_path/'allocation.log';log.write_text('salloc: error: no controlling terminal: please set --no-shell\n')
    assert failed_before_submission(dict(state='owner_exited',log=str(log)))
    log.write_text('salloc: Pending job allocation 123\n')
    assert not failed_before_submission(dict(state='owner_exited',log=str(log)))


def test_active_positive_tree_count(tmp_path,monkeypatch):
    import look.runtime.project_dispatch as module
    claim=tmp_path/'claim.json';write(claim,dict(state='running'))
    monkeypatch.setattr(module,'work',lambda config:[dict(execution='look_search',search_mode='positive_forward_tree',run_dir='tree')])
    class Claims:
        def path(self,run):return claim
    assert module.active_search_modes({},Claims())['positive_forward_tree']==1
