import pytest
from look.runtime.finite_search_continuation import recoverable


def test_live_or_unknown_does_not_recover():
    record=dict(job_id='1',step='2',state='running')
    for alive in (True,None):
        assert not recoverable(record,lambda *a:alive,lambda *a:'TIMEOUT')


def test_only_lease_loss_or_clean_pause_recovers():
    record=dict(job_id='1',step='2',state='running')
    for state in ('TIMEOUT','PREEMPTED','NODE_FAIL'):
        assert recoverable(record,lambda *a:False,lambda *a:state)
    for state in ('RUNNING','COMPLETED','CANCELLED','FAILED','UNKNOWN'):
        assert not recoverable(record,lambda *a:False,lambda *a:state)
    assert recoverable(dict(record,state='paused'),lambda *a:False,lambda *a:'RUNNING')
    with pytest.raises(ValueError):recoverable(dict(record,step=None),lambda *a:False,lambda *a:'TIMEOUT')
    with pytest.raises(ValueError):recoverable(dict(record,state='completed'),lambda *a:False,lambda *a:'TIMEOUT')


def test_dead_paused_run_reuses_identity_and_new_attempt(tmp_path,monkeypatch):
    import json
    import look.runtime.finite_search_continuation as module
    import look.studies.search_case as case
    import sys
    from types import SimpleNamespace
    class Claims:
        def __init__(self,root):
            self.root=__import__("pathlib").Path(root);self.root.mkdir(exist_ok=True)
        def path(self,run):return self.root/"claim.json"
        def mutate(self,run,fn):
            value=fn(json.loads(self.path(run).read_text()));self.path(run).write_text(json.dumps(value));return value
    policy=SimpleNamespace(Claims=Claims)
    liveness=SimpleNamespace(step_presence=lambda *a:False)
    monkeypatch.setitem(sys.modules,"scheduling",SimpleNamespace(policy=policy,slurm_liveness=liveness))
    monkeypatch.setitem(sys.modules,"scheduling.policy",policy)
    monkeypatch.setitem(sys.modules,"scheduling.slurm_liveness",liveness)
    from look.runtime.state import file_sha256
    run=tmp_path/'run';run.mkdir();spec=tmp_path/'spec.json';spec.write_text('{}')
    manager=tmp_path/'old';manager.mkdir()
    (manager/'launch.json').write_text(json.dumps(dict(owner='old',generation=1)))
    (run/'pause.json').write_text(json.dumps(dict(reason='allocation_expiry')))
    out=tmp_path/'continuation';config=tmp_path/'config.json'
    env=dict(PYTHONPATH='',MALLOC_ARENA_MAX='1',MALLOC_TRIM_THRESHOLD_='131072',MALLOC_MMAP_THRESHOLD_='131072')
    cfg=dict(job='old_job',manager=str(manager),claims=str(tmp_path/'claims'),python='python',env=env,
             task=dict(run_dir=str(run),spec=str(spec),spec_sha256=file_sha256(spec)))
    config.write_text(json.dumps(cfg));claim=dict(owner='old',generation=1,job_id='old_job',step='3',state='paused',spec_sha256=file_sha256(spec))
    claims=policy.Claims(cfg['claims']);claims.path(run).write_text(json.dumps(claim))
    monkeypatch.setattr(liveness,'step_presence',lambda *a:False)
    monkeypatch.setattr(module,'allocation_state',lambda *a:'TIMEOUT')
    monkeypatch.setattr(case,'dependencies',lambda *a:None)
    def launch(args,**kwargs):
        renewed=json.loads(__import__('pathlib').Path(args[-1]).read_text())
        assert renewed['job']=='new_job' and renewed['task']==cfg['task'] and renewed['env']==env
        assert renewed['manager']!=str(manager) and not (run/'pause.json').exists()
        return type('Exit',(),dict(returncode=75))()
    monkeypatch.setattr(module.subprocess,'run',launch)
    binding=dict(config=str(config),pins={str(config):file_sha256(config)},output=str(out),manager_program='accepted_manager.py')
    assert module.execute(binding,'new_job',out)==75
    assert len(list((out/'attempts').glob('*/recovery.json')))==1
    assert len(list((out/'attempts').glob('*/previous_pause.json')))==1
    assert json.loads(config.read_text())==cfg
