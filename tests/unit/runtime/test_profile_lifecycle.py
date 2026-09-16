import json
import subprocess
import pytest
from look.runtime.profile_lifecycle import run_profile, request_expiry_pause, profile_pause_state


def test_pause_is_not_a_failed_worker(monkeypatch,tmp_path):
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,75))
    with pytest.raises(SystemExit) as e:run_profile(['worker'],{})
    assert e.value.code==75
    profile=tmp_path/'run/resource_profile';profile.mkdir(parents=True)
    (profile/'status.json').write_text(json.dumps(dict(state='paused')))
    profile_pause_state(tmp_path/'run',profile)
    r=json.loads((tmp_path/'run/status.json').read_text())
    assert r['state']=='paused' and r['scientific_acceptance'] is False


def test_deterministic_failure_not_automatically_resumed(monkeypatch):
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,1))
    with pytest.raises(subprocess.CalledProcessError):run_profile(['worker'],{})


def test_both_formal_and_profile_get_expiry_request(tmp_path):
    run=tmp_path/'run';attempt=tmp_path/'attempt'
    request_expiry_pause(run,attempt)
    assert all((p/'pause.json').exists() for p in [run,run/'resource_profile',attempt/'profile'])
    with pytest.raises(ValueError):profile_pause_state(run,run/'resource_profile')


def test_profile_migration_requires_closed_identical_writer(tmp_path):
    import fcntl
    from look.runtime.profile_lifecycle import adopt_profile
    source=tmp_path/'source';source.mkdir();destination=tmp_path/'run/resource_profile'
    spec={'identity':'one'}
    (source/'spec.json').write_text(json.dumps(spec))
    (source/'status.json').write_text(json.dumps({'state':'paused'}))
    with (source/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):adopt_profile(source,destination,spec)
    with pytest.raises(ValueError):adopt_profile(source,destination,{'identity':'two'})
    adopt_profile(source,destination,spec);adopt_profile(source,destination,spec)
    assert destination.resolve()==source
    assert json.loads((destination.parent/'resource_profile_migration.json').read_text())['scientific_acceptance'] is False


def test_running_profile_and_existing_destination_are_protected(tmp_path):
    from look.runtime.profile_lifecycle import adopt_profile
    source=tmp_path/'source';source.mkdir();destination=tmp_path/'destination'
    (source/'spec.json').write_text('{}');(source/'status.json').write_text('{"state":"running"}')
    with pytest.raises(ValueError):adopt_profile(source,destination,{})
    (source/'status.json').write_text('{"state":"completed"}');destination.mkdir()
    with pytest.raises(ValueError):adopt_profile(source,destination,{})


def test_expiry_guard_requests_pause_then_adopts_without_stopping_other_work(tmp_path,monkeypatch):
    from look.runtime import qualification_guard as guard
    from look.runtime.state import file_sha256
    source=tmp_path/'profile';source.mkdir();spec=tmp_path/'spec.json';spec.write_text('{}')
    (source/'spec.json').write_text('{}');(source/'status.json').write_text('{"state":"running"}')
    cfg=dict(profile=str(source),run_dir=str(tmp_path/'formal'),output=str(tmp_path/'guard'),
             spec=str(spec),spec_sha256=file_sha256(spec),pause_at=0,deadline=100)
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    monkeypatch.setattr(guard.time,'time',lambda:10)
    def pause_then_close(_):
        assert (source/'pause.json').exists()
        (source/'status.json').write_text('{"state":"paused"}')
    monkeypatch.setattr(guard.time,'sleep',pause_then_close)
    guard.run(path)
    assert (tmp_path/'formal/resource_profile').resolve()==source
    result=json.loads((tmp_path/'guard/status.json').read_text())
    assert result['state']=='resumable' and result['scientific_acceptance'] is False
