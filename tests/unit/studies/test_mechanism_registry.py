import json
from pathlib import Path
import sys
import types
from look.runtime.state import atomic_write_json,file_sha256
from look.studies.mechanism_registry import tick


def test_one_host_releases_only_its_dependencies_and_is_idempotent(tmp_path,monkeypatch):
    # The existing shared registrar is covered by Model_Training's process tests;
    # this test checks the supplement's dependency/identity contract at that API.
    def reserve(root,namespace,task_id,configuration,**kwargs):
        p=Path(root)/'runs'/task_id;p.mkdir(parents=True,exist_ok=True);return p
    monkeypatch.setitem(sys.modules,'mhd_models.runtime.run_registry',types.SimpleNamespace(reserve=reserve))
    from look.studies import project_case
    monkeypatch.setattr(project_case,'verify_case',lambda root,spec:json.loads((Path(root)/'accepted.json').read_text()))
    run=tmp_path/'source';(run/'host').mkdir(parents=True);best=run/'host/best.pt';best.write_bytes(b'fixture')
    spec=run/'spec.json';atomic_write_json(dict(disease='glaucoma',model=dict(name='resnet50'),position='middle',seed=3416),spec)
    atomic_write_json(dict(files={'host/best.pt':file_sha256(best)}),run/'accepted.json')
    base=tmp_path/'base.json';atomic_write_json(dict(schema='look_project_work_feed_v1',test_access=False,
        tasks=[dict(id='base',run_dir=str(run),spec=str(spec),spec_sha256=file_sha256(spec))]),base)
    cfg=dict(output=str(tmp_path/'supplement'),base_feed=str(base),source_pins=[])
    first=tick(cfg);second=tick(cfg)
    assert first['registered']==second['registered']==90
    assert second['waiting_dependencies']==7092 and second['accepted_base_hosts']==1
    test=json.loads((tmp_path/'supplement/report/test_preparation.json').read_text())
    assert not test['all_training_resolved'] and not test['test_access']
