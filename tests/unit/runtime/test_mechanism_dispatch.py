import json
import pytest
from look.runtime.project_dispatch import work,eligible
from look.runtime.state import file_sha256,stable_hash
from look.studies.mechanism_case import verify_case
from look.studies.mechanism_protocol import VERSION,protocol,matrix,validate_task


def test_serialized_protocol_and_feed(tmp_path):
    spec=dict(schema=VERSION,protocol=protocol(),task=matrix()[0],test_access=False)
    validate_task(json.loads(json.dumps(spec)))
    p=tmp_path/'spec.json';p.write_text(json.dumps(spec));feed=tmp_path/'feed.json'
    task=dict(id=spec['task']['id'],spec=str(p),spec_sha256=file_sha256(p),run_dir=str(tmp_path/'run'))
    feed.write_text(json.dumps(dict(schema='look_mechanism_work_feed_v1',test_access=False,tasks=[task])))
    cfg=dict(project_feed=str(tmp_path/'empty'),native_feed=str(tmp_path/'empty'),supplement_feeds=[str(feed),str(feed)])
    assert len(work(cfg))==1 and work(cfg)[0]['execution']=='look_mechanism'
    p.write_text('{}')
    with pytest.raises(ValueError,match='specification'):work(cfg)


def test_false_completion_rejected(tmp_path):
    spec=dict(schema=VERSION,protocol=protocol(),task=matrix()[0],test_access=False)
    (tmp_path/'accepted.json').write_text(json.dumps(dict(schema=VERSION,identity=stable_hash(spec),test_access=False,state='accepted',files={})))
    with pytest.raises(ValueError,match='Incomplete'):verify_case(tmp_path,spec)


def test_paths_from_json_have_same_digest(tmp_path):
    from look.runtime.state import atomic_write_json
    p=tmp_path/'json_path.json';atomic_write_json({'x':1},str(p))
    assert file_sha256(str(p))==file_sha256(p)


def test_monitor_cache_rechecks_changed_evidence(tmp_path):
    from look.runtime.mechanism_receipts import monitored_acceptance
    from look.runtime.state import atomic_write_json
    root=tmp_path/'run';root.mkdir();artifact=root/'best.pt';artifact.write_bytes(b'best')
    atomic_write_json({'files':{'best.pt':file_sha256(artifact)}},root/'accepted.json')
    calls=[]
    def verifier(r,s):
        calls.append(1);value=json.loads((r/'accepted.json').read_text())
        if file_sha256(artifact)!=value['files']['best.pt']:raise ValueError('changed')
        return value
    for _ in range(2):monitored_acceptance(root,{},verifier,tmp_path/'cache')
    assert len(calls)==1
    artifact.write_bytes(b'bad')
    with pytest.raises(ValueError,match='changed'):monitored_acceptance(root,{},verifier,tmp_path/'cache')


def test_handover_refuses_gpu_owners_and_other_config():
    from look.runtime.mechanism_handover import validate_manager
    p=dict(args=['python','-m','look.runtime.project_dispatch','--config','exact.json'])
    validate_manager(p,'exact.json')
    for flag in ('--allocation-owner','--gpu-owner','--execute'):
        with pytest.raises(ValueError):validate_manager(dict(args=p['args']+[flag]),'exact.json')
    with pytest.raises(ValueError):validate_manager(p,'other.json')


def test_handover_copies_journal_and_signals_only_old_cpu(tmp_path,monkeypatch):
    from look.runtime import mechanism_handover as h
    from look.runtime.state import atomic_write_json
    import signal
    oldroot=tmp_path/'old';newroot=tmp_path/'new';oldroot.mkdir();newroot.mkdir()
    lock=tmp_path/'account.lock';lock.touch()
    old=tmp_path/'old.json';new=tmp_path/'new.json'
    common=dict(claims='same',account_submission_lock=str(lock),maximum_workflow_allocations=2)
    atomic_write_json(dict(common,output=str(oldroot)),old);atomic_write_json(dict(common,output=str(newroot)),new)
    journal=dict(requests=[dict(job_id='123'),dict(job_id='456')]);atomic_write_json(journal,oldroot/'requests.json')
    atomic_write_json(dict(pid=2,state='ready'),newroot/'handover_ready.json')
    def process(pid):return dict(pid=pid,start='stable',state='S',args=['python','-m','look.runtime.project_dispatch','--config',str(old if pid==1 else new)]+([] if pid==1 else ['--standby']))
    monkeypatch.setattr(h,'process',process);signals=[];monkeypatch.setattr(h.os,'kill',lambda p,s:signals.append((p,s)))
    h.handover(old,new,1)
    assert signals==[(1,signal.SIGSTOP)]
    assert json.loads((newroot/'requests.json').read_text())==journal
    assert (newroot/'handover_armed.json').exists()
