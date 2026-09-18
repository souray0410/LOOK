import json
import subprocess
import pytest
from look.runtime.state import file_sha256
from look.studies import cohort_sequence as m


def plan(tmp_path):
    rows=[]
    for i in range(2):
        root=tmp_path/f'r{i}';root.mkdir()
        spec=dict(run_id=f'r{i}',output=str(root),test_access=False,seed=3416,
            data_audit_sha256='x',data_root='same',training={},factor=16,rank=32,position='deep',
            arms=['residual_rrr','pca_free_mean'],search='positive_forward_tree',framework_commit='fixed')
        path=root/'spec.json';path.write_text(json.dumps(spec))
        rows.append(dict(role='execute',spec=str(path),spec_sha256=file_sha256(path)))
    return dict(schema='look_cohort_sequence_v1',test_access=False,sequence_id='test',
        tasks=rows,root=str(tmp_path/'queue'),publication=str(tmp_path/'publication'))


def test_reject_duplicate_and_changed_fixed_factors(tmp_path):
    p=plan(tmp_path);m.validate_sequence(p)
    p['tasks'][1]=p['tasks'][0]
    with pytest.raises(ValueError,match='Duplicate'):m.validate_sequence(p)


def test_changed_spec_rejected(tmp_path):
    p=plan(tmp_path);path=m.Path(p['tasks'][1]['spec']);s=m.read(path);s['seed']=3417;path.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='Spec changed'):m.validate_sequence(p)
    p['tasks'][1]['spec_sha256']=file_sha256(path)
    with pytest.raises(ValueError,match='repeat seed'):m.validate_sequence(p)


def test_failure_isolated_and_success_not_rerun(tmp_path,monkeypatch):
    p=plan(tmp_path);events=[]
    monkeypatch.setattr(m,'refresh',lambda *args:None)
    monkeypatch.setattr(m,'publish',lambda *args:dict(matched_package='accepted'))
    class Child:
        def __init__(self,code):self.returncode=code
        def poll(self):return self.returncode
    def launch(s,row):
        events.append(s['run_id'])
        if s['run_id']=='r1':
            d=m.Path(s['output'])/'delivery';d.mkdir();(d/'accepted.json').write_text('{}')
        return Child(1 if s['run_id']=='r0' else 0)
    m.run(p,launch=launch,interval=.001)
    assert events==['r0','r1']
    assert m.read(m.Path(p['root'])/'completion.json')['state']=='needs_review'
    m.run(p,launch=launch,interval=.001)
    assert events==['r0','r1']
    assert m.read(m.Path(p['root'])/'status.json')['tasks']['r0']['state']=='needs_review'


def test_exclusive_sequence_lock(tmp_path,monkeypatch):
    p=plan(tmp_path);root=m.Path(p['root']);root.mkdir()
    with (root/'sequence.lock').open('a') as lock:
        m.fcntl.flock(lock,m.fcntl.LOCK_EX|m.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):m.run(p)
