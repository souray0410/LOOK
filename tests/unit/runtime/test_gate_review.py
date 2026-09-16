import json
from pathlib import Path
from types import SimpleNamespace
from look.runtime.gate_review import tick


def setup(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'accepted.json').write_text('{}')
    feed=tmp_path/'feed.json';feed.write_text(json.dumps(dict(test_access=False,tasks=[dict(run_dir=str(source))])))
    return dict(output=str(tmp_path/'out'),feeds=[str(feed)],python='/python',pythonpath='/source',ld_library_path='/lib')


def test_one_submission_and_restart_preserves_live_request(tmp_path,monkeypatch):
    c=setup(tmp_path);calls=[]
    def run(cmd,**kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0,stdout='123\n',stderr='')
    monkeypatch.setattr('look.runtime.gate_review.subprocess.run',run)
    tick(c);tick(c)
    assert [x[0] for x in calls]==['sbatch','squeue']
    assert '--time=48:00:00' in calls[0] and not any('--gres' in x for x in calls[0])


def test_ambiguous_submission_is_not_retried(tmp_path,monkeypatch):
    c=setup(tmp_path);calls=[]
    def run(cmd,**kw):
        calls.append(cmd);raise TimeoutError('submission response lost')
    monkeypatch.setattr('look.runtime.gate_review.subprocess.run',run)
    tick(c);r=tick(c)
    assert len(calls)==1 and r['errors']


def test_failed_job_is_not_blindly_restarted(tmp_path,monkeypatch):
    c=setup(tmp_path);calls=[]
    def run(cmd,**kw):
        calls.append(cmd)
        out={'sbatch':'123\n','squeue':'','sacct':'FAILED|\n'}[cmd[0]]
        return SimpleNamespace(returncode=0,stdout=out,stderr='')
    monkeypatch.setattr('look.runtime.gate_review.subprocess.run',run)
    tick(c);r=tick(c)
    assert [x[0] for x in calls]==['sbatch','squeue','sacct'] and r['errors']
