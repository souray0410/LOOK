import json
from pathlib import Path
import pytest
from look.runtime.attached_search import tick
from look.runtime.state import file_sha256


def fixture(tmp_path):
    spec=tmp_path/'spec.json';spec.write_text('{}')
    run=tmp_path/'run';run.mkdir();claim=tmp_path/'claim.json'
    cfg=dict(job='123',task=dict(run_dir=str(run),spec=str(spec),spec_sha256=file_sha256(spec)))
    launch=dict(owner='old',generation=2)
    claim.write_text(json.dumps(dict(**launch,job_id='123',step='4',state='running',spec_sha256=file_sha256(spec))))
    class Claims:
        def path(self,run):return claim
        def mutate(self,run,fn):
            value=fn(json.loads(claim.read_text()));claim.write_text(json.dumps(value));return value
    return cfg,launch,Claims(),run,claim


def test_live_and_unknown_never_release_and_expiry_requests_checkpoint(tmp_path):
    c,l,claims,run,path=fixture(tmp_path)
    assert tick(c,l,claims,lambda *a:True,None,None,2000,now=100)['state']=='observing'
    assert tick(c,l,claims,lambda *a:None,None,None,2000,now=1900)['state']=='liveness_unknown'
    assert json.loads(path.read_text())['state']=='running'
    assert (run/'pause.json').exists()


def test_forged_acceptance_and_replaced_claim_cannot_release(tmp_path):
    c,l,claims,run,path=fixture(tmp_path);(run/'accepted.json').write_text('{}')
    def reject(*a):raise ValueError('bad evidence')
    with pytest.raises(ValueError,match='bad evidence'):tick(c,l,claims,lambda *a:False,reject,None,2000,now=100)
    assert json.loads(path.read_text())['state']=='running'
    claims.mutate(run,lambda x:dict(x,generation=3))
    with pytest.raises(ValueError,match='identity'):tick(c,l,claims,lambda *a:False,reject,None,2000,now=100)


def test_completed_requires_verification_and_publishing_before_release(tmp_path):
    c,l,claims,run,path=fixture(tmp_path);(run/'accepted.json').write_text('{}');events=[]
    result=tick(c,l,claims,lambda *a:False,lambda *a:events.append('verify'),lambda:events.append('publish'),2000,now=100)
    assert events==['verify','publish'] and result['state']=='completed'
    assert json.loads(path.read_text())['generation']==2
