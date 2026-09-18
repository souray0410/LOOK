import importlib.util
import json
import os
from pathlib import Path
import pytest
from look.runtime.state import file_sha256, stable_hash

candidate=os.environ.get('LOOK_PUBLICATION_CANDIDATE')
if candidate:
    spec=importlib.util.spec_from_file_location('publication_candidate',candidate)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    publish=module.publish
else:
    from look.analysis.cohort_publication import publish


def fixture(root):
    s=dict(run_id='synthetic',source_commit='source',framework_commit='framework',architecture='resnet50',
        position='deep',seed=3416,factor=16,rank=32,search='positive_forward_tree',
        arms=['residual_rrr','pca_free_mean'],training={},initialization={'kind':'public','path':'SECRET_PATH','sha256':'init'})
    (root/'spec.json').write_text(json.dumps(s))
    for arm in s['arms']:
        p=root/arm;p.mkdir();rows=[]
        for pattern in ('oct_missing','cfp_missing'):
            q=p/'corrections'/pattern;q.mkdir(parents=True);(q/'bank.pt').write_bytes(b'bank')
            for name in (arm,'host'):
                f=p/f'{name}_{pattern}.npz';f.write_bytes(b'private')
                rows.append(dict(method=name,scenario=pattern,path=str(f),sha256=file_sha256(f),
                    metrics={'macro_f1':.6,'macro_auroc_ovr':.7}))
        (p/'accepted.json').write_text(json.dumps(dict(state='accepted',identity=stable_hash(s),records=rows)))
        (root/(arm+'_status.json')).write_text(json.dumps(dict(state='completed',time=1000)))
    return s


def test_idempotence_and_unique_results(tmp_path):
    fixture(tmp_path);out=tmp_path/'public';v=publish(tmp_path,out)
    assert len(v['results'])==6
    assert 'SECRET_PATH' not in (out/'current.json').read_text()
    original={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()}
    assert publish(tmp_path,out)==v
    assert original=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()}
    assert v['evidence_cutoff_utc'].startswith('1970-01-01T00:16:40')


@pytest.mark.parametrize('fault',['identity','prediction','duplicate'])
def test_corruption_preserves_last_good_snapshot(tmp_path,fault):
    fixture(tmp_path);out=tmp_path/'public';publish(tmp_path,out)
    original={p.name:p.read_bytes() for p in out.iterdir()}
    p=tmp_path/'pca_free_mean/accepted.json';v=json.loads(p.read_text())
    if fault=='identity':v['identity']='bad'
    elif fault=='prediction':Path(v['records'][0]['path']).write_bytes(b'changed')
    else:v['records'][1]['metrics']['macro_f1']=.2
    p.write_text(json.dumps(v))
    with pytest.raises(ValueError):publish(tmp_path,out)
    assert original=={p.name:p.read_bytes() for p in out.iterdir()}


def test_reject_incomplete_delivery_and_then_publish_increment(tmp_path):
    s=fixture(tmp_path);out=tmp_path/'public';v=publish(tmp_path,out)
    delivery=tmp_path/'delivery';delivery.mkdir()
    def receipt(rows):
        (delivery/'results.json').write_text(json.dumps(dict(results=rows,comparisons=[],statistics={'contrasts':[]})))
        (delivery/'accepted.json').write_text(json.dumps(dict(state='accepted',identity=stable_hash(s),files={'results.json':file_sha256(delivery/'results.json')})))
    receipt([])
    with pytest.raises(ValueError,match='disagree'):publish(tmp_path,out)
    assert json.loads((out/'current.json').read_text())==v
    receipt(v['results']);new=publish(tmp_path,out)
    assert new['matched_package']=='accepted' and len(new['results'])==6
    assert new['source_evidence_sha256']!=v['source_evidence_sha256']
