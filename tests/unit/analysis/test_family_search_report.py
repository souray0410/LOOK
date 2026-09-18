import json
import numpy as np
import pytest
from look.analysis import family_search_report as module
from look.studies import family_search_case
from look.runtime.state import atomic_write_json,file_sha256
from tests.unit.studies.test_family_search_protocol import spec


def fixture(tmp_path,monkeypatch,duplicate=False):
    atomic_write_json(spec(),tmp_path/'spec.json');atomic_write_json({},tmp_path/'accepted.json')
    monkeypatch.setattr(family_search_case,'verify_case',lambda *a: {})
    rows=[]
    for pattern in module.PATTERNS:
        for method in ('host','search'):
            p=tmp_path/f'{pattern}_{method}.npz'
            np.savez(p,participant_ids=np.arange(4),labels=np.array([0,0,1,1]),logits=np.array([[2.,0.],[2.,0.],[0.,2.],[0.,2.]]))
            rows.append(dict(scenario=pattern,method=method,path=str(p),sha256=file_sha256(p),metrics={'macro_f1':1.}))
        atomic_write_json({},tmp_path/'corrections'/pattern/'selection.json')
        atomic_write_json({'full_mhd_replay':True},tmp_path/'corrections'/pattern/'replay.json')
    if duplicate:rows.append(rows[-1])
    atomic_write_json({'records':rows},tmp_path/'development/suite.json')
    atomic_write_json({},tmp_path/'costs.json')
    return rows


def test_report_rejects_duplicate_and_incomplete_rows(tmp_path,monkeypatch):
    rows=fixture(tmp_path,monkeypatch,True)
    with pytest.raises(ValueError,match='Duplicate'):module.report(tmp_path)
    atomic_write_json({'records':rows[:3]},tmp_path/'development/suite.json')
    with pytest.raises(ValueError,match='Both missing'):module.report(tmp_path)


def test_report_is_case_only_and_idempotent(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    def bootstrap(labels,predictions,weights,iterations,seed):
        assert predictions.shape==(4,4) and iterations==10000
        return {'test':'synthetic'}
    monkeypatch.setattr(module,'simultaneous_bootstrap',bootstrap)
    result=module.report(tmp_path)
    assert result['matched_family_complete'] is False and result['arm']=='residual_rrr'
    assert module.report(tmp_path)==result
    (tmp_path/'delivery/results.json').write_text('[]')
    with pytest.raises(ValueError,match='changed'):module.report(tmp_path)


def package_fixture(tmp_path,monkeypatch,complete):
    from look.studies.family_search_protocol import ARMS
    tasks=[]
    for arm in ARMS:
        root=tmp_path/arm;root.mkdir()
        fixture(root,monkeypatch)
        s=spec();s['arm']=arm;atomic_write_json(s,root/'spec.json')
        if arm not in complete:(root/'accepted.json').unlink()
        tasks.append(dict(execution='look_family_search',test_access=False,arm=arm,
            spec=str(root/'spec.json'),spec_sha256=file_sha256(root/'spec.json'),run_dir=str(root)))
    feed=tmp_path/'feed.json';atomic_write_json(dict(schema='look_family_search_feed_v1',test_access=False,tasks=tasks),feed)
    monkeypatch.setattr(module,'report',lambda root: {'state':'accepted'})
    return feed


def test_package_core_incomplete_has_no_ranking(tmp_path,monkeypatch):
    feed=package_fixture(tmp_path,monkeypatch,{'residual_rrr'})
    r=module.refresh_package(feed,tmp_path/'report')
    assert r['state']=='incomplete' and r['ranking_available'] is False
    assert r['waiting']==['pca_free_mean'] and len(r['deferred'])==2
    assert not (tmp_path/'report/current.json').exists()


def test_package_two_free_mean_arms_complete_without_deferred_and_preserve_failure(tmp_path,monkeypatch):
    feed=package_fixture(tmp_path,monkeypatch,{'residual_rrr','pca_free_mean'})
    out=tmp_path/'report'
    calls=[]
    def bootstrap(labels,predictions,weights,iterations,seed):
        calls.append((predictions.shape,len(weights),iterations));return {'contrasts':[]}
    monkeypatch.setattr(module,'simultaneous_bootstrap',bootstrap)
    r=module.refresh_package(feed,out)
    assert r['matched_package_complete'] is True and r['matched_family_complete'] is False
    assert calls==[((6,4),6,10000)]
    before=file_sha256(out/'current.json')
    assert module.refresh_package(feed,out)==r and len(calls)==1
    # Different frozen host logits with same argmax/metrics must fail and preserve valid publication.
    p=tmp_path/'pca_free_mean/oct_missing_host.npz'
    with np.load(p) as f:a={k:f[k] for k in f.files}
    a['logits']=a['logits']+1;np.savez(p,**a)
    suite=tmp_path/'pca_free_mean/development/suite.json';rows=json.loads(suite.read_text())
    for row in rows['records']:
        if row['path']==str(p):row['sha256']=file_sha256(p)
    atomic_write_json(rows,suite)
    with pytest.raises(ValueError,match='Frozen host logits'):module.refresh_package(feed,out)
    assert file_sha256(out/'current.json')==before
    assert json.loads((out/'status.json').read_text())['last_success_preserved'] is True


def test_full_package_contrasts_and_feed_tampering(tmp_path,monkeypatch):
    from look.studies.family_search_protocol import ARMS
    feed=package_fixture(tmp_path,monkeypatch,set(ARMS));calls=[]
    def bootstrap(labels,predictions,weights,iterations,seed):
        calls.append((predictions.shape,len(weights)));return {}
    monkeypatch.setattr(module,'simultaneous_bootstrap',bootstrap)
    r=module.refresh_package(feed,tmp_path/'report',ARMS)
    assert r['matched_family_complete'] is True and calls==[((10,4),18)]
    f=json.loads(feed.read_text());f['tasks'][0]['spec_sha256']='wrong';atomic_write_json(f,feed)
    with pytest.raises(ValueError,match='spec changed'):module.refresh_package(feed,tmp_path/'report',ARMS)
