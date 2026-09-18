import copy
import json
import pytest
from look.runtime.state import file_sha256
from look.studies.search_protocol import VERSION as SEARCH_VERSION, protocol as search_protocol, sites
from look.studies.family_search_protocol import VERSION, ARMS, protocol, validate
from look.studies.family_search_registry import register


def reference():
    ordered=sites('resnet50','deep')
    return dict(schema=SEARCH_VERSION,protocol=search_protocol('positive_forward_tree'),test_access=False,
        host=dict(disease='cataract',architecture='resnet50',position='deep',seed=3416),
        mode='positive_forward_tree',candidate_sites=ordered,eligible_sites=ordered,
        spatial_factors=[16],latent_dims=[32])


def spec():
    return dict(schema=VERSION,protocol=protocol(),test_access=False,reference_search=reference(),host=reference()['host'],
        arm='residual_rrr',candidates=[dict(rank=32,ridge_lambda=None)],penalty_policy='prefix_train_pca_gcv',
        candidate_provenance='train GCV',source_pins=[dict(path='source',sha256='sha')],workspace_bytes=1024)


def test_family_protocol_train_gcv_and_scope():
    s=spec();validate(s)
    for field,value in [('test_access',True),('arm','residual_ridge'),('workspace_bytes',0),
                        ('penalty_policy','fixed_candidates'),('candidates',[dict(rank=32,ridge_lambda=.01)])]:
        bad=copy.deepcopy(s);bad[field]=value
        with pytest.raises(ValueError):validate(bad)
    for key,value in [('seed',3417),('position','middle'),('architecture','resnet18')]:
        bad=copy.deepcopy(s);bad['reference_search']['host'][key]=value
        with pytest.raises((ValueError,KeyError)):validate(bad)
    assert protocol()['legacy_gcv_equivalent'] is False


def test_registry_idempotence_identity_and_no_dispatch(tmp_path):
    p=tmp_path/'reference.json';p.write_text(json.dumps(reference()))
    c=dict(output=str(tmp_path/'registry'),reference_search_spec=str(p),reference_search_sha256=file_sha256(p),
        source_pins=[dict(path='source',sha256='sha')],workspace_bytes=1024,
        run_dirs={arm:str(tmp_path/'runs'/arm) for arm in ARMS})
    a=register(c);before=file_sha256(tmp_path/'registry/feed.json')
    assert a['state']=='registered_not_dispatched' and a['replication_released'] is False
    assert [t['arm'] for t in a['tasks']]==list(ARMS)
    assert not (tmp_path/'runs').exists()
    assert register(c)==a and file_sha256(tmp_path/'registry/feed.json')==before
    c['workspace_bytes']=2048
    with pytest.raises(ValueError,match='Immutable'):register(c)
    assert file_sha256(tmp_path/'registry/feed.json')==before


def test_registry_rejects_reused_scientific_run(tmp_path):
    p=tmp_path/'reference.json';p.write_text(json.dumps(reference()))
    run=tmp_path/'existing';run.mkdir();(run/'spec.json').write_text('{}')
    c=dict(output=str(tmp_path/'registry'),reference_search_spec=str(p),reference_search_sha256=file_sha256(p),
        source_pins=[dict(path='source',sha256='sha')],workspace_bytes=1024,
        run_dirs={arm:str(tmp_path/'runs'/arm) for arm in ARMS})
    c['run_dirs']['residual_rrr']=str(run)
    with pytest.raises(ValueError,match='different scientific identity'):register(c)
    assert not (tmp_path/'registry/feed.json').exists()
