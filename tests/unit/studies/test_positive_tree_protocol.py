import json
import pytest

from look.runtime.state import file_sha256
from look.studies.search_protocol import VERSION, TREE_MODE, TREE_VERSION, protocol, validate, sites
from look.studies.search_delivery_queue import compile_sequence, register_tree_case


def tree_spec():
    ordered=sites('resnet50','deep')
    return dict(schema=VERSION,protocol=protocol(TREE_MODE),test_access=False,
        host=dict(architecture='resnet50',position='deep',disease='cataract',seed=3416),
        mode=TREE_MODE,candidate_sites=ordered,eligible_sites=ordered,
        spatial_factors=[16],latent_dims=[32])


def test_tree_is_explicit_identity_and_old_contract_is_unchanged():
    spec=tree_spec();validate(spec)
    assert spec['protocol']['schema']==TREE_VERSION
    assert protocol()==protocol('greedy')==protocol('best_forward')
    assert protocol()['main']=='best_forward'
    assert spec['protocol']!=protocol()
    for changed in (dict(spec,protocol=protocol()),dict(spec,mode='best_forward'),
                    dict(spec,start_ordinal=2),dict(spec,test_access=True)):
        with pytest.raises(ValueError):validate(changed)
    bad=tree_spec();bad['protocol']['branch_acceptance']='nonnegative'
    with pytest.raises(ValueError):validate(bad)


def test_tree_first_seed_registration_is_independent_immutable_and_not_accepted(tmp_path):
    path=tmp_path/'spec.json';path.write_text(json.dumps(tree_spec()))
    out=tmp_path/'registration';run=tmp_path/'run'
    receipt=register_tree_case(path,run,out)
    assert receipt['state']=='prepared_not_activated'
    assert receipt['weekly_package_released'] is False
    assert not run.exists()
    policy=json.loads((out/'delivery_policy.json').read_text())
    assert [row['role'] for row in policy['tasks']]==['delivery']
    feed=json.loads((out/'feed.json').read_text())
    assert [row['search_mode'] for row in feed['tasks']]==[TREE_MODE]
    before=file_sha256(out/'feed.json')
    assert register_tree_case(path,run,out)==receipt
    assert file_sha256(out/'feed.json')==before
    with pytest.raises(ValueError,match='Immutable'):
        register_tree_case(path,tmp_path/'different_run',out)
    assert file_sha256(out/'feed.json')==before


def test_tree_cannot_substitute_for_old_weekly_package_or_register_repeat(tmp_path):
    spec=tree_spec();path=tmp_path/'spec.json';path.write_text(json.dumps(spec))
    task=dict(spec=str(path),spec_sha256=file_sha256(path),run_dir=str(tmp_path/'run'))
    launch=tmp_path/'launch.json';launch.write_text(json.dumps(dict(task=task)))
    sequence=tmp_path/'sequence.json';sequence.write_text(json.dumps(dict(tasks=[
        dict(config=str(launch),sha256=file_sha256(launch))])))
    with pytest.raises(ValueError,match='Incomplete representative'):
        compile_sequence(sequence,tmp_path/'legacy_package')
    spec['host']['seed']=3417
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):register_tree_case(path,tmp_path/'run',tmp_path/'repeat')


def test_tree_cannot_rebind_existing_run(tmp_path):
    path=tmp_path/'spec.json';path.write_text(json.dumps(tree_spec()))
    run=tmp_path/'run';run.mkdir();(run/'spec.json').write_text('{}')
    with pytest.raises(ValueError,match='different scientific identity'):
        register_tree_case(path,run,tmp_path/'registration')
    assert not (tmp_path/'registration').exists()


def test_old_pilot_cannot_release_tree_repeat_seeds():
    for seed in (3417,3418):
        spec=tree_spec();spec['host']['seed']=seed
        spec['pilot']=[{'path':'old_greedy'},{'path':'old_best_forward'}]
        with pytest.raises(ValueError,match='whole weekly package'):
            validate(spec)
