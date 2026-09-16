import json
import pytest
from look.runtime.state import stable_hash, file_sha256
from look.runtime.weekly_delivery import release_state, filter_tasks, SECTIONS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return str(path)


def fixture(tmp_path):
    spec = write(tmp_path/'spec.json', {'seed': 3416})
    row = dict(spec=spec, spec_sha256=file_sha256(spec), run_dir=str(tmp_path/'run'))
    acceptance = write(tmp_path/'run/accepted.json', dict(state='accepted', test_access=False, profile=False))
    policy = dict(schema='look_weekly_delivery_v1', package_id='week1', first_seed=3416,
                  test_access=False, required_cases=[row], release_receipt=str(tmp_path/'release.json'))
    path = write(tmp_path/'policy.json', policy)
    artifact = write(tmp_path/'report.json', {'a': 1})
    sections = {k:[dict(path=artifact,sha256=file_sha256(artifact))] for k in SECTIONS}
    receipt = dict(state='accepted',profile=False,test_access=False,policy_sha256=stable_hash(policy),
                   cases={row['run_dir']:file_sha256(acceptance)},sections=sections)
    return path, policy, receipt


def test_whole_package_including_figures_required(tmp_path):
    path, policy, receipt = fixture(tmp_path)
    assert not release_state(path)['released']
    write(tmp_path/'release.json', receipt)
    assert release_state(path)['released']
    receipt['sections']['figures'] = []
    write(tmp_path/'release.json', receipt)
    assert not release_state(path)['released']


def test_reject_stale_changed_or_profile_receipts(tmp_path):
    path, policy, receipt = fixture(tmp_path)
    write(tmp_path/'release.json', receipt)
    write(tmp_path/'run/accepted.json', dict(state='accepted', test_access=False, profile=True))
    assert not release_state(path)['released']
    receipt['cases'][str(tmp_path/'run')] = file_sha256(tmp_path/'run/accepted.json')
    write(tmp_path/'release.json', receipt)
    assert not release_state(path)['released']


def test_no_score_gate_and_no_implicit_per_host_release(tmp_path):
    path, policy, receipt = fixture(tmp_path)
    tasks=[]
    for i, (kind, seed) in enumerate([('look',3416),('look_search',3417),('look_mechanism',3418),('native',3418)]):
        spec=write(tmp_path/f'{i}.json', {'host':{'seed':seed}})
        tasks.append(dict(execution=kind,spec=spec,run_dir=str(i)))
    ready, held=filter_tasks(tasks,path)
    assert [t['run_dir'] for t in ready]==['0','3']
    assert len(held)==2
    # The gate has no performance threshold, accepts negative as well as positive findings.
    write(tmp_path/'release.json', receipt)
    assert len(filter_tasks(tasks,path)[0])==4
    write(tmp_path/'report.json', {'a':2})
    assert not release_state(path)['released']


def test_empty_manifest_and_unknown_seed_fail_closed(tmp_path):
    path, policy, receipt=fixture(tmp_path)
    policy['required_cases']=[]
    write(tmp_path/'policy.json', policy)
    receipt['policy_sha256']=stable_hash(policy)
    write(tmp_path/'release.json', receipt)
    assert not release_state(path)['released']
    spec=write(tmp_path/'unknown.json', {})
    assert not filter_tasks([dict(spec=spec,run_dir='unknown',execution='look')],path)[0]
