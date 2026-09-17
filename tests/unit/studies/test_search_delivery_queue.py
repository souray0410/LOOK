import json
from pathlib import Path
import pytest
from look.studies.search_delivery_queue import compile_sequence
from look.studies.search_protocol import protocol,VERSION,representative_starts
from look.studies.suffix_protocol import sites
from look.runtime.state import file_sha256


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def fixture(tmp_path):
    ordered=sites('resnet50','deep');items=[]
    for i,(mode,start) in enumerate([('best_forward',1)]+[('greedy',s) for s in representative_starts('resnet50','deep')]):
        spec=dict(schema=VERSION,protocol=protocol(),test_access=False,
            host=dict(architecture='resnet50',position='deep',disease='cataract',seed=3416),mode=mode,start_ordinal=start,
            candidate_sites=ordered,eligible_sites=ordered[start-1:],source={'same':'host'},pca={'same':'basis'},
            spatial_factors=[16],latent_dims=[32])
        path=write(tmp_path/f'spec{i}.json',spec)
        task=dict(id=str(i),spec=str(path),spec_sha256=file_sha256(path),run_dir=str(tmp_path/f'run{i}'))
        launch=write(tmp_path/f'launch{i}.json',dict(task=task))
        items.append(dict(config=str(launch),sha256=file_sha256(launch)))
    sequence=write(tmp_path/'sequence.json',dict(tasks=items))
    return sequence


def test_same_runs_specs_independent_controls_and_idempotence(tmp_path):
    sequence=fixture(tmp_path);out=tmp_path/'compiled'
    a=compile_sequence(sequence,out)
    assert a['tasks']==5 and a['state']=='prepared_not_activated'
    policy=json.loads((out/'delivery_policy.json').read_text())
    assert [t['role'] for t in policy['tasks']]==['delivery']+['matched_control']*4
    assert all(not t['dependencies'] for t in policy['tasks'])
    first=file_sha256(out/'feed.json');compile_sequence(sequence,out)
    assert file_sha256(out/'feed.json')==first
    assert not any(tmp_path.glob('run*/accepted.json'))


def test_incomplete_and_changed_input_rejected(tmp_path):
    sequence=fixture(tmp_path);value=json.loads(sequence.read_text());value['tasks'].pop();write(sequence,value)
    with pytest.raises(ValueError,match='Incomplete'):compile_sequence(sequence,tmp_path/'compiled')
    (tmp_path/'spec0.json').write_text('{}')
    with pytest.raises(ValueError):compile_sequence(sequence,tmp_path/'compiled')
