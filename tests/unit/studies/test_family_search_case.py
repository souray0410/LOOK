import copy
import json
import pytest
import torch
from look.studies import family_search_case as module
from look.runtime.state import atomic_write_json, stable_hash
from tests.unit.studies.test_family_search_protocol import spec


def test_formal_and_profile_outputs_cannot_contaminate(tmp_path):
    s=spec();(tmp_path/'profile_accepted.json').write_text('{}')
    with pytest.raises(ValueError,match='train-only profile'):module.execute(s,tmp_path)
    (tmp_path/'profile_accepted.json').unlink();(tmp_path/'accepted.json').write_text('{}')
    with pytest.raises(ValueError,match='separate output'):module.execute(s,tmp_path,profile=True)


def test_reject_cpu_runtime_even_with_valid_dependency_stub(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'dependencies',lambda s: ({},tmp_path,{}))
    with pytest.raises(ValueError,match='Slurm GPU'):module.run(spec(),tmp_path,torch.device('cpu'),lambda:False)


def test_receipt_requires_both_states_replay_and_identity(tmp_path):
    s=spec();r=dict(schema=module.VERSION,identity=stable_hash(s),state='accepted',test_access=False,profile=False,
        host_frozen=True,reload_exact=True,rng_restored=True,upstream_refit_verified=True,pca_unchanged=True,files={})
    atomic_write_json(r,tmp_path/'accepted.json')
    with pytest.raises(ValueError,match='Missing family'):module.verify_case(tmp_path,s)
    r['identity']='wrong';atomic_write_json(r,tmp_path/'accepted.json')
    with pytest.raises(ValueError,match='technical acceptance'):module.verify_case(tmp_path,s)


def test_pause_is_not_completion(tmp_path,monkeypatch):
    def pause(*a,**kw):raise module.SelectionPaused()
    monkeypatch.setattr(module,'run',pause)
    assert module.execute(spec(),tmp_path)=={'state':'paused'}
    assert json.loads((tmp_path/'status.json').read_text())['state']=='paused'
    assert not (tmp_path/'accepted.json').exists()
