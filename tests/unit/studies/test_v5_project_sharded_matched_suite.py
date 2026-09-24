import json
from pathlib import Path

import pytest

from look.studies import v5_project_sharded_matched_suite as suite
from look.runtime.state import file_sha256, stable_hash


def test_full_method_scope_and_resume(tmp_path, monkeypatch):
    calls = []
    cfg = dict(factors=[4, 8], latent_dims=[8, 16], max_rank=16)
    sites = ['joint_input', 'fusion_participant_feature']
    def fit(*args, **kwargs):
        calls.append((args, kwargs))
        destination = args[9]
        destination.mkdir(parents=True)
        (destination / 'factor_selection.json').write_text('{}')
        return [destination.name], []
    monkeypatch.setattr(suite, 'greedy_fit_look', fit)
    monkeypatch.setattr(suite, 'load_selected_bank', lambda p: [p.name])
    result = suite.fit_banks('graph', 'train', 'dev', sites, cfg, 'pca', 'cpu', tmp_path)
    assert len(calls) == 6
    for args, kwargs in calls:
        assert args[1:3] == ('train', 'dev')
        assert args[3] in ('oct_missing', 'cfp_missing')
        assert args[5:8] == ([4, 8], [8, 16], 16)
        method = args[9].parent.name
        assert args[4] == (['fusion_participant_feature'] if method == 'single_final' else sites)
        assert kwargs == dict(resume=True, force_enable=method == 'all_on')
    calls.clear()
    assert suite.fit_banks('graph', 'train', 'dev', sites, cfg, 'pca', 'cpu', tmp_path) == result
    assert not calls


def test_acceptance_rejects_modified_evidence(tmp_path):
    identity = dict(test_access=False)
    evidence = tmp_path / 'suite.json'
    evidence.write_text('{}')
    receipt = dict(schema='look_formal_v5_matched_suite_receipt_v1', state='accepted',
        test_access=False, identity_sha256=stable_hash(identity), files={'suite.json':file_sha256(evidence)})
    (tmp_path / 'accepted.json').write_text(json.dumps(receipt))
    assert suite.verify_complete(tmp_path, identity) == receipt
    evidence.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='evidence changed'):
        suite.verify_complete(tmp_path, identity)


def test_rejects_incomplete_cache_before_cuda_or_output(tmp_path):
    source, cache = tmp_path/'source', tmp_path/'cache'
    source.mkdir(); cache.mkdir()
    (source/'spec.json').write_text(json.dumps(dict(schema='look_project_case_v1',
        test_access=False, methods=suite.METHODS)))
    (cache/'accepted.json').write_text(json.dumps(dict(state='partial')))
    output=tmp_path/'output'
    with pytest.raises(ValueError, match='Complete feature cache'):
        suite.execute(source_run=source, cache=cache, checkpoint=tmp_path/'best.pt',
            pca=tmp_path/'pca', output=output, inputs_factory=None, device='cuda:0', gpu_budget_bytes=10)
    assert not output.exists()


def test_pause_is_propagated_to_lease_owner(tmp_path, monkeypatch):
    def paused(*args, **kwargs):
        raise suite.Paused()
    monkeypatch.setattr(suite, 'greedy_fit_look', paused)
    with pytest.raises(suite.Paused):
        suite.fit_banks(None, None, None, [],
            dict(factors=[4], latent_dims=[8], max_rank=8), {}, 'cpu', tmp_path)
