import json
import pytest
from look.methods import embracenet_family as family


def test_full_suite_preserves_independent_sites_searches_and_arms(tmp_path, monkeypatch):
    calls = []
    def run(graph, train, dev, **kw):
        calls.append(kw)
        kw['output'].mkdir(parents=True, exist_ok=True)
        (kw['output'] / 'selection.json').write_text('{}')
        for name in ('bank.pt', 'replay.json', 'prediction.npz'):
            (kw['output'] / name).write_text('fixture')
        return [], {'final': {'score': .5, 'prediction': str(kw['output'] / 'prediction.npz')}}
    monkeypatch.setattr(family, 'fit_embracenet_family_trajectory', run)
    arms = ['shared_pca_ridge', 'pca_free_mean', 'rrr_shared_intercept', 'residual_rrr']
    candidates = {arm: [{'rank': 2}] for arm in arms}
    rows = family.fit_embracenet_search_suite(None, None, None, sites=['a', 'b'],
        candidates_by_arm=candidates, output=tmp_path, identity={'host': 'frozen'})
    assert len(rows) == 4 * 2 * (2 + 2)
    assert len({str(call['output']) for call in calls}) == len(rows)
    for arm in arms:
        for pattern in ['oct_missing', 'cfp_missing']:
            group = [call for call in calls if call['arm'] == arm and call['pattern'] == pattern]
            assert [call['sites'] for call in group] == [['a'], ['b'], ['a', 'b'], ['a', 'b']]
            assert [call['identity']['search'] for call in group] == ['single_site', 'single_site', 'best_forward', 'positive_forward_tree']
            assert all(call['candidates'] == candidates[arm] for call in group)
    assert json.loads((tmp_path / 'coverage.json').read_text())['scientific_acceptance'] is False
    calls.clear()
    again = family.fit_embracenet_search_suite(None, None, None, sites=['a', 'b'],
        candidates_by_arm=candidates, output=tmp_path, identity={'host': 'frozen'})
    assert again == rows and not calls
    (tmp_path / arms[0] / 'oct_missing/single_000/bank.pt').write_text('changed')
    with pytest.raises(ValueError, match='artifact changed'):
        family.fit_embracenet_search_suite(None, None, None, sites=['a', 'b'],
            candidates_by_arm=candidates, output=tmp_path, identity={'host': 'frozen'})


def test_suite_refuses_missing_family(tmp_path):
    with pytest.raises(ValueError, match='four'):
        family.fit_embracenet_search_suite(None, None, None, sites=['a'],
            candidates_by_arm={}, output=tmp_path, identity={'host': 'frozen'})
