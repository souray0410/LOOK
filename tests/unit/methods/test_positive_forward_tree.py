import json
import pytest
from look.methods.independent_greedy import fit_trajectory, SelectionPaused


def execute(root, scores, sites='abc', pause=lambda: False, candidates=('q',), identity='fixed'):
    calls = []
    def fit(site, bank, where):
        calls.append((tuple(a['site'] for a in bank), site))
        for key in candidates:
            yield key, dict(site=site, key=key)
    def evaluate(bank):
        path = tuple(a['site'] for a in bank)
        value = scores(path) if callable(scores) else scores.get(path, .1)
        return dict(role='development', score=value)
    result = fit_trajectory(identity=identity, sites=list(sites), mode='positive_forward_tree',
        output=root, fit_candidates=fit, evaluate=evaluate,
        save_artifact=lambda a, p: p.write_text(json.dumps(a)),
        load_artifact=lambda p: json.loads(p.read_text()), should_pause=pause)
    return result, calls


def test_every_positive_root_and_previously_negative_downstream(tmp_path):
    scores = {(): .5, ('a',): .6, ('b',): .4, ('c',): .8,
              ('a', 'b'): .7, ('a', 'c'): .59, ('a', 'b', 'c'): .9}
    (bank, result), calls = execute(tmp_path, scores)
    assert [a['site'] for a in bank] == list('abc')
    assert calls == [((), 'a'), ((), 'b'), ((), 'c'), (('a',), 'b'),
                     (('a',), 'c'), (('a', 'b'), 'c')]
    assert result['site_attempts'] == 6
    assert result['prefix_count'] == 5
    # The best single root c remains a branch, but cannot suppress a's descendants.
    assert [c['winner']['node'] for c in result['decisions'][0]['children']] == ['a', 'c']


def test_all_positive_worst_case_and_one_winner_per_site(tmp_path):
    (bank, result), calls = execute(tmp_path, lambda p: .1 + len(p) * .1,
                                   candidates=('z', 'a'))
    assert result['site_attempts'] == len(calls) == 2**3 - 1
    assert result['candidate_evaluations'] == 14
    assert result['prefix_count'] == 8
    assert all(a['key'] == 'a' for a in bank)


def test_no_improvement_ties_and_negative_extensions_are_not_explored(tmp_path):
    (bank, result), calls = execute(tmp_path, lambda p: .5 if not p or p == ('a',) else .4)
    assert bank == [] and len(calls) == 3 and result['prefix_count'] == 1
    assert result['final']['score'] == .5


def test_final_tie_prefers_fewer_sites_then_earlier_path(tmp_path):
    scores = {(): .1, ('a',): .2, ('b',): .4, ('c',): .4, ('a', 'b'): .4}
    (bank, _), _ = execute(tmp_path, scores)
    assert [a['site'] for a in bank] == ['b']


def test_resume_reuses_committed_sites_without_changing_frontier(tmp_path):
    scores = lambda p: .1 + len(p) * .1
    pause = lambda: (tmp_path / 'prefixes/root/site_001.json').exists()
    with pytest.raises(SelectionPaused):
        execute(tmp_path, scores, pause=pause)
    (bank, result), calls = execute(tmp_path, scores)
    assert ((), 'a') not in calls and ((), 'b') not in calls
    (again, repeated), calls = execute(tmp_path, scores)
    assert calls == [] and again == bank and repeated == result
    (reference, reference_result), _ = execute(tmp_path / 'reference', scores)
    assert reference == bank and reference_result['final'] == result['final']


def test_tamper_identity_artifact_prediction_and_child_baseline_rejected(tmp_path):
    execute(tmp_path, lambda p: .1 + len(p) * .1)
    with pytest.raises(ValueError, match='identity'):
        execute(tmp_path, lambda p: .1, identity='changed')
    candidate = next((tmp_path / 'prefixes/root/artifacts').glob('*.pt'))
    candidate.write_text('{}')
    with pytest.raises(ValueError, match='evidence'):
        execute(tmp_path, lambda p: .1 + len(p) * .1)


def test_child_caches_cannot_be_reused_from_a_different_prefix(tmp_path):
    execute(tmp_path, lambda p: .1 + len(p) * .1)
    baseline = next(p for p in (tmp_path / 'prefixes').glob('*/baseline.json') if p.parent.name != 'root')
    payload = json.loads(baseline.read_text())
    payload['evidence']['score'] = .99
    baseline.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='baseline differs'):
        execute(tmp_path, lambda p: .1 + len(p) * .1)


def test_rejected_candidate_prediction_is_still_verified(tmp_path):
    from look.runtime.state import file_sha256
    prediction = tmp_path / 'prediction.npz'
    prediction.write_bytes(b'fixed')
    digest = file_sha256(prediction)
    kw = dict(identity='x', sites=['a'], mode='positive_forward_tree', output=tmp_path,
        fit_candidates=lambda *a: [('q', 0)],
        evaluate=lambda bank: dict(role='development', score=.5 if not bank else .4,
                                  prediction=str(prediction), sha256=digest),
        save_artifact=lambda a,p: p.write_text('0'), load_artifact=lambda p: 0)
    fit_trajectory(**kw)
    prediction.write_bytes(b'changed')
    with pytest.raises(ValueError, match='evidence'):
        fit_trajectory(**kw)


@pytest.mark.parametrize('score,role', [(float('nan'), 'development'), (.5, 'test')])
def test_invalid_scores_and_test_access_rejected(tmp_path, score, role):
    with pytest.raises(ValueError, match='test is sealed'):
        fit_trajectory(identity='x', sites=['a'], mode='positive_forward_tree', output=tmp_path,
            fit_candidates=lambda *a: [('q', 0)], evaluate=lambda b: dict(role=role, score=score),
            save_artifact=lambda a,p: p.write_text('0'), load_artifact=lambda p: 0)


@pytest.mark.parametrize('scores', [
    # Best-forward takes the late root; the tree retains it and a better deep path.
    {(): .5, ('a',): .6, ('b',): .4, ('c',): .8,
     ('a','b'): .7, ('a','c'): .59, ('a','b','c'): .9},
    # b is harmful at root but useful after a; final equal gains are disabled.
    {(): .5, ('a',): .7, ('b',): .4, ('c',): .6,
     ('a','b'): .8, ('a','c'): .7, ('a','b','c'): .8},
    # Equal root winners choose the earlier site; candidate ties use key order.
    {(): .5, ('a',): .7, ('b',): .7, ('c',): .6,
     ('a','b'): .8, ('a','c'): .75, ('b','c'): .85,
     ('a','b','c'): .8},
    # Both searches retain the empty path when all root gains are nonpositive.
    {(): .5, ('a',): .5, ('b',): .4, ('c',): .45},
])
def test_tree_contains_best_forward_under_identical_prefix_fits_and_dev_rule(tmp_path, scores):
    """Finite deterministic algorithm invariant, not an empirical efficacy claim.

    Assumptions: same sites/candidate keys, deterministic fitting conditional on
    the entire prefix, identical dev scores, strict-positive gating and shared
    per-site candidate tie rule. This says nothing about unseen/test performance
    or compute cost; more retained dev alternatives do not establish either.
    """
    fits = {}
    def run(mode):
        seen = {}
        def fit(site, bank, where):
            prefix = [(a['site'], a['key']) for a in bank]
            # Deliberately emit reverse tie order to exercise key tie-breaking.
            for key in ('z', 'a', 'bad'):
                artifact = dict(site=site, key=key, fitted_prefix=[list(x) for x in prefix])
                seen[(tuple(prefix), site, key)] = artifact
                yield key, artifact
        def evaluate(bank):
            for i, artifact in enumerate(bank):
                assert artifact['fitted_prefix'] == [[a['site'],a['key']] for a in bank[:i]]
            path = tuple(a['site'] for a in bank)
            value = scores.get(path, .1)
            if bank and bank[-1]['key'] == 'bad':
                value -= .2
            return dict(role='development', score=value)
        result = fit_trajectory(identity='matched-prefix-fixture', sites=list('abc'), mode=mode,
            output=tmp_path/mode, fit_candidates=fit, evaluate=evaluate,
            save_artifact=lambda a,p:p.write_text(json.dumps(a)),
            load_artifact=lambda p:json.loads(p.read_text()))
        fits[mode] = seen
        return result

    best_bank,best = run('best_forward')
    _,tree = run('positive_forward_tree')
    best_path = tuple((a['site'],a['key']) for a in best_bank)
    retained = {tuple((r['node'],r['key']) for r in d['path']) for d in tree['decisions']}
    assert best_path in retained
    assert all(best_path[:length] in retained for length in range(len(best_path)+1))
    assert tree['final']['score'] >= best['final']['score']
    assert all(a['key'] == 'a' for a in best_bank)
    # Every fit consulted by best-forward exists identically in the tree, including
    # rejected candidates; no stale parent artifacts can manufacture containment.
    for identity, artifact in fits['best_forward'].items():
        assert fits['positive_forward_tree'][identity] == artifact
