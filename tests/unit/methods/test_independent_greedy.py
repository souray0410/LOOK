import json
import pytest
from look.methods.independent_greedy import fit_trajectory, SelectionPaused


def execute(root, mode='greedy', pause=lambda:False, identity='locked'):
    calls=[]
    def fit(site, bank, out):
        calls.append((site, tuple(bank)))
        # First hurts, second helps only if rejected first did not leak.
        value = -1 if site=='a' else 2 if not bank else -5
        yield 'b', value
        yield 'a', value  # Tie break independent of enumeration order.
    def evaluate(bank):
        return dict(role='development', score=.5+.01*sum(bank))
    def save(a,p):p.write_text(json.dumps(a))
    result=fit_trajectory(identity=identity,sites=['a','b'],mode=mode,output=root,
        fit_candidates=fit,evaluate=evaluate,save_artifact=save,
        load_artifact=lambda p:json.loads(p.read_text()),should_pause=pause)
    return result,calls


def test_rejected_candidate_does_not_leak_and_all_on_refits(tmp_path):
    (bank,r),calls=execute(tmp_path/'greedy')
    assert bank==[2] and calls==[('a',()),('b',())]
    assert [d['enabled'] for d in r['decisions']]==[False,True]
    assert all(d['selected_key']=='a' for d in r['decisions'])
    (_,a),calls=execute(tmp_path/'all_on','all_on')
    assert calls==[('a',()),('b',(-1,))]
    assert a['final']['score'] < r['final']['score']


def test_restart_reuses_decisions_and_rejects_changed_identity(tmp_path):
    first,_=execute(tmp_path)
    second,calls=execute(tmp_path)
    assert first==second and calls==[]
    with pytest.raises(ValueError,match='identity'):execute(tmp_path,identity='changed')
    candidate=next((tmp_path/'candidates').glob('*.pt'));candidate.write_text('0')
    with pytest.raises(ValueError,match='evidence'):execute(tmp_path)


def test_pause_after_first_decision_then_resume(tmp_path):
    def pause():return (tmp_path/'decisions/000.json').exists()
    with pytest.raises(SelectionPaused):execute(tmp_path,pause=pause)
    (bank,r),calls=execute(tmp_path)
    assert bank==[2] and calls==[('b',())]
    reference,_=execute(tmp_path/'reference')
    assert r['final']==reference[1]['final']


def test_ties_off_test_rejected_and_no_feasible_candidates(tmp_path):
    common=dict(identity='x',sites=['a'],mode='greedy',fit_candidates=lambda *a: [('x',0)],
                save_artifact=lambda a,p:p.write_text('0'),load_artifact=lambda p:0)
    bank,result=fit_trajectory(output=tmp_path/'tie',evaluate=lambda b:dict(role='development',score=.5),**common)
    assert not bank and not result['decisions'][0]['enabled']
    with pytest.raises(ValueError,match='test is sealed'):
        fit_trajectory(output=tmp_path/'test',evaluate=lambda b:dict(role='test',score=.5),**common)
    common['fit_candidates']=lambda *a:[]
    with pytest.raises(ValueError,match='No feasible'):
        fit_trajectory(output=tmp_path/'empty',evaluate=lambda b:dict(role='development',score=.5),**common)


def test_methods_have_independent_states(tmp_path):
    a,_=execute(tmp_path/'method_a',identity='a')
    b,calls=execute(tmp_path/'method_b',identity='b')
    assert calls==[('a',()),('b',())] and a[0]==b[0]


def forward(root, values, *, pause=lambda:False):
    calls=[]
    def fit(site,bank,out):
        calls.append((site,tuple(bank)))
        yield 'q1',values(site,bank)
    bank,result=fit_trajectory(identity='forward',sites=list('abc'),mode='best_forward',output=root,
        fit_candidates=fit,evaluate=lambda b:dict(role='development',score=.1+sum(b)/100),
        save_artifact=lambda a,p:p.write_text(json.dumps(a)),load_artifact=lambda p:json.loads(p.read_text()),
        should_pause=pause)
    return bank,result,calls


def test_forward_selects_winner_then_refits_downstream_only(tmp_path):
    def values(site,bank):
        return dict(a=1,b=4,c=2)[site] if not bank else 3
    bank,result,calls=forward(tmp_path,values)
    assert bank==[4,3]
    assert calls==[('a',()),('b',()),('c',()),('c',(4,))]
    assert result['site_attempts']==4
    again,r,calls=forward(tmp_path,values)
    assert again==bank and r==result and calls==[]


def test_forward_worst_attempts_and_global_stop(tmp_path):
    _,r,calls=forward(tmp_path/'worst',lambda site,bank:dict(a=3,b=2,c=1)[site])
    assert len(calls)==r['site_attempts']==6
    # Negative a cannot stop the scan; c wins even when earlier candidates hurt.
    bank,r,calls=forward(tmp_path/'late',lambda site,bank:dict(a=-3,b=-2,c=1)[site])
    assert bank==[1] and len(calls)==3
    bank,r,calls=forward(tmp_path/'stop',lambda site,bank:0)
    assert bank==[] and len(calls)==3 and not r['decisions'][0]['enabled']


def test_forward_resumes_completed_site_without_leaking_candidates(tmp_path):
    values=lambda site,bank:dict(a=1,b=4,c=2)[site] if not bank else 3
    pause=lambda:(tmp_path/'rounds/000/site_000.json').exists()
    with pytest.raises(SelectionPaused):forward(tmp_path,values,pause=pause)
    bank,r,calls=forward(tmp_path,values)
    assert bank==[4,3] and calls==[('b',()),('c',()),('c',(4,))]
    artifact=next((tmp_path/'rounds/000/artifacts').glob('*.pt'));artifact.write_text('99')
    with pytest.raises(ValueError,match='evidence'):forward(tmp_path,values)


@pytest.mark.parametrize('mode', ['greedy','best_forward'])
def test_resume_checks_baseline_and_rejected_prediction_files(tmp_path, mode):
    from look.runtime.state import file_sha256
    for damaged in ('off','rejected'):
        root=tmp_path/damaged
        root.mkdir()
        evidence={key:root/f'{key}.npz' for key in ('off','rejected')}
        for p in evidence.values():p.write_bytes(b'original')
        hashes={key:file_sha256(p) for key,p in evidence.items()}
        def evaluate(bank):
            key='rejected' if bank else 'off'
            return dict(role='development',score=.4 if bank else .5,
                        prediction=str(evidence[key]),sha256=hashes[key])
        kw=dict(identity='fixed',sites=['a'],mode=mode,output=root,
            fit_candidates=lambda *args:[('only',1)],evaluate=evaluate,
            save_artifact=lambda a,p:p.write_text('1'),load_artifact=lambda p:1)
        bank,_=fit_trajectory(**kw)
        assert not bank
        evidence[damaged].write_bytes(b'corrupt')
        with pytest.raises(ValueError,match='Prediction evidence'):
            fit_trajectory(**kw)
