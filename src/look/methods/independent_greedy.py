"""Independent, restartable per-method node selection; never inherits PCA gates.

The caller supplies a locked candidate generator and a full-graph evaluator.
Statistics and fitting depend on the accepted upstream bank. All-on is a fresh
trajectory, not a concatenation of maps rejected by a greedy trajectory.
"""
import fcntl
import json
import math
from pathlib import Path

from look.runtime.state import atomic_write_json, file_sha256, stable_hash


VERSION = 'look_independent_greedy_v1'


class SelectionPaused(Exception):
    pass


def _check_prediction_evidence(evidence):
    if evidence.get('role') != 'development' or not math.isfinite(evidence['score']):
        raise ValueError('Finite development score required; test is sealed')
    if evidence.get('prediction') and file_sha256(evidence['prediction']) != evidence['sha256']:
        raise ValueError('Prediction evidence changed')


def fit_trajectory(*, identity, sites, mode, output, fit_candidates, evaluate,
                   save_artifact, load_artifact, should_pause=lambda: False):
    """Each candidate is (stable key, artifact); evaluate returns score + evidence.

    A node decision is the recovery boundary. Candidate fitting can implement
    finer recovery itself. The enclosing identity must bind method, host, data,
    candidate table, source and starting point. No test evaluation is permitted.
    """
    if mode == 'positive_forward_tree':
        from look.methods.positive_forward_tree import fit_positive_forward_tree
        return fit_positive_forward_tree(identity=identity, sites=sites, output=output,
            fit_candidates=fit_candidates, evaluate=evaluate, save_artifact=save_artifact,
            load_artifact=load_artifact, should_pause=should_pause)
    if mode == 'best_forward':
        return fit_best_forward(identity=identity, sites=sites, output=output,
            fit_candidates=fit_candidates, evaluate=evaluate, save_artifact=save_artifact,
            load_artifact=load_artifact, should_pause=should_pause)
    if mode not in ('greedy', 'all_on') or not identity:
        raise ValueError('Explicit scientific identity and registered mode required')
    if not sites or len(set(sites)) != len(sites):
        raise ValueError('Nonempty ordered unique sites required')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    contract = dict(schema=VERSION, identity=identity, sites=list(sites), mode=mode,
                    test_access=False, ties='off; candidate keys ascending')
    with (root/'trajectory.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cp = root/'contract.json'
        if cp.exists() and json.loads(cp.read_text()) != contract:
            raise ValueError('Trajectory identity changed')
        atomic_write_json(contract, cp)
        bank, decisions = [], []
        for index, site in enumerate(sites):
            if should_pause():
                raise SelectionPaused()
            node_identity = stable_hash(dict(contract=contract, index=index,
                                             upstream=decisions))
            dp = root/'decisions'/f'{index:03d}.json'
            if dp.exists():
                decision = json.loads(dp.read_text())
                if decision['identity'] != node_identity:
                    raise ValueError('Upstream decision identity changed')
                _check_prediction_evidence(decision['baseline'])
                # Verify even rejected candidates: their evidence justified off.
                for row in decision['candidates']:
                    path = (root/row['artifact']).resolve()
                    if not path.is_relative_to(root.resolve()) or file_sha256(path) != row['sha256']:
                        raise ValueError('Candidate evidence changed')
                    _check_prediction_evidence(row['evidence'])
                if decision['enabled']:
                    chosen = next(r for r in decision['candidates'] if r['key'] == decision['selected_key'])
                    bank.append(load_artifact(root/chosen['artifact']))
                decisions.append(decision)
                continue

            def assessed(current):
                result = evaluate(tuple(current))
                if result.get('role') != 'development' or not math.isfinite(result['score']):
                    raise ValueError('Finite development score required; test is sealed')
                return result

            off = assessed(bank)
            rows, best_artifact, best_key, best_score = [], None, None, -math.inf
            seen = set()
            for key, artifact in fit_candidates(site, tuple(bank), root/'fit'/f'{index:03d}'):
                if should_pause():
                    raise SelectionPaused()
                if not isinstance(key, str) or key in seen:
                    raise ValueError('Candidate keys must be unique strings')
                seen.add(key)
                path = root/'candidates'/f'{index:03d}_{stable_hash(key)}.pt'
                path.parent.mkdir(parents=True, exist_ok=True)
                save_artifact(artifact, path)
                result = assessed([*bank, artifact])
                rows.append(dict(key=key, score=result['score'], evidence=result,
                                 artifact=str(path.relative_to(root)), sha256=file_sha256(path)))
                if result['score'] > best_score or (result['score'] == best_score and key < best_key):
                    best_score, best_key, best_artifact = result['score'], key, artifact
            if not rows:
                raise ValueError('No feasible candidates; do not silently skip node')
            enabled = mode == 'all_on' or best_score > off['score']
            decision = dict(identity=node_identity, node=site, mode=mode, baseline=off,
                            candidates=sorted(rows, key=lambda r:r['key']), selected_key=best_key,
                            enabled=enabled, selected_score=best_score if enabled else off['score'],
                            reason='all_on' if mode == 'all_on' else
                                   'strict_improvement' if enabled else 'no_strict_improvement')
            atomic_write_json(decision, dp)
            decisions.append(decision)
            if enabled:
                bank.append(best_artifact)
        final = evaluate(tuple(bank))
        if final.get('role') != 'development' or not math.isfinite(final['score']):
            raise ValueError('Invalid final evaluation role or score')
        if final['score'] != decisions[-1]['selected_score']:
            raise ValueError('Final replay differs from selected trajectory')
        result = dict(schema=VERSION, contract_sha256=file_sha256(cp), mode=mode,
                      decisions=decisions, final=final, test_access=False,
                      scientific_acceptance=False)
        atomic_write_json(result, root/'selection.json')
        return bank, result


def fit_best_forward(*, identity, sites, output, fit_candidates, evaluate,
                     save_artifact, load_artifact, should_pause=lambda:False):
    """Scan all remaining sites under one accepted bank, then take one winner.

    Each next round refits strictly downstream sites after the winner is applied.
    A nonpositive maximum gain stops the whole trajectory. There is no lookahead
    over synergistic pairs. Candidate caches are bound to the complete upstream
    decision history, so rejected alternatives cannot leak into later rounds.
    """
    if not identity or not sites or len(set(sites))!=len(sites):
        raise ValueError('Explicit identity and unique ordered sites required')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    contract=dict(schema='look_best_forward_v1',identity=identity,sites=list(sites),
        mode='best_forward',test_access=False,ties='off; earlier site; candidate key ascending')
    def assess(bank):
        result=evaluate(tuple(bank))
        if result.get('role')!='development' or not math.isfinite(result['score']):
            raise ValueError('Finite development score required; test is sealed')
        return result
    def checked(row):
        path=(root/row['artifact']).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha256(path)!=row['sha256']:
            raise ValueError('Candidate evidence changed')
        _check_prediction_evidence(row['evidence'])
        return load_artifact(path)
    with (root/'trajectory.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cp=root/'contract.json'
        if cp.exists() and json.loads(cp.read_text())!=contract:raise ValueError('Trajectory identity changed')
        atomic_write_json(contract,cp)
        bank=[];decisions=[];start=0
        while start<len(sites):
            if should_pause():raise SelectionPaused()
            round_index=len(decisions);folder=root/'rounds'/f'{round_index:03d}'
            rid=stable_hash(dict(contract=contract,upstream=decisions,start=start))
            dp=folder/'decision.json'
            if dp.exists():
                d=json.loads(dp.read_text())
                if d['identity']!=rid:raise ValueError('Upstream decision identity changed')
                _check_prediction_evidence(d['baseline'])
                for row in d['candidates']:checked(row)
            else:
                baseline=assess(bank);rows=[]
                for index in range(start,len(sites)):
                    if should_pause():raise SelectionPaused()
                    site=sites[index];cache=folder/f'site_{index:03d}.json'
                    sid=stable_hash(dict(round=rid,index=index))
                    if cache.exists():
                        saved=json.loads(cache.read_text())
                        if saved['identity']!=sid:raise ValueError('Candidate cache identity changed')
                        for row in saved['candidates']:checked(row)
                        rows.extend(saved['candidates']);continue
                    candidates=[];seen=set()
                    for key,artifact in fit_candidates(site,tuple(bank),folder/'fit'/f'{index:03d}'):
                        if should_pause():raise SelectionPaused()
                        if not isinstance(key,str) or key in seen:raise ValueError('Unique string candidate keys required')
                        seen.add(key)
                        path=folder/'artifacts'/f'{index:03d}_{stable_hash(key)}.pt'
                        path.parent.mkdir(parents=True,exist_ok=True);save_artifact(artifact,path)
                        evidence=assess([*bank,artifact])
                        candidates.append(dict(index=index,node=site,key=key,score=evidence['score'],
                            evidence=evidence,artifact=str(path.relative_to(root)),sha256=file_sha256(path)))
                    if not candidates:raise ValueError('No feasible candidates; do not silently skip node')
                    atomic_write_json(dict(identity=sid,candidates=candidates),cache);rows.extend(candidates)
                winner=min(rows,key=lambda r:(-r['score'],r['index'],r['key']))
                enabled=winner['score']>baseline['score']
                d=dict(identity=rid,baseline=baseline,candidates=rows,winner=winner,enabled=enabled,
                    selected_score=winner['score'] if enabled else baseline['score'],
                    reason='strict_improvement' if enabled else 'no_remaining_strict_improvement')
                atomic_write_json(d,dp)
            decisions.append(d)
            if not d['enabled']:break
            bank.append(checked(d['winner']));start=d['winner']['index']+1
        final=assess(bank)
        if final['score']!=decisions[-1]['selected_score']:raise ValueError('Final replay differs from selected trajectory')
        result=dict(schema=contract['schema'],contract_sha256=file_sha256(cp),mode='best_forward',
            decisions=decisions,final=final,test_access=False,scientific_acceptance=False,
            site_attempts=sum(len({r['index'] for r in d['candidates']}) for d in decisions),
            candidate_evaluations=sum(len(d['candidates']) for d in decisions))
        atomic_write_json(result,root/'selection.json');return bank,result
