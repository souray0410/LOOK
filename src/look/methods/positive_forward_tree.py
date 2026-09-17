"""Restartable downstream tree retaining every strictly beneficial site extension.

Only the best candidate *within each site* becomes a child. Children are fitted
under their own complete accepted prefix; a rejected extension has no children.
This is development selection, not an exhaustive search or independent evidence.
"""
import fcntl
import json
from pathlib import Path

from look.runtime.state import atomic_write_json, file_sha256, stable_hash

VERSION = 'look_positive_forward_tree_v1'


def tree_contract(identity, sites):
    return dict(schema=VERSION, identity=identity, sites=list(sites),
                mode='positive_forward_tree', test_access=False,
                expansion='all_strictly_positive_site_winners; downstream_only',
                ties='off; candidate_key; final_fewer_sites_then_lexicographic_path')


def prefix_identity(contract, path):
    return stable_hash(dict(contract=contract, prefix=path))


def candidate_identity(contract, path, index):
    return stable_hash(dict(prefix=prefix_identity(contract, path), index=index))


def fit_positive_forward_tree(*, identity, sites, output, fit_candidates, evaluate,
                              save_artifact, load_artifact, should_pause=lambda: False):
    from look.methods.independent_greedy import SelectionPaused, _check_prediction_evidence
    if not identity or not sites or len(set(sites)) != len(sites):
        raise ValueError('Explicit identity and unique ordered sites required')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    contract = tree_contract(identity, sites)

    def assess(bank):
        value = evaluate(tuple(bank))
        _check_prediction_evidence(value)
        return value

    def checked(row, index):
        path = (root / row['artifact']).resolve()
        if (not path.is_relative_to(root.resolve()) or file_sha256(path) != row['sha256']
                or row['index'] != index or row['node'] != sites[index]
                or row['score'] != row['evidence']['score']
                or not isinstance(row['key'], str)):
            raise ValueError('Tree candidate evidence changed')
        _check_prediction_evidence(row['evidence'])
        return load_artifact(path)

    def descriptor(row):
        return {k: row[k] for k in ('index', 'node', 'key', 'artifact', 'sha256')}

    def rank(path, evidence):
        return (-evidence['score'], len(path), tuple((r['index'], r['key']) for r in path))

    with (root / 'trajectory.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cp = root / 'contract.json'
        if cp.exists() and json.loads(cp.read_text()) != contract:
            raise ValueError('Tree identity changed')
        atomic_write_json(contract, cp)
        # Reconstruct the frontier from verified committed site evidence on every
        # restart. A stale status or frontier is never an execution authority.
        queue = [([], [], None)]
        decisions = []
        best = None
        site_attempts = candidate_evaluations = 0
        cursor = 0
        while cursor < len(queue):
            if should_pause():
                raise SelectionPaused()
            path, bank, inherited = queue[cursor]
            pid = prefix_identity(contract, path)
            folder = root / 'prefixes' / ('root' if not path else pid)
            bp = folder / 'baseline.json'
            if bp.exists():
                saved = json.loads(bp.read_text())
                if saved['identity'] != pid:
                    raise ValueError('Tree prefix identity changed')
                baseline = saved['evidence']
                _check_prediction_evidence(baseline)
                if inherited is not None and baseline != inherited:
                    raise ValueError('Tree child baseline differs from its parent candidate')
            else:
                baseline = inherited if inherited is not None else assess(bank)
                atomic_write_json(dict(identity=pid, evidence=baseline), bp)
            if best is None or rank(path, baseline) < rank(best[0], best[2]):
                best = (path, bank, baseline)
            rows = []
            children = []
            start = path[-1]['index'] + 1 if path else 0
            for index in range(start, len(sites)):
                if should_pause():
                    raise SelectionPaused()
                site = sites[index]
                sid = candidate_identity(contract, path, index)
                cache = folder / f'site_{index:03d}.json'
                artifacts = {}
                if cache.exists():
                    saved = json.loads(cache.read_text())
                    if saved['identity'] != sid:
                        raise ValueError('Tree site or upstream identity changed')
                    candidates = saved['candidates']
                    if not candidates or len({r['key'] for r in candidates}) != len(candidates):
                        raise ValueError('Invalid tree candidates')
                    for row in candidates:
                        artifacts[row['key']] = checked(row, index)
                else:
                    candidates = []
                    for key, artifact in fit_candidates(site, tuple(bank), folder / 'fit' / f'{index:03d}'):
                        if should_pause():
                            raise SelectionPaused()
                        if not isinstance(key, str) or key in artifacts:
                            raise ValueError('Unique string candidate keys required')
                        target = folder / 'artifacts' / f'{index:03d}_{stable_hash(key)}.pt'
                        target.parent.mkdir(parents=True, exist_ok=True)
                        save_artifact(artifact, target)
                        evidence = assess([*bank, artifact])
                        row = dict(index=index, node=site, key=key, score=evidence['score'],
                                   evidence=evidence, artifact=str(target.relative_to(root)),
                                   sha256=file_sha256(target))
                        candidates.append(row)
                        artifacts[key] = artifact
                    if not candidates:
                        raise ValueError('No feasible candidates; do not silently skip node')
                    atomic_write_json(dict(identity=sid, candidates=candidates), cache)
                winner = min(candidates, key=lambda r: (-r['score'], r['key']))
                enabled = winner['score'] > baseline['score']
                rows.extend(candidates)
                site_attempts += 1
                candidate_evaluations += len(candidates)
                if enabled:
                    child_path = [*path, descriptor(winner)]
                    child_bank = [*bank, artifacts[winner['key']]]
                    children.append(dict(path=child_path, winner=winner))
                    queue.append((child_path, child_bank, winner['evidence']))
                    if rank(child_path, winner['evidence']) < rank(best[0], best[2]):
                        best = (child_path, child_bank, winner['evidence'])
                atomic_write_json(dict(schema=VERSION, state='running', active_prefix=path,
                    active_site=site, completed_prefixes=cursor, discovered_prefixes=len(queue),
                    site_attempts=site_attempts, candidate_evaluations=candidate_evaluations,
                    best_path=best[0], best_score=best[2]['score'], test_access=False), root / 'tree_progress.json')
            decision = dict(identity=pid, path=path, baseline=baseline, candidates=rows,
                            children=children, terminal=not children)
            atomic_write_json(decision, folder / 'decision.json')
            decisions.append(decision)
            cursor += 1
        final = assess(best[1])
        if final['score'] != best[2]['score']:
            raise ValueError('Final tree replay differs from selected path')
        result = dict(schema=VERSION, contract_sha256=file_sha256(cp),
            mode='positive_forward_tree', decisions=decisions, final=final,
            selected_path=best[0], site_attempts=site_attempts,
            candidate_evaluations=candidate_evaluations, prefix_count=len(queue),
            test_access=False, scientific_acceptance=False)
        atomic_write_json(result, root / 'selection.json')
        atomic_write_json(dict(schema=VERSION, state='completed', completed_prefixes=cursor,
            site_attempts=site_attempts, candidate_evaluations=candidate_evaluations,
            best_path=best[0], best_score=final['score'], test_access=False), root / 'tree_progress.json')
        return best[1], result
