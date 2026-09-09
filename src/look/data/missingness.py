"""Paired, label-independent missingness shared across every experimental case."""
from __future__ import annotations
import hashlib
import math
from collections import Counter
from look.runtime.state import stable_hash

PROTOCOL = 'nested_exact_count_fixed_direction_v1'


def missingness_plan(participant_ids, ratio: float, seed: int = 3407):
    ids = [str(pid) for pid in participant_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Missingness requires nonempty unique participant IDs')
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('Missingness ratio must lie in [0, 1]')
    # Domain-separated hash order; neither labels nor backbone seed enter this rule.
    ordered = sorted(ids, key=lambda pid: (hashlib.sha256(f'{PROTOCOL}:{seed}:{pid}'.encode()).hexdigest(), pid))
    count = math.floor(ratio * len(ids) + 0.5)
    offset = int(hashlib.sha256(f'{PROTOCOL}:direction:{seed}'.encode()).hexdigest(), 16) % 2
    plan = {pid: ('oct_missing' if (rank + offset) % 2 == 0 else 'cfp_missing')
            if rank < count else 'complete' for rank, pid in enumerate(ordered)}
    counts = Counter(plan.values())
    metadata = dict(protocol=PROTOCOL, seed=seed, requested_ratio=ratio,
        actual_ratio=count / len(ids), total_participants=len(ids), missing_participants=count,
        counts={key: counts[key] for key in ('complete', 'oct_missing', 'cfp_missing')},
        participant_set_sha256=stable_hash(sorted(ids)), assignment_sha256=stable_hash(plan),
        rule='round-half-up count; hash-ranked prefix; alternating fixed missing direction; both eyes; one modality always retained')
    return plan, metadata
