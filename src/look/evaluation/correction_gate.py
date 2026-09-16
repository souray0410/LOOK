"""A dev-selected, method-specific global correction gate (never per person).

The same locked decision is applied to later evaluation without reading its labels.
This gate is NOT a replacement for a progressive algorithm's per-node search.
"""
import numpy as np
from look.runtime.state import stable_hash

VERSION = 'look_correction_gate_v1'
PATTERNS = ('oct_missing', 'cfp_missing')


def macro_f1(labels, logits):
    y = np.asarray(labels)
    z = np.asarray(logits)
    if (y.ndim != 1 or z.shape != (len(y), 2) or not len(y)
            or not np.isin(y, [0, 1]).all() or not np.isfinite(z).all()):
        raise ValueError('Finite paired binary predictions required')
    pred = z.argmax(1)
    values = []
    for c in (0, 1):
        tp = np.sum((y == c) & (pred == c))
        den = np.sum(y == c) + np.sum(pred == c)
        values.append(2 * tp / den if den else 0.)
    return float(np.mean(values))


def fit_gate(labels, candidate, baseline, *, data_role, identity, method, pattern):
    if data_role != 'development':
        raise ValueError('Gate selection is development-only; test selection prohibited')
    if pattern not in PATTERNS or not identity or not method:
        raise ValueError('Explicit method, pattern and source identity required')
    selected = macro_f1(labels, candidate)
    host = macro_f1(labels, baseline)
    body = dict(schema=VERSION, data_role=data_role, identity=identity, method=method,
                pattern=pattern, enabled=selected > host, metric='macro_f1',
                rule='strict_improvement_else_identity', candidate_macro_f1=selected,
                host_macro_f1=host, scope='whole_candidate_bank_not_per_node',
                test_selected=False)
    return dict(body, sha256=stable_hash(body))


def validate_gate(gate, *, identity, method, pattern):
    body = {k: v for k, v in gate.items() if k != 'sha256'}
    if (gate.get('sha256') != stable_hash(body) or gate.get('schema') != VERSION
            or gate.get('identity') != identity or gate.get('method') != method
            or gate.get('pattern') != pattern or gate.get('data_role') != 'development'
            or gate.get('test_selected') is not False or type(gate.get('enabled')) is not bool
            or gate.get('rule') != 'strict_improvement_else_identity'
            or gate['enabled'] != (gate['candidate_macro_f1'] > gate['host_macro_f1'])):
        raise ValueError('Invalid or mismatched locked gate')


def apply_gate(candidate_logits, baseline_logits, gate, *, identity, method, pattern):
    """No labels or evaluation score accepted: a later test cannot change this gate."""
    validate_gate(gate, identity=identity, method=method, pattern=pattern)
    a, b = np.asarray(candidate_logits), np.asarray(baseline_logits)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError('Unpaired gate logits')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Non-finite gate logits')
    return (a if gate['enabled'] else b).copy()


def gated_bank(bank, gate, *, identity, method, pattern):
    """Apply the locked switch to the actual inference artifact list."""
    validate_gate(gate, identity=identity, method=method, pattern=pattern)
    return list(bank) if gate['enabled'] else []
