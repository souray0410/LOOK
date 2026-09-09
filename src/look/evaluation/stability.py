"""LOOK-only binary scoring; backbone training identity is deliberately unchanged."""
from typing import Dict
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score, average_precision_score
from look.evaluation.metrics import classification_metrics, expected_calibration_error


def probabilities_from_logits(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2 or not np.isfinite(logits).all():
        raise ValueError('Finite two-class logits required')
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def logit_metrics(y_true, logits):
    y = np.asarray(y_true, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    probs = probabilities_from_logits(logits)
    prediction = logits.argmax(axis=1)
    result = classification_metrics(y, np.eye(2)[prediction])
    scores = logits[:, 1] - logits[:, 0]
    auc, ap = [], []
    for c, score in ((0, -scores), (1, scores)):
        target = (y == c).astype(int)
        auc.append(float(roc_auc_score(target, score)) if len(np.unique(target)) == 2 else float('nan'))
        ap.append(float(average_precision_score(target, score)) if target.any() else float('nan'))
    result.update(macro_auroc_ovr=auc[1], macro_auprc_ovr=float(np.nanmean(ap)),
        auroc_per_class=auc, auprc_per_class=ap,
        ece_15=expected_calibration_error(y, probs),
        multiclass_brier=float(np.square(probs-np.eye(2)[y]).sum(axis=1).mean()),
        negative_log_likelihood=float((logsumexp(logits, axis=1)-logits[np.arange(len(y)),y]).mean()),
        ranking_score='logit_1_minus_logit_0',
        probability_saturation={'rows_float64': int(((probs == 0)|(probs == 1)).any(axis=1).sum()),
            'rows_float32': int(((probs.astype(np.float32) == 0)|(probs.astype(np.float32) == 1)).any(axis=1).sum()),
            'total_rows': len(y), 'max_abs_logit_difference': float(np.abs(scores).max())})
    return result


def participant_cluster_bootstrap(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    participant_ids: np.ndarray,
    metric: str = "macro_f1",
    iterations: int = 2000,
    seed: int = 3407,
) -> Dict[str, float]:
    participant_ids = np.asarray(participant_ids).astype(str)
    unique = np.unique(participant_ids)
    indices = {participant: np.where(participant_ids == participant)[0] for participant in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[participant] for participant in sampled])
        values.append(float(logit_metrics(y_true[rows], probabilities[rows])[metric]))
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite bootstrap replicates")
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {"estimate": float(logit_metrics(y_true, probabilities)[metric]), "lower": float(lower), "upper": float(upper), "valid_replicates": len(values), "requested_replicates": iterations}


def paired_participant_bootstrap(
    y_true: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    participant_ids: np.ndarray,
    metric: str = "macro_f1",
    iterations: int = 2000,
    seed: int = 3407,
) -> Dict[str, float]:
    participant_ids = np.asarray(participant_ids).astype(str)
    unique = np.unique(participant_ids)
    indices = {participant: np.where(participant_ids == participant)[0] for participant in unique}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[participant] for participant in sampled])
        score_a = logit_metrics(y_true[rows], probabilities_a[rows])[metric]
        score_b = logit_metrics(y_true[rows], probabilities_b[rows])[metric]
        differences.append(float(score_a - score_b))
    differences = np.asarray(differences)
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        raise ValueError("No finite paired bootstrap replicates")
    lower, upper = np.quantile(differences, [0.025, 0.975])
    p_value = 2 * min(np.mean(np.asarray(differences) <= 0), np.mean(np.asarray(differences) >= 0))
    return {
        "difference": float(logit_metrics(y_true, probabilities_a)[metric] - logit_metrics(y_true, probabilities_b)[metric]),
        "lower": float(lower),
        "upper": float(upper),
        "p_value": float(min(1.0, p_value)),
        "valid_replicates": len(differences), "requested_replicates": iterations,
    }


