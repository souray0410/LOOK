from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y_true
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prediction = probabilities.argmax(axis=1)
    num_classes = probabilities.shape[1]
    matrix = confusion_matrix(y_true, prediction, labels=np.arange(num_classes))
    sensitivity, specificity = [], []
    for class_id in range(num_classes):
        tp = matrix[class_id, class_id]
        fn = matrix[class_id].sum() - tp
        fp = matrix[:, class_id].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        sensitivity.append(float(tp / (tp + fn)) if tp + fn else float("nan"))
        specificity.append(float(tn / (tn + fp)) if tn + fp else float("nan"))
    one_hot = np.eye(num_classes)[y_true]
    try:
        macro_auc = float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"))
    except ValueError:
        macro_auc = float("nan")
    return {
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, prediction, average="weighted", zero_division=0)),
        "macro_auroc_ovr": macro_auc,
        "cohen_kappa": float(cohen_kappa_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "sensitivity_per_class": sensitivity,
        "specificity_per_class": specificity,
        "ece_15": expected_calibration_error(y_true, probabilities, 15),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "confusion_matrix": matrix.tolist(),
    }


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
        values.append(float(classification_metrics(y_true[rows], probabilities[rows])[metric]))
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {"estimate": float(classification_metrics(y_true, probabilities)[metric]), "lower": float(lower), "upper": float(upper)}


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
        score_a = classification_metrics(y_true[rows], probabilities_a[rows])[metric]
        score_b = classification_metrics(y_true[rows], probabilities_b[rows])[metric]
        differences.append(float(score_a - score_b))
    lower, upper = np.quantile(differences, [0.025, 0.975])
    p_value = 2 * min(np.mean(np.asarray(differences) <= 0), np.mean(np.asarray(differences) >= 0))
    return {
        "difference": float(classification_metrics(y_true, probabilities_a)[metric] - classification_metrics(y_true, probabilities_b)[metric]),
        "lower": float(lower),
        "upper": float(upper),
        "p_value": float(min(1.0, p_value)),
    }


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted
