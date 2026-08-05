from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, roc_curve


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class BinaryMetrics:
    sensitivity: float
    specificity: float
    auc: float
    f1: float
    ppv: float
    npv: float
    threshold: float
    brier: float


def threshold_at_specificity(labels: IntArray, scores: FloatArray, target: float) -> float:
    false_positive_rate, _, thresholds = roc_curve(labels, scores)
    valid = np.flatnonzero(false_positive_rate <= 1.0 - target)
    if not len(valid):
        return float("inf")
    return float(thresholds[valid[-1]])


def confusion(labels: IntArray, scores: FloatArray, threshold: float) -> tuple[int, int, int, int]:
    prediction = scores >= threshold
    positive = labels == 1
    negative = ~positive
    true_positive = int(np.sum(prediction & positive))
    false_positive = int(np.sum(prediction & negative))
    true_negative = int(np.sum(~prediction & negative))
    false_negative = int(np.sum(~prediction & positive))
    return true_positive, false_positive, true_negative, false_negative


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def binary_metrics(
    labels: IntArray,
    scores: FloatArray,
    specificity_target: float = 0.99,
    fixed_threshold: float | None = None,
) -> BinaryMetrics:
    threshold = (
        threshold_at_specificity(labels, scores, specificity_target)
        if fixed_threshold is None
        else fixed_threshold
    )
    true_positive, false_positive, true_negative, false_negative = confusion(
        labels,
        scores,
        threshold,
    )
    sensitivity = safe_ratio(true_positive, true_positive + false_negative)
    specificity = safe_ratio(true_negative, true_negative + false_positive)
    ppv = safe_ratio(true_positive, true_positive + false_positive)
    npv = safe_ratio(true_negative, true_negative + false_negative)
    f1 = safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    auc = float(roc_auc_score(labels, scores))
    brier = float(np.mean((scores - labels) ** 2))
    return BinaryMetrics(sensitivity, specificity, auc, f1, ppv, npv, threshold, brier)
