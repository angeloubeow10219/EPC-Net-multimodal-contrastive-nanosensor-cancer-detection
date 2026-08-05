from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def percentile_interval(values: FloatArray, confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def bootstrap_interval(
    labels: IntArray,
    scores: FloatArray,
    statistic: Callable[[IntArray, FloatArray], float],
    resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, FloatArray]:
    generator = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    size = len(labels)
    for index in range(resamples):
        sample = generator.integers(0, size, size=size)
        values[index] = statistic(labels[sample], scores[sample])
    lower, upper = percentile_interval(values, confidence)
    return lower, upper, values


def independent_difference_interval(
    left_labels: IntArray,
    left_scores: FloatArray,
    right_labels: IntArray,
    right_scores: FloatArray,
    statistic: Callable[[IntArray, FloatArray], float],
    resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        left_indices = generator.integers(0, len(left_labels), size=len(left_labels))
        right_indices = generator.integers(0, len(right_labels), size=len(right_labels))
        differences[index] = statistic(
            left_labels[left_indices], left_scores[left_indices]
        ) - statistic(right_labels[right_indices], right_scores[right_indices])
    return percentile_interval(differences, confidence)
