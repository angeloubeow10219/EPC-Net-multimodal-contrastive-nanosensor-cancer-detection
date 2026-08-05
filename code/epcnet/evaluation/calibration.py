from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class CalibrationResult:
    expected_error: float
    maximum_error: float
    brier_score: float
    intercept: float
    slope: float
    bin_confidence: FloatArray
    bin_accuracy: FloatArray
    bin_count: IntArray


def calibration(
    labels: IntArray,
    scores: FloatArray,
    bins: int = 15,
) -> CalibrationResult:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.clip(np.digitize(scores, boundaries) - 1, 0, bins - 1)
    confidence = np.zeros(bins)
    accuracy = np.zeros(bins)
    count = np.zeros(bins, dtype=np.int64)
    for index in range(bins):
        selected = assignment == index
        count[index] = int(selected.sum())
        if count[index]:
            confidence[index] = float(scores[selected].mean())
            accuracy[index] = float(labels[selected].mean())
    weight = count / max(count.sum(), 1)
    error = np.abs(confidence - accuracy)
    expected = float(np.sum(weight * error))
    maximum = float(error[count > 0].max(initial=0.0))
    brier = float(np.mean((scores - labels) ** 2))
    clipped = np.clip(scores, 1.0e-6, 1.0 - 1.0e-6)
    logits = np.log(clipped / (1.0 - clipped))
    design = np.column_stack([np.ones_like(logits), logits])
    coefficients, _, _, _ = np.linalg.lstsq(design, labels, rcond=None)
    return CalibrationResult(
        expected,
        maximum,
        brier,
        float(coefficients[0]),
        float(coefficients[1]),
        confidence,
        accuracy,
        count,
    )
