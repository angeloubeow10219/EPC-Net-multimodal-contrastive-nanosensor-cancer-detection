from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SiteEstimate:
    name: str
    estimate: float
    variance: float
    sample_size: int


@dataclass(frozen=True)
class RandomEffectsResult:
    estimate: float
    lower: float
    upper: float
    tau_squared: float
    heterogeneity: float
    degrees_of_freedom: int
    weights: FloatArray


@dataclass(frozen=True)
class ConcordanceResult:
    kappa: float
    observed: float
    expected: float
    matrix: IntArray


def cohen_kappa(left: IntArray, right: IntArray) -> ConcordanceResult:
    if left.shape != right.shape:
        raise ValueError("prediction arrays must have equal shapes")
    categories = np.union1d(left, right)
    matrix = np.zeros((len(categories), len(categories)), dtype=np.int64)
    mapping = {int(value): index for index, value in enumerate(categories)}
    for first, second in zip(left, right):
        matrix[mapping[int(first)], mapping[int(second)]] += 1
    total = matrix.sum()
    observed = float(np.trace(matrix) / total) if total else float("nan")
    left_marginal = matrix.sum(axis=1)
    right_marginal = matrix.sum(axis=0)
    expected = (
        float(np.dot(left_marginal, right_marginal) / (total * total)) if total else float("nan")
    )
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    return ConcordanceResult(kappa, observed, expected, matrix)


def dersimonian_laird(
    estimates: list[SiteEstimate], confidence: float = 0.95
) -> RandomEffectsResult:
    if len(estimates) < 2:
        raise ValueError("random-effects analysis requires at least two sites")
    effects = np.asarray([item.estimate for item in estimates], dtype=np.float64)
    variances = np.asarray([item.variance for item in estimates], dtype=np.float64)
    fixed_weights = 1.0 / np.maximum(variances, np.finfo(np.float64).eps)
    fixed_effect = float(np.sum(fixed_weights * effects) / np.sum(fixed_weights))
    heterogeneity = float(np.sum(fixed_weights * (effects - fixed_effect) ** 2))
    degrees = len(estimates) - 1
    denominator = np.sum(fixed_weights) - np.sum(fixed_weights**2) / np.sum(fixed_weights)
    tau_squared = max(0.0, (heterogeneity - degrees) / denominator)
    random_weights = 1.0 / (variances + tau_squared)
    pooled = float(np.sum(random_weights * effects) / np.sum(random_weights))
    standard_error = float(np.sqrt(1.0 / np.sum(random_weights)))
    critical = float(norm.ppf(0.5 + confidence / 2.0))
    return RandomEffectsResult(
        pooled,
        pooled - critical * standard_error,
        pooled + critical * standard_error,
        tau_squared,
        heterogeneity,
        degrees,
        random_weights,
    )


def site_spread(estimates: list[SiteEstimate]) -> float:
    values = np.asarray([item.estimate for item in estimates])
    return float(values.max() - values.min())


def mcid_crossing_rate(improvements: FloatArray, mcid: float = 0.05) -> float:
    if not len(improvements):
        return float("nan")
    return float(np.mean(improvements >= mcid))


def benjamini_hochberg(p_values: FloatArray, q: float = 0.05) -> NDArray[np.bool_]:
    count = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    bounds = q * np.arange(1, count + 1) / count
    passing = ranked <= bounds
    decisions = np.zeros(count, dtype=bool)
    if passing.any():
        final = int(np.flatnonzero(passing)[-1])
        decisions[order[: final + 1]] = True
    return decisions


def sensitivity_variance(sensitivity: float, positives: int) -> float:
    if positives <= 0:
        raise ValueError("positive sample count must be greater than zero")
    return sensitivity * (1.0 - sensitivity) / positives


def specificity_variance(specificity: float, negatives: int) -> float:
    if negatives <= 0:
        raise ValueError("negative sample count must be greater than zero")
    return specificity * (1.0 - specificity) / negatives


def prevalence_adjusted_ppv(sensitivity: float, specificity: float, prevalence: float) -> float:
    numerator = sensitivity * prevalence
    denominator = numerator + (1.0 - specificity) * (1.0 - prevalence)
    return numerator / denominator if denominator else float("nan")


def prevalence_adjusted_npv(sensitivity: float, specificity: float, prevalence: float) -> float:
    numerator = specificity * (1.0 - prevalence)
    denominator = numerator + (1.0 - sensitivity) * prevalence
    return numerator / denominator if denominator else float("nan")


def independent_effect_size(
    left_mean: float,
    right_mean: float,
    left_std: float,
    right_std: float,
    left_size: int,
    right_size: int,
) -> float:
    numerator = (left_size - 1) * left_std**2 + (right_size - 1) * right_std**2
    denominator = left_size + right_size - 2
    pooled = np.sqrt(numerator / denominator)
    return (left_mean - right_mean) / pooled if pooled else float("nan")


def noninferiority_z(
    experimental: float,
    reference: float,
    experimental_variance: float,
    reference_variance: float,
    margin: float,
) -> tuple[float, float]:
    difference = experimental - reference + margin
    standard_error = np.sqrt(experimental_variance + reference_variance)
    statistic = difference / standard_error
    p_value = float(norm.sf(-statistic))
    return float(statistic), p_value


def pairwise_concordance(
    predictions: dict[str, IntArray],
) -> dict[tuple[str, str], ConcordanceResult]:
    names = sorted(predictions)
    results: dict[tuple[str, str], ConcordanceResult] = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            results[(left_name, right_name)] = cohen_kappa(
                predictions[left_name],
                predictions[right_name],
            )
    return results


def stratified_rates(
    labels: IntArray,
    predictions: IntArray,
    strata: IntArray,
) -> dict[int, float]:
    results: dict[int, float] = {}
    for stratum in np.unique(strata):
        selected = strata == stratum
        positives = labels[selected] == 1
        denominator = int(positives.sum())
        numerator = int(np.sum(predictions[selected][positives] == 1))
        results[int(stratum)] = numerator / denominator if denominator else float("nan")
    return results


def max_stratum_gap(rates: dict[int, float]) -> float:
    values = np.asarray([value for value in rates.values() if np.isfinite(value)])
    return float(values.max() - values.min()) if len(values) else float("nan")


def leave_one_site_out_indices(sites: IntArray) -> list[tuple[IntArray, IntArray]]:
    splits: list[tuple[IntArray, IntArray]] = []
    indices = np.arange(len(sites))
    for site in np.unique(sites):
        test = indices[sites == site]
        train = indices[sites != site]
        splits.append((train, test))
    return splits
