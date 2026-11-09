"""Statistical helpers for comparing experiment results.

All functions are deterministic given ``seed`` and use only the standard
library:

- :func:`bootstrap_ci` — percentile bootstrap confidence interval for a mean.
- :func:`paired_permutation_test` — exact-ish paired test for whether two
  per-example score lists have different means.
- :func:`win_rate` — head-to-head win/tie/loss counts for two score lists.
- :func:`summarize` — mean, median, stdev, min, and max of one score list.
- :func:`percentile` — a single percentile of one score list.
- :func:`interquartile_range` — spread between the 25th and 75th percentiles.
- :func:`median_absolute_deviation` — median distance from the median.
- :func:`standard_error` — standard error of the mean of one score list.
- :func:`coefficient_of_variation` — stdev relative to the mean of one list.
- :func:`value_range` — difference between the max and min of one list.
- :func:`population_variance` — variance of one list, denominator n.
- :func:`root_mean_square` — square root of the mean of squared values.
- :func:`winsorized_mean` — mean after clamping extreme values to percentiles.
- :func:`skewness` — asymmetry of one score list around its mean.
"""

from __future__ import annotations

import random
import statistics
from typing import List, Sequence


def _check_paired(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b):
        raise ValueError(f"paired inputs must have equal length ({len(a)} != {len(b)})")
    if not a:
        raise ValueError("inputs must be non-empty")


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Resamples with replacement ``n_resamples`` times, takes the mean of each
    resample, and returns the ``(1 - ci) / 2`` and ``1 - (1 - ci) / 2``
    percentiles.  Deterministic for a fixed ``seed``.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")
    if n_resamples < 100:
        raise ValueError("n_resamples must be >= 100 for a stable interval")

    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.mean(rng.choices(values, k=n)) for _ in range(n_resamples)
    )
    tail = (1.0 - ci) / 2.0
    lo_idx = max(0, min(n_resamples - 1, int(tail * n_resamples)))
    hi_idx = max(0, min(n_resamples - 1, int((1.0 - tail) * n_resamples)))
    return means[lo_idx], means[hi_idx]


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided paired permutation test for a difference in means.

    For each resample, the sign of every paired difference is flipped with
    probability 0.5; the p-value is the fraction of resamples whose absolute
    mean difference is at least as large as the observed one (with the
    observed arrangement counted, so p is never exactly 0).

    Returns a p-value in (0, 1].  Small p suggests the means genuinely differ.
    """
    a, b = list(a), list(b)
    _check_paired(a, b)
    diffs = [x - y for x, y in zip(a, b)]
    observed = abs(statistics.mean(diffs))

    rng = random.Random(seed)
    extreme = 1  # count the observed arrangement itself
    for _ in range(n_resamples):
        permuted = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(statistics.mean(permuted)) >= observed:
            extreme += 1
    return extreme / (n_resamples + 1)


def win_rate(a: Sequence[float], b: Sequence[float]) -> dict:
    """Head-to-head comparison of two per-example score lists.

    Returns counts of examples where ``a`` beats ``b``, ``b`` beats ``a``,
    and ties, plus ``win_rate_a`` (fraction of non-tied examples won by ``a``,
    or 0.5 when every example ties).
    """
    a, b = list(a), list(b)
    _check_paired(a, b)
    a_wins = sum(1 for x, y in zip(a, b) if x > y)
    b_wins = sum(1 for x, y in zip(a, b) if y > x)
    ties = len(a) - a_wins - b_wins
    decisive = a_wins + b_wins
    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "win_rate_a": a_wins / decisive if decisive else 0.5,
    }


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardized mean difference of two independent score lists.

    Uses the pooled standard deviation. Returns 0.0 when both lists are
    constant (no variance to standardize by) instead of dividing by zero.
    Positive values mean ``a`` scores higher on average than ``b``.
    """
    a, b = list(a), list(b)
    if not a or not b:
        raise ValueError("inputs must be non-empty")
    n_a, n_b = len(a), len(b)
    var_a = statistics.variance(a) if n_a > 1 else 0.0
    var_b = statistics.variance(b) if n_b > 1 else 0.0
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(1, n_a + n_b - 2)
    if pooled <= 0.0:
        return 0.0
    return (statistics.mean(a) - statistics.mean(b)) / (pooled ** 0.5)


def percentile(values: Sequence[float], p: float) -> float:
    """The ``p``-th percentile of ``values`` (0 <= p <= 100).

    Uses linear interpolation between the closest ranks, the same method
    as the default in numpy. A single value returns itself for any ``p``.
    """
    values = sorted(values)
    if not values:
        raise ValueError("values must be non-empty")
    if not 0.0 <= p <= 100.0:
        raise ValueError("p must be in [0, 100]")
    if len(values) == 1:
        return float(values[0])
    rank = (p / 100.0) * (len(values) - 1)
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    frac = rank - low
    return float(values[low] + (values[high] - values[low]) * frac)


def interquartile_range(values: Sequence[float]) -> float:
    """Difference between the 75th and 25th percentiles of ``values``.

    A robust spread measure that ignores outliers in either tail. A
    single value (or a constant list) has an IQR of 0.0.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    return percentile(values, 75) - percentile(values, 25)


def median_absolute_deviation(values: Sequence[float]) -> float:
    """Median of the absolute deviations from the median of ``values``.

    A robust spread measure: unlike the standard deviation it barely
    moves when a single outlier lands in the tail. A single value (or a
    constant list) has a MAD of 0.0.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    med = statistics.median(values)
    return float(statistics.median([abs(v - med) for v in values]))


def trimmed_mean(values: Sequence[float], proportion: float = 0.1) -> float:
    """Mean after dropping the lowest and highest ``proportion`` of values.

    A robust center estimate: a few outlier scores at either end do not
    drag it the way a plain mean moves. ``proportion`` must be in
    [0, 0.5); at 0.5 or above nothing would be left to average.
    """
    values = sorted(values)
    if not values:
        raise ValueError("values must be non-empty")
    if not 0 <= proportion < 0.5:
        raise ValueError("proportion must be in [0, 0.5)")
    k = int(len(values) * proportion)
    trimmed = values[k:len(values) - k] if k else values
    return float(statistics.mean(trimmed))


def standard_error(values: Sequence[float]) -> float:
    """Standard error of the mean of ``values``.

    The sample standard deviation divided by the square root of the
    count. A single value (or a constant list) has a standard error
    of 0.0, since there is no measured spread to scale.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        return 0.0
    return float(statistics.stdev(values) / (len(values) ** 0.5))


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Sample standard deviation of ``values`` divided by its mean.

    A scale-free spread measure, handy for comparing score lists whose
    means differ. A single value (or a constant list) returns 0.0, and
    so does a list whose mean is 0, where the ratio is undefined.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return float(statistics.stdev(values) / mean)


def value_range(values: Sequence[float]) -> float:
    """Difference between the largest and smallest values in ``values``.

    The simplest spread measure there is. A single value (or a
    constant list) has a range of 0.0.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    return float(max(values) - min(values))


def sample_variance(values: Sequence[float]) -> float:
    """Sample variance of ``values`` (Bessel-corrected, denominator n - 1).

    A single value (or a constant list) has a variance of 0.0, matching
    the convention the other helpers here use for degenerate inputs.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        return 0.0
    return float(statistics.variance(values))


def population_variance(values: Sequence[float]) -> float:
    """Population variance of ``values`` (denominator n, not n - 1).

    Use this when the list is the whole population rather than a
    sample of it; for a sample, :func:`sample_variance` is the
    unbiased estimate. A single value (or a constant list) has a
    variance of 0.0.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    return float(statistics.pvariance(values))


def root_mean_square(values: Sequence[float]) -> float:
    """Square root of the mean of the squared values in ``values``.

    A magnitude measure that weights larger scores more heavily than a
    plain mean does, handy for error-like scores where big misses should
    stand out. A single value returns its absolute value.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    return float((sum(v * v for v in values) / len(values)) ** 0.5)


def geometric_mean(values: Sequence[float]) -> float:
    """Geometric mean of ``values``, for ratio-like scores.

    Every value must be positive: the geometric mean is undefined for
    zero or negative inputs, so those raise ``ValueError`` instead of
    returning a misleading number. A single value returns itself.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    if any(v <= 0 for v in values):
        raise ValueError("geometric mean requires positive values")
    return float(statistics.geometric_mean(values))


def harmonic_mean(values: Sequence[float]) -> float:
    """Harmonic mean of ``values``, for averaging rates and ratios.

    Every value must be positive, like the geometric mean: a zero or
    negative input raises ``ValueError``. A single value returns itself,
    and equal values return that value.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    if any(v <= 0 for v in values):
        raise ValueError("harmonic mean requires positive values")
    return float(statistics.harmonic_mean(values))


def midrange(values: Sequence[float]) -> float:
    """Midrange of ``values``: the mean of its minimum and maximum.

    A quick centre estimate from just the extremes, useful as a sanity
    check alongside the median when a score list is short. A single
    value, or a constant list, returns that value.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    return float((min(values) + max(values)) / 2)


def z_scores(values: Sequence[float]) -> list[float]:
    """Z-scores of ``values`` against their own mean and sample stdev.

    Each score comes back as its distance from the mean in standard
    deviations, which makes outlier runs easy to spot in a score list.
    A single value, or a constant list, has no spread, so every z-score
    is 0.0.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        return [0.0]
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    if stdev == 0.0:
        return [0.0] * len(values)
    return [float((v - mean) / stdev) for v in values]


def mean_absolute_deviation(values: Sequence[float]) -> float:
    """Mean absolute deviation of ``values`` around their own mean.

    The average distance from the mean, in the same units as the scores,
    which makes it a more readable spread measure than variance when
    reporting how consistent a model's scores are. A single value, or
    a constant list, returns 0.0.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    mean = statistics.mean(values)
    return float(statistics.mean(abs(v - mean) for v in values))


def interpercentile_range(
    values: Sequence[float], low: float = 10.0, high: float = 90.0
) -> float:
    """Distance between the ``high`` and ``low`` percentiles of ``values``.

    A robust spread measure for score distributions: unlike the full
    range it ignores the most extreme runs on each side, so one outlier
    run does not dominate the reported spread.
    """
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    return percentile(values, high) - percentile(values, low)


def winsorized_mean(values: Sequence[float], proportion: float = 0.1) -> float:
    """Mean of ``values`` after clamping the extremes to percentiles.

    Values below the ``proportion`` percentile are raised to that
    percentile and values above the ``1 - proportion`` percentile are
    lowered to it, so a few outlier runs still count but cannot drag
    the mean the way they would in a plain average. ``proportion``
    must be in [0, 0.5); at 0 the result is the plain mean.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    if not 0 <= proportion < 0.5:
        raise ValueError("proportion must be in [0, 0.5)")
    low = percentile(values, proportion * 100.0)
    high = percentile(values, (1.0 - proportion) * 100.0)
    clamped = [min(max(v, low), high) for v in values]
    return float(statistics.mean(clamped))


def skewness(values: Sequence[float]) -> float:
    """Skewness of ``values``: the population moment ratio m3 / m2 ** 1.5.

    Positive values mean a long right tail (a few unusually high runs),
    negative values a long left tail, and 0.0 a symmetric list. A single
    value, or a constant list, has no spread to be asymmetric, so it
    returns 0.0.
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        return 0.0
    mean = statistics.mean(values)
    m2 = statistics.mean((v - mean) ** 2 for v in values)
    if m2 == 0.0:
        return 0.0
    m3 = statistics.mean((v - mean) ** 3 for v in values)
    return float(m3 / (m2 ** 1.5))


def summarize(values: Sequence[float]) -> dict:
    """Mean, median, standard deviation, min, max, and count for a score list."""
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }
