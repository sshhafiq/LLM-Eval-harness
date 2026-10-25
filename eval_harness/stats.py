"""Statistical helpers for comparing experiment results.

All functions are deterministic given ``seed`` and use only the standard
library:

- :func:`bootstrap_ci` — percentile bootstrap confidence interval for a mean.
- :func:`paired_permutation_test` — exact-ish paired test for whether two
  per-example score lists have different means.
- :func:`win_rate` — head-to-head win/tie/loss counts for two score lists.
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


def summarize(values: Sequence[float]) -> dict:
    """Mean, standard deviation, min, max, and count for a score list."""
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }
