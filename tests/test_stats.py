"""Tests for eval_harness.stats."""

import pytest

from eval_harness.stats import (
    bootstrap_ci,
    paired_permutation_test,
    summarize,
    win_rate,
)


def test_bootstrap_ci_constant():
    low, high = bootstrap_ci([0.7] * 50, seed=0)
    assert low == pytest.approx(0.7)
    assert high == pytest.approx(0.7)


def test_bootstrap_ci_contains_mean():
    values = [0.0, 0.0, 1.0, 1.0, 0.5, 0.5]
    low, high = bootstrap_ci(values, n_resamples=2000, seed=123)
    assert low <= 0.5 <= high
    assert low < high


def test_bootstrap_ci_deterministic():
    values = [0.1, 0.4, 0.9, 0.2]
    assert bootstrap_ci(values, seed=5) == bootstrap_ci(values, seed=5)


def test_bootstrap_ci_validation():
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], n_resamples=10)


def test_permutation_identical_samples():
    # Identical paired samples: every permutation has |diff| == 0 == observed.
    p = paired_permutation_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], seed=0)
    assert p == pytest.approx(1.0)


def test_permutation_clear_difference():
    # b is uniformly much better than a -> tiny p-value.
    a = [0.0] * 20
    b = [1.0] * 20
    p = paired_permutation_test(a, b, n_resamples=2000, seed=0)
    assert p < 0.01


def test_permutation_validation():
    with pytest.raises(ValueError):
        paired_permutation_test([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        paired_permutation_test([], [])


def test_win_rate_known():
    result = win_rate([1.0, 2.0, 3.0], [1.0, 1.0, 4.0])
    assert result == {
        "a_wins": 1,
        "b_wins": 1,
        "ties": 1,
        "win_rate_a": pytest.approx(0.5),
    }


def test_win_rate_all_ties():
    result = win_rate([1.0, 1.0], [1.0, 1.0])
    assert result["ties"] == 2
    assert result["win_rate_a"] == pytest.approx(0.5)


def test_summarize():
    summary = summarize([1.0, 2.0, 3.0, 4.0])
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["min"] == 1.0
    assert summary["max"] == 4.0
    assert summary["n"] == 4
    assert summary["stdev"] == pytest.approx(1.2909944487358056)


def test_summarize_single_value():
    assert summarize([5.0])["stdev"] == 0.0
