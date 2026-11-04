"""Tests for eval_harness.stats."""

import pytest

from eval_harness.stats import (
    bootstrap_ci,
    cohens_d,
    interquartile_range,
    median_absolute_deviation,
    paired_permutation_test,
    percentile,
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


def test_summarize_median():
    assert summarize([1.0, 2.0, 3.0, 4.0])["median"] == pytest.approx(2.5)
    assert summarize([1.0, 2.0, 3.0])["median"] == pytest.approx(2.0)


def test_cohens_d_positive_and_constant():
    assert cohens_d([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert cohens_d([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == pytest.approx(-1.0)
    assert cohens_d([0.5, 0.5], [0.5, 0.5]) == 0.0
    with pytest.raises(ValueError, match="non-empty"):
        cohens_d([], [1.0])


def test_percentile_endpoints_and_median():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0) == pytest.approx(1.0)
    assert percentile(values, 100) == pytest.approx(4.0)
    assert percentile(values, 50) == pytest.approx(2.5)
    assert percentile([5.0], 90) == pytest.approx(5.0)


def test_interquartile_range():
    assert interquartile_range([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.5)
    assert interquartile_range([5.0]) == pytest.approx(0.0)
    assert interquartile_range([2.0, 2.0, 2.0]) == pytest.approx(0.0)
    # An outlier in the tail does not move the IQR the way it moves stdev.
    assert interquartile_range([1.0, 2.0, 3.0, 4.0, 100.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        interquartile_range([])


def test_median_absolute_deviation():
    assert median_absolute_deviation([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)
    assert median_absolute_deviation([5.0]) == pytest.approx(0.0)
    assert median_absolute_deviation([2.0, 2.0, 2.0]) == pytest.approx(0.0)
    # An outlier barely moves the MAD, unlike the standard deviation.
    assert median_absolute_deviation([1.0, 2.0, 3.0, 4.0, 100.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        median_absolute_deviation([])


def test_percentile_interpolates_and_validates():
    assert percentile([0.0, 10.0], 25) == pytest.approx(2.5)
    # Input order does not matter.
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_trimmed_mean_drops_outliers():
    from eval_harness.stats import trimmed_mean

    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    # proportion 0.2 drops one value from each end: mean of [2, 3, 4]
    assert trimmed_mean(values, 0.2) == 3.0
    assert trimmed_mean(values, 0.0) == 22.0
    assert trimmed_mean([5.0], 0.4) == 5.0


def test_trimmed_mean_validation():
    from eval_harness.stats import trimmed_mean

    with pytest.raises(ValueError):
        trimmed_mean([], 0.1)
    with pytest.raises(ValueError):
        trimmed_mean([1.0, 2.0], 0.5)
    with pytest.raises(ValueError):
        trimmed_mean([1.0, 2.0], -0.1)


def test_standard_error_scales_with_spread_and_count():
    from eval_harness.stats import standard_error

    # stdev of [1, 2, 3, 4] is about 1.291; divided by sqrt(4) = 2.
    assert standard_error([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.6455, abs=1e-3)
    assert standard_error([5.0]) == pytest.approx(0.0)
    assert standard_error([2.0, 2.0, 2.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        standard_error([])
