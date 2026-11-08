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


def test_coefficient_of_variation_relative_spread():
    from eval_harness.stats import coefficient_of_variation

    # stdev of [1, 2, 3, 4] is about 1.291; divided by mean 2.5.
    assert coefficient_of_variation([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.5164, abs=1e-3)
    # Doubling every value doubles both stdev and mean, so CV is unchanged.
    assert coefficient_of_variation([2.0, 4.0, 6.0, 8.0]) == pytest.approx(
        coefficient_of_variation([1.0, 2.0, 3.0, 4.0])
    )
    assert coefficient_of_variation([5.0]) == pytest.approx(0.0)
    assert coefficient_of_variation([2.0, 2.0, 2.0]) == pytest.approx(0.0)
    assert coefficient_of_variation([-1.0, 1.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        coefficient_of_variation([])


def test_standard_error_scales_with_spread_and_count():
    from eval_harness.stats import standard_error

    # stdev of [1, 2, 3, 4] is about 1.291; divided by sqrt(4) = 2.
    assert standard_error([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.6455, abs=1e-3)
    assert standard_error([5.0]) == pytest.approx(0.0)
    assert standard_error([2.0, 2.0, 2.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        standard_error([])


def test_value_range_spread():
    from eval_harness.stats import value_range

    assert value_range([1.0, 4.0, 2.0]) == 3.0
    assert value_range([-2.0, 2.0]) == 4.0
    assert value_range([5.0]) == 0.0
    assert value_range([3.0, 3.0, 3.0]) == 0.0
    with pytest.raises(ValueError):
        value_range([])


def test_sample_variance_bessel_corrected():
    from eval_harness.stats import sample_variance

    # Variance of [1, 2, 3, 4] with denominator n - 1 is 5/3.
    assert sample_variance([1.0, 2.0, 3.0, 4.0]) == pytest.approx(5.0 / 3.0)
    assert sample_variance([5.0]) == pytest.approx(0.0)
    assert sample_variance([2.0, 2.0, 2.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        sample_variance([])


def test_population_variance_uses_n_denominator():
    from eval_harness.stats import population_variance

    # Population variance of [1, 2, 3, 4] with denominator n is 1.25,
    # smaller than the sample variance (5/3) of the same values.
    assert population_variance([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.25)
    assert population_variance([5.0]) == pytest.approx(0.0)
    assert population_variance([2.0, 2.0, 2.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        population_variance([])


def test_root_mean_square_weights_large_values():
    from eval_harness.stats import root_mean_square

    # sqrt((9 + 16) / 2) = sqrt(12.5); RMS sits above the plain mean of 3.5.
    assert root_mean_square([3.0, 4.0]) == pytest.approx(12.5 ** 0.5)
    assert root_mean_square([3.0, 4.0]) > 3.5
    assert root_mean_square([-5.0]) == pytest.approx(5.0)
    assert root_mean_square([0.0, 0.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        root_mean_square([])


def test_geometric_mean_for_ratio_scores():
    from eval_harness.stats import geometric_mean

    assert geometric_mean([1.0, 4.0]) == pytest.approx(2.0)
    assert geometric_mean([2.0, 8.0, 32.0]) == pytest.approx(8.0)
    assert geometric_mean([5.0]) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        geometric_mean([])
    with pytest.raises(ValueError):
        geometric_mean([1.0, 0.0])
    with pytest.raises(ValueError):
        geometric_mean([1.0, -2.0])


def test_harmonic_mean_for_rates():
    from eval_harness.stats import harmonic_mean

    # Harmonic mean of 1 and 4 is 2 / (1 + 1/4) = 1.6, below the arithmetic mean.
    assert harmonic_mean([1.0, 4.0]) == pytest.approx(1.6)
    assert harmonic_mean([2.0, 2.0, 2.0]) == pytest.approx(2.0)
    assert harmonic_mean([5.0]) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        harmonic_mean([])
    with pytest.raises(ValueError):
        harmonic_mean([1.0, 0.0])


def test_midrange_averages_the_extremes():
    from eval_harness.stats import midrange

    assert midrange([1.0, 4.0]) == pytest.approx(2.5)
    assert midrange([2.0, 2.0, 2.0]) == pytest.approx(2.0)
    assert midrange([5.0]) == pytest.approx(5.0)
    assert midrange([-3.0, 7.0, 1.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        midrange([])


def test_z_scores_measure_distance_from_mean():
    from eval_harness.stats import z_scores

    scores = z_scores([1.0, 2.0, 3.0])
    assert scores == pytest.approx([-1.0, 0.0, 1.0])
    # Z-scores of any non-constant list sum to zero.
    assert sum(z_scores([0.2, 0.5, 0.9, 0.4])) == pytest.approx(0.0)
    assert z_scores([5.0]) == [0.0]
    assert z_scores([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        z_scores([])


def test_mean_absolute_deviation_measures_spread_around_mean():
    from eval_harness.stats import mean_absolute_deviation

    assert mean_absolute_deviation([1.0, 2.0, 3.0]) == pytest.approx(2 / 3)
    assert mean_absolute_deviation([5.0]) == pytest.approx(0.0)
    assert mean_absolute_deviation([2.0, 2.0, 2.0]) == pytest.approx(0.0)
    assert mean_absolute_deviation([0.0, 10.0]) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        mean_absolute_deviation([])


def test_interpercentile_range_ignores_extreme_runs():
    from eval_harness.stats import interpercentile_range

    values = [float(i) for i in range(101)]
    assert interpercentile_range(values) == pytest.approx(80.0)
    assert interpercentile_range(values, 25.0, 75.0) == pytest.approx(50.0)
    assert interpercentile_range([3.0, 3.0, 3.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        interpercentile_range(values, 90.0, 10.0)
    with pytest.raises(ValueError):
        interpercentile_range([])
