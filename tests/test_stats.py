"""Tests for statistical significance testing."""

import math

from gridmind.core.stats import (
    welch_t_test,
    confidence_interval,
    minimum_sample_size,
    metric_is_improving,
)


def test_welch_t_test_identical_groups():
    a = [0.5, 0.5, 0.5, 0.5, 0.5]
    b = [0.5, 0.5, 0.5, 0.5, 0.5]
    result = welch_t_test(a, b)
    assert not result.is_significant
    assert result.improvement == 0.0


def test_welch_t_test_clearly_different():
    a = [0.1, 0.12, 0.11, 0.09, 0.1, 0.13, 0.08, 0.11, 0.1, 0.12]
    b = [0.9, 0.88, 0.91, 0.87, 0.9, 0.92, 0.89, 0.91, 0.88, 0.9]
    result = welch_t_test(a, b)
    assert result.is_significant
    assert result.improvement > 5.0  # Huge improvement
    assert result.effect_label == "large"
    assert result.p_value < 0.01


def test_welch_t_test_small_improvement():
    a = [0.50, 0.51, 0.49, 0.52, 0.50, 0.48, 0.51, 0.50]
    b = [0.52, 0.53, 0.51, 0.54, 0.52, 0.50, 0.53, 0.52]
    result = welch_t_test(a, b)
    assert result.improvement > 0
    assert result.mean_b > result.mean_a


def test_welch_t_test_insufficient_data():
    result = welch_t_test([0.5], [0.9])
    assert not result.is_significant
    assert result.effect_label == "insufficient_data"


def test_welch_t_test_summary():
    a = [0.1, 0.12, 0.11, 0.09, 0.1]
    b = [0.5, 0.48, 0.51, 0.47, 0.5]
    result = welch_t_test(a, b)
    summary = result.summary()
    assert "improved" in summary or "declined" in summary
    assert "p=" in summary


def test_confidence_interval_basic():
    values = [0.5, 0.6, 0.4, 0.55, 0.45]
    lower, mean, upper = confidence_interval(values)
    assert lower < mean < upper
    assert abs(mean - 0.5) < 0.01


def test_confidence_interval_single_value():
    lower, mean, upper = confidence_interval([0.5])
    assert lower == mean == upper == 0.5


def test_confidence_interval_tight():
    """Low variance = tight CI."""
    values = [0.500, 0.501, 0.499, 0.500, 0.501, 0.499, 0.500]
    lower, mean, upper = confidence_interval(values)
    assert upper - lower < 0.01  # Very tight


def test_minimum_sample_size():
    # Large effect = fewer samples needed
    n_large = minimum_sample_size(effect_size=0.8)
    n_small = minimum_sample_size(effect_size=0.2)
    assert n_large < n_small
    assert n_large >= 2
    assert n_small >= 2


def test_minimum_sample_size_zero_effect():
    n = minimum_sample_size(effect_size=0)
    assert n == 1000  # Can't detect zero effect


def test_metric_is_improving_upward():
    """Clearly improving metric should be detected."""
    # Early: around 0.1, Late: around 0.9
    history = [0.1 + i * 0.001 for i in range(20)] + [0.9 + i * 0.001 for i in range(20)]
    result = metric_is_improving(history, window=10)
    assert result is not None
    assert result.is_significant
    assert result.improvement > 0


def test_metric_is_improving_flat():
    """Flat metric should not show improvement."""
    import random
    random.seed(42)
    history = [0.5 + random.gauss(0, 0.01) for _ in range(40)]
    result = metric_is_improving(history, window=10)
    assert result is not None
    # With such small noise, shouldn't be significantly improving
    # (could be significant by chance, but improvement should be small)
    assert abs(result.improvement) < 0.1


def test_metric_is_improving_insufficient_data():
    result = metric_is_improving([0.5, 0.6, 0.7], window=10)
    assert result is None  # Not enough data


def test_effect_size_labels():
    # Negligible: same data with tiny noise difference
    a = [0.50, 0.51, 0.49, 0.50, 0.51, 0.49, 0.50, 0.50, 0.51, 0.49]
    b = [0.501, 0.511, 0.491, 0.501, 0.511, 0.491, 0.501, 0.501, 0.511, 0.491]
    result = welch_t_test(a, b)
    assert result.effect_label in ["negligible", "small"]

    # Large
    a2 = [0.1, 0.12, 0.11, 0.09, 0.1]
    b2 = [0.9, 0.92, 0.91, 0.89, 0.9]
    result2 = welch_t_test(a2, b2)
    assert result2.effect_label == "large"


def test_confidence_level_parameter():
    a = [0.1, 0.12, 0.11, 0.09, 0.1]
    b = [0.5, 0.48, 0.51, 0.47, 0.5]
    result_95 = welch_t_test(a, b, confidence=0.95)
    result_99 = welch_t_test(a, b, confidence=0.99)
    assert result_95.confidence_level == 0.95
    assert result_99.confidence_level == 0.99
