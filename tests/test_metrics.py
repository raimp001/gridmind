"""Tests for metrics tracking."""

from gridmind.core.metrics import MetricTracker, MetricDirection


def test_record_first_value():
    tracker = MetricTracker()
    is_best = tracker.record("reply_rate", 0.05, iteration=1, direction="higher")
    assert is_best is True
    assert tracker.get_best("reply_rate") == 0.05


def test_record_improvement():
    tracker = MetricTracker()
    tracker.record("reply_rate", 0.05, iteration=1, direction="higher")
    is_best = tracker.record("reply_rate", 0.08, iteration=2, direction="higher")
    assert is_best is True
    assert tracker.get_best("reply_rate") == 0.08


def test_record_no_improvement():
    tracker = MetricTracker()
    tracker.record("reply_rate", 0.08, iteration=1, direction="higher")
    is_best = tracker.record("reply_rate", 0.05, iteration=2, direction="higher")
    assert is_best is False
    assert tracker.get_best("reply_rate") == 0.08


def test_lower_is_better():
    tracker = MetricTracker()
    tracker.record("cost", 10.0, iteration=1, direction="lower")
    is_best = tracker.record("cost", 5.0, iteration=2, direction="lower")
    assert is_best is True
    assert tracker.get_best("cost") == 5.0

    is_best = tracker.record("cost", 8.0, iteration=3, direction="lower")
    assert is_best is False
    assert tracker.get_best("cost") == 5.0


def test_history_tracking():
    tracker = MetricTracker()
    tracker.record("reply_rate", 0.05, iteration=1, direction="higher")
    tracker.record("reply_rate", 0.08, iteration=2, direction="higher")
    tracker.record("reply_rate", 0.03, iteration=3, direction="higher")
    assert len(tracker.history) == 3


def test_summary():
    tracker = MetricTracker()
    tracker.record("reply_rate", 0.05, iteration=1, direction="higher")
    tracker.record("reply_rate", 0.08, iteration=2, direction="higher")

    summary = tracker.summary()
    assert "reply_rate" in summary
    assert summary["reply_rate"]["best_value"] == 0.08
    assert summary["reply_rate"]["best_iteration"] == 2
    assert summary["reply_rate"]["total_measurements"] == 2


def test_get_best_unknown_metric():
    tracker = MetricTracker()
    assert tracker.get_best("nonexistent") is None
