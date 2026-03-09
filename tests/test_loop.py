"""Tests for the autonomous research loop (dry run mode)."""

import json
import tempfile
from pathlib import Path

from gridmind.core.loop import LoopConfig, ResearchLoop
from gridmind.core.experiment import ExperimentStatus


STRATEGY_CONTENT = """---
name: test-loop
domain: sales_pipeline
objective: Maximize reply rate
metrics:
  - name: reply_rate
    direction: higher
    baseline: 0.02
variables:
  tone: [casual, professional, consultative]
  length: [short, medium]
max_iterations: 5
time_budget_seconds: 10
---

# Experiment Instructions

Generate a test email. This is a dry run.
"""


def test_dry_run_loop():
    """Test the full loop in dry-run mode (no LLM calls)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        strategy_path = Path(tmpdir) / "test.md"
        strategy_path.write_text(STRATEGY_CONTENT)

        config = LoopConfig(
            strategy_path=str(strategy_path),
            results_dir=str(Path(tmpdir) / "results"),
            dry_run=True,
        )

        events = []
        loop = ResearchLoop(config)
        loop.on_event(lambda t, d: events.append((t, d)))

        state = loop.run()

        assert state.status == "completed"
        assert len(state.experiments) == 5
        assert state.current_iteration == 5

        # Check events fired
        event_types = [e[0] for e in events]
        assert "strategy_loaded" in event_types
        assert "loop_started" in event_types
        assert "loop_completed" in event_types

        # Check results saved
        results_dir = Path(tmpdir) / "results" / "test-loop"
        assert (results_dir / "experiments.json").exists()
        assert (results_dir / "summary.json").exists()
        assert (results_dir / "metrics_history.json").exists()

        summary = json.loads((results_dir / "summary.json").read_text())
        assert summary["total_iterations"] == 5


def test_dry_run_produces_metrics():
    """Test that dry run produces reasonable metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        strategy_path = Path(tmpdir) / "test.md"
        strategy_path.write_text(STRATEGY_CONTENT)

        config = LoopConfig(
            strategy_path=str(strategy_path),
            results_dir=str(Path(tmpdir) / "results"),
            dry_run=True,
        )

        loop = ResearchLoop(config)
        state = loop.run()

        # Should have metrics tracked
        assert state.metrics.get_best("reply_rate") is not None
        best = state.metrics.get_best("reply_rate")
        assert 0.0 <= best <= 1.0

        # Should have history
        assert len(state.metrics.history) > 0


def test_event_callbacks():
    """Test that events are properly emitted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        strategy_path = Path(tmpdir) / "test.md"
        strategy_path.write_text(STRATEGY_CONTENT)

        config = LoopConfig(
            strategy_path=str(strategy_path),
            results_dir=str(Path(tmpdir) / "results"),
            dry_run=True,
        )

        iteration_events = []
        loop = ResearchLoop(config)
        loop.on_event(
            lambda t, d: iteration_events.append(d["iteration"])
            if t == "iteration_completed"
            else None
        )

        loop.run()
        assert iteration_events == [1, 2, 3, 4, 5]
