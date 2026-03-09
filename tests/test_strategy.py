"""Tests for strategy document parsing."""

import pytest
from gridmind.core.strategy import Strategy, StrategyLoader, Metric


SAMPLE_STRATEGY = """---
name: test-strategy
domain: sales_pipeline
objective: Maximize reply rate
metrics:
  - name: reply_rate
    direction: higher
    baseline: 0.02
  - name: cost_per_lead
    direction: lower
    baseline: 5.0
variables:
  tone: [casual, professional]
  length: [short, long]
max_iterations: 50
time_budget_seconds: 60
constraints:
  - Keep it short
  - Be professional
---

# Experiment Instructions

Generate an email using the assigned variables.
Test the hypothesis and measure results.
"""


def test_parse_strategy():
    strategy = StrategyLoader.parse(SAMPLE_STRATEGY)
    assert strategy.name == "test-strategy"
    assert strategy.domain == "sales_pipeline"
    assert strategy.objective == "Maximize reply rate"
    assert strategy.max_iterations == 50
    assert strategy.time_budget_seconds == 60


def test_parse_metrics():
    strategy = StrategyLoader.parse(SAMPLE_STRATEGY)
    assert len(strategy.metrics) == 2
    assert strategy.metrics[0].name == "reply_rate"
    assert strategy.metrics[0].direction == "higher"
    assert strategy.metrics[0].baseline == 0.02
    assert strategy.metrics[1].name == "cost_per_lead"
    assert strategy.metrics[1].direction == "lower"


def test_primary_metric():
    strategy = StrategyLoader.parse(SAMPLE_STRATEGY)
    assert strategy.primary_metric is not None
    assert strategy.primary_metric.name == "reply_rate"


def test_parse_variables():
    strategy = StrategyLoader.parse(SAMPLE_STRATEGY)
    assert "tone" in strategy.variables
    assert strategy.variables["tone"] == ["casual", "professional"]
    assert strategy.variables["length"] == ["short", "long"]


def test_parse_constraints():
    strategy = StrategyLoader.parse(SAMPLE_STRATEGY)
    assert len(strategy.constraints) == 2
    assert "Keep it short" in strategy.constraints


def test_parse_experiment_template():
    strategy = StrategyLoader.parse(SAMPLE_STRATEGY)
    assert "Generate an email" in strategy.experiment_template
    assert "# Experiment Instructions" in strategy.experiment_template


def test_invalid_strategy_no_frontmatter():
    with pytest.raises(ValueError, match="YAML frontmatter"):
        StrategyLoader.parse("No frontmatter here")


def test_metric_comparison():
    m = Metric(name="revenue", direction="higher")
    assert m.is_better(10, 5) is True
    assert m.is_better(5, 10) is False

    m2 = Metric(name="cost", direction="lower")
    assert m2.is_better(5, 10) is True
    assert m2.is_better(10, 5) is False
