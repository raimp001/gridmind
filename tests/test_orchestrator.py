"""Tests for multi-loop orchestrator and campaign runner."""

import json
from pathlib import Path

from gridmind.core.orchestrator import (
    CampaignConfig,
    Orchestrator,
    SignalSpeed,
    DOMAIN_SIGNAL_SPEED,
    estimate_experiments,
)
from gridmind.core.agent import AgentConfig


def test_signal_speed_mapping():
    """All 13 domains should have a signal speed."""
    assert len(DOMAIN_SIGNAL_SPEED) == 13
    fast = [k for k, v in DOMAIN_SIGNAL_SPEED.items() if v == SignalSpeed.FAST]
    medium = [k for k, v in DOMAIN_SIGNAL_SPEED.items() if v == SignalSpeed.MEDIUM]
    slow = [k for k, v in DOMAIN_SIGNAL_SPEED.items() if v == SignalSpeed.SLOW]
    assert len(fast) >= 2
    assert len(medium) >= 3
    assert len(slow) >= 3


def test_estimate_experiments():
    est = estimate_experiments()
    assert est["loops"] == 13
    assert est["yearly_experiments"] > 100000  # 13 * 100 * 365
    assert "vs_traditional" in est
    assert est["signal_tiers"]["fast_loops"] >= 2


def test_campaign_dry_run(tmp_path):
    """Run a full campaign with multiple strategies."""
    # Create strategy files
    for name, domain in [("email", "sales_pipeline"), ("ads", "ad_creative")]:
        (tmp_path / f"{name}.md").write_text(f'''---
name: {name}-test
domain: {domain}
objective: Test {name}
metrics:
  - name: score
    direction: higher
    baseline: 0.3
variables:
  variant: [a, b, c]
max_iterations: 5
---

# Test

Generate a {name} variant.
''')

    config = CampaignConfig(
        name="test-campaign",
        strategies_dir=str(tmp_path),
        results_dir=str(tmp_path / "results"),
        agent_config=AgentConfig(),
        max_workers=2,
        dry_run=True,
    )

    orchestrator = Orchestrator(config)
    keys = orchestrator.add_strategies_from_dir()
    assert len(keys) == 2

    results = orchestrator.run()
    assert len(results) == 2

    # Check campaign summary was saved
    summary_path = tmp_path / "results" / "test-campaign" / "campaign_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["total_loops"] == 2
    assert summary["total_experiments"] == 10  # 5 * 2


def test_campaign_cross_loop_insights(tmp_path):
    """Cross-loop insights should be extracted when metrics improve enough."""
    # Strategy with high improvement potential (low baseline)
    (tmp_path / "email.md").write_text('''---
name: email-test
domain: sales_pipeline
objective: Test cross-loop
metrics:
  - name: reply_rate
    direction: higher
    baseline: 0.01
variables:
  variant: [a, b, c]
max_iterations: 20
---

# Test
Generate email.
''')

    config = CampaignConfig(
        name="insight-test",
        strategies_dir=str(tmp_path),
        results_dir=str(tmp_path / "results"),
        max_workers=1,
        dry_run=True,
        enable_cross_learning=True,
    )

    orchestrator = Orchestrator(config)
    orchestrator.add_strategies_from_dir()
    orchestrator.run()

    # With 20 iterations and 0.01 baseline, mock evaluator should
    # produce enough improvement to generate insights
    # (insights threshold is 20% improvement)
    assert len(orchestrator.insights) >= 1


def test_orchestrator_status(tmp_path):
    (tmp_path / "test.md").write_text('''---
name: status-test
domain: ad_creative
objective: Test status
metrics:
  - name: click_rate
    direction: higher
    baseline: 0.01
variables:
  v: [a, b]
max_iterations: 3
---

# Test
''')

    config = CampaignConfig(
        name="status-test",
        strategies_dir=str(tmp_path),
        results_dir=str(tmp_path / "results"),
        dry_run=True,
    )

    orchestrator = Orchestrator(config)
    orchestrator.add_strategies_from_dir()

    status = orchestrator.status()
    assert status["total_loops"] == 1
    assert "loops" in status
    assert "status-test" in status["loops"]
