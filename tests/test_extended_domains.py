"""Tests for extended domain adapters."""

from gridmind.domains.registry import DomainRegistry
from gridmind.core.strategy import StrategyLoader


def test_all_13_domains_registered():
    domains = DomainRegistry.list_domains()
    assert len(domains) == 13
    expected = [
        "sales_pipeline", "ad_creative", "lead_gen", "client_onboarding",
        "landing_page", "seo_aeo", "pricing", "warm_outreach",
        "ap_ar", "procurement", "job_posting", "yt_thumbnail", "call_script",
    ]
    for name in expected:
        assert name in domains, f"Missing domain: {name}"


def test_extended_templates_are_valid_strategies():
    """Every extended domain template should parse as a valid strategy."""
    extended = [
        "landing_page", "seo_aeo", "pricing", "warm_outreach",
        "ap_ar", "procurement", "job_posting", "yt_thumbnail", "call_script",
    ]
    for name in extended:
        adapter = DomainRegistry.get(name)
        assert adapter is not None, f"Adapter not found: {name}"
        template = adapter.get_strategy_template()
        strategy = StrategyLoader.parse(template)
        assert strategy.name, f"{name} template has no name"
        assert strategy.objective, f"{name} template has no objective"
        assert len(strategy.metrics) > 0, f"{name} template has no metrics"
        assert len(strategy.variables) > 0, f"{name} template has no variables"


def test_extended_evaluation_prompts():
    extended = [
        "landing_page", "seo_aeo", "pricing", "warm_outreach",
        "ap_ar", "procurement", "job_posting", "yt_thumbnail", "call_script",
    ]
    for name in extended:
        adapter = DomainRegistry.get(name)
        prompt = adapter.get_evaluation_prompt("test artifact", {"var": "val"})
        assert len(prompt) > 50, f"{name} prompt too short"
        assert "test artifact" in prompt


def test_templates_have_signal_speed():
    """Extended templates should include signal_speed field."""
    extended = [
        "landing_page", "seo_aeo", "pricing", "warm_outreach",
        "ap_ar", "procurement", "job_posting", "yt_thumbnail", "call_script",
    ]
    for name in extended:
        adapter = DomainRegistry.get(name)
        template = adapter.get_strategy_template()
        assert "signal_speed:" in template, f"{name} template missing signal_speed"
