"""Tests for domain adapters."""

from gridmind.domains.registry import DomainRegistry
from gridmind.core.strategy import StrategyLoader


def test_built_in_domains_registered():
    domains = DomainRegistry.list_domains()
    assert "sales_pipeline" in domains
    assert "ad_creative" in domains
    assert "lead_gen" in domains
    assert "client_onboarding" in domains


def test_domain_templates_are_valid_strategies():
    """Every domain template should parse as a valid strategy."""
    for name, adapter in DomainRegistry.all().items():
        template = adapter.get_strategy_template()
        strategy = StrategyLoader.parse(template)
        assert strategy.name, f"{name} template has no name"
        assert strategy.objective, f"{name} template has no objective"
        assert len(strategy.metrics) > 0, f"{name} template has no metrics"
        assert len(strategy.variables) > 0, f"{name} template has no variables"


def test_domain_evaluation_prompts():
    for name, adapter in DomainRegistry.all().items():
        prompt = adapter.get_evaluation_prompt("test artifact", {"var": "value"})
        assert len(prompt) > 50, f"{name} evaluation prompt too short"
        assert "test artifact" in prompt
