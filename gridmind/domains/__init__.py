"""Domain adapters. Each domain knows how to evaluate experiments in context."""

from gridmind.domains.registry import DomainRegistry, DomainAdapter

__all__ = ["DomainRegistry", "DomainAdapter"]
