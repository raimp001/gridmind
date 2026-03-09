"""Domain adapters. Each domain knows how to evaluate experiments in context."""

from gridmind.domains.registry import DomainRegistry, DomainAdapter
import gridmind.domains.extended  # noqa: F401 - registers extended domains

__all__ = ["DomainRegistry", "DomainAdapter"]
