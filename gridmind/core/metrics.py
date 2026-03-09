"""Metrics tracking. The clear metric that decides what stays."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MetricDirection(str, Enum):
    HIGHER = "higher"  # higher is better (reply rates, conversion, revenue)
    LOWER = "lower"  # lower is better (cost, churn, time-to-close)


@dataclass
class MetricSnapshot:
    """A single metric measurement at a point in time."""

    name: str
    value: float
    iteration: int
    direction: MetricDirection = MetricDirection.HIGHER

    @property
    def is_better_than(self):
        if self.direction == MetricDirection.HIGHER:
            return lambda other: self.value > other
        return lambda other: self.value < other


@dataclass
class MetricTracker:
    """Tracks the best-known value for each metric across all experiments."""

    best: dict[str, MetricSnapshot] = field(default_factory=dict)
    history: list[MetricSnapshot] = field(default_factory=list)

    def record(self, name: str, value: float, iteration: int, direction: str = "higher") -> bool:
        """Record a metric value. Returns True if it's a new best."""
        d = MetricDirection(direction)
        snapshot = MetricSnapshot(name=name, value=value, iteration=iteration, direction=d)
        self.history.append(snapshot)

        current_best = self.best.get(name)
        if current_best is None or snapshot.is_better_than(current_best.value):
            self.best[name] = snapshot
            return True
        return False

    def get_best(self, name: str) -> float | None:
        snap = self.best.get(name)
        return snap.value if snap else None

    def summary(self) -> dict[str, dict]:
        return {
            name: {
                "best_value": snap.value,
                "best_iteration": snap.iteration,
                "direction": snap.direction.value,
                "total_measurements": sum(
                    1 for h in self.history if h.name == name
                ),
            }
            for name, snap in self.best.items()
        }
