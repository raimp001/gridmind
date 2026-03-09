"""Experiment execution. Each experiment is a single iteration of the loop."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"  # metrics didn't improve


@dataclass
class ExperimentResult:
    """The outcome of a single experiment run."""

    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)  # name -> content
    logs: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and len(self.metrics) > 0


@dataclass
class Experiment:
    """A single experiment in the autonomous loop."""

    iteration: int
    variables: dict[str, str]  # the specific variable combo for this run
    hypothesis: str = ""
    approach: str = ""  # what the agent plans to do
    status: ExperimentStatus = ExperimentStatus.PENDING
    result: ExperimentResult | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    parent_iteration: int | None = None  # which experiment this builds on

    def start(self):
        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def complete(self, result: ExperimentResult):
        self.result = result
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        self.status = (
            ExperimentStatus.COMPLETED if result.succeeded else ExperimentStatus.FAILED
        )

    def reject(self):
        """Mark as rejected - metrics didn't beat the current best."""
        self.status = ExperimentStatus.REJECTED

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "variables": self.variables,
            "hypothesis": self.hypothesis,
            "approach": self.approach,
            "status": self.status.value,
            "metrics": self.result.metrics if self.result else {},
            "duration_seconds": self.duration_seconds,
            "parent_iteration": self.parent_iteration,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
