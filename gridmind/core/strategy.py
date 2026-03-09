"""Strategy document parser. The strategy .md file is the human-agent interface."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Metric:
    """A single metric to track."""

    name: str
    direction: str  # "higher" or "lower"
    baseline: float | None = None

    @property
    def is_better(self):
        """Return a comparison function based on direction."""
        if self.direction == "higher":
            return lambda new, old: new > old
        return lambda new, old: new < old


@dataclass
class Strategy:
    """Parsed strategy document. This is the .md file that tells AI how to think."""

    name: str
    domain: str
    objective: str
    constraints: list[str] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    experiment_template: str = ""
    variables: dict[str, list[str]] = field(default_factory=dict)
    max_iterations: int = 100
    time_budget_seconds: int = 300  # 5 min per experiment, like autoresearch
    raw_content: str = ""
    source_path: str = ""

    @property
    def primary_metric(self) -> Metric | None:
        return self.metrics[0] if self.metrics else None


class StrategyLoader:
    """Parse a strategy .md file into a Strategy object.

    Strategy docs use YAML frontmatter for structured config and markdown
    body for the experiment template / instructions.

    Example:

        ---
        name: cold-email-optimizer
        domain: lead_gen
        objective: Maximize reply rate for cold outreach emails
        metrics:
          - name: reply_rate
            direction: higher
            baseline: 0.02
        variables:
          subject_style: [question, stat, pain_point, curiosity]
          length: [short, medium, long]
          cta: [soft_ask, direct_ask, value_offer]
        max_iterations: 100
        time_budget_seconds: 120
        ---

        # Experiment Instructions

        Generate a cold email variant using the assigned variables...
    """

    FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)

    @classmethod
    def load(cls, path: str | Path) -> Strategy:
        path = Path(path)
        content = path.read_text()
        return cls.parse(content, source_path=str(path))

    @classmethod
    def parse(cls, content: str, source_path: str = "") -> Strategy:
        match = cls.FRONTMATTER_RE.match(content)
        if not match:
            raise ValueError(
                "Strategy doc must have YAML frontmatter between --- markers"
            )

        frontmatter = yaml.safe_load(match.group(1))
        body = match.group(2).strip()

        metrics = []
        for m in frontmatter.get("metrics", []):
            metrics.append(
                Metric(
                    name=m["name"],
                    direction=m.get("direction", "higher"),
                    baseline=m.get("baseline"),
                )
            )

        return Strategy(
            name=frontmatter["name"],
            domain=frontmatter.get("domain", "general"),
            objective=frontmatter["objective"],
            constraints=frontmatter.get("constraints", []),
            metrics=metrics,
            experiment_template=body,
            variables=frontmatter.get("variables", {}),
            max_iterations=frontmatter.get("max_iterations", 100),
            time_budget_seconds=frontmatter.get("time_budget_seconds", 300),
            raw_content=content,
            source_path=source_path,
        )
