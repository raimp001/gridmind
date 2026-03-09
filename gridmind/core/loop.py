"""The autonomous research loop. This is the core pattern.

1. Human writes strategy doc (.md)
2. Agent runs experiments autonomously
3. Clear metric decides what stays
4. Repeat 100x while you sleep
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from gridmind.core.agent import AgentConfig, ResearchAgent
from gridmind.core.experiment import Experiment, ExperimentResult, ExperimentStatus
from gridmind.core.metrics import MetricTracker
from gridmind.core.strategy import Strategy, StrategyLoader


@dataclass
class LoopConfig:
    """Configuration for the research loop."""

    strategy_path: str = ""
    results_dir: str = "results"
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    dry_run: bool = False  # if True, don't call LLM - use mock data
    verbose: bool = True


@dataclass
class LoopState:
    """Mutable state of a running loop."""

    strategy: Strategy | None = None
    experiments: list[Experiment] = field(default_factory=list)
    metrics: MetricTracker = field(default_factory=MetricTracker)
    current_iteration: int = 0
    best_artifact: str = ""
    best_iteration: int = 0
    started_at: datetime | None = None
    status: str = "idle"  # idle, running, paused, completed

    def past_results(self) -> list[dict]:
        return [e.to_dict() for e in self.experiments]


class ResearchLoop:
    """The main autonomous loop. Write the .md, run the loop, sleep."""

    def __init__(self, config: LoopConfig):
        self.config = config
        self.state = LoopState()
        self.agent = ResearchAgent(config.agent_config)
        self._callbacks: list = []

    def on_event(self, callback):
        """Register a callback for loop events: (event_type, data)."""
        self._callbacks.append(callback)

    def _emit(self, event_type: str, data: dict):
        for cb in self._callbacks:
            cb(event_type, data)

    def load_strategy(self, path: str | None = None) -> Strategy:
        path = path or self.config.strategy_path
        self.state.strategy = StrategyLoader.load(path)
        self._emit("strategy_loaded", {"name": self.state.strategy.name})
        return self.state.strategy

    def run(self, strategy_path: str | None = None) -> LoopState:
        """Run the full autonomous loop."""
        if self.state.strategy and not strategy_path:
            strategy = self.state.strategy
        else:
            strategy = self.load_strategy(strategy_path)
        self.state.started_at = datetime.now(timezone.utc)
        self.state.status = "running"

        results_dir = Path(self.config.results_dir) / strategy.name
        results_dir.mkdir(parents=True, exist_ok=True)

        self._emit("loop_started", {
            "name": strategy.name,
            "max_iterations": strategy.max_iterations,
        })

        # Set baselines
        for m in strategy.metrics:
            if m.baseline is not None:
                self.state.metrics.record(
                    m.name, m.baseline, iteration=0, direction=m.direction
                )

        for i in range(1, strategy.max_iterations + 1):
            self.state.current_iteration = i

            try:
                experiment = self._run_iteration(strategy, i)
                self.state.experiments.append(experiment)

                # Save checkpoint after each iteration
                self._save_checkpoint(results_dir)

            except KeyboardInterrupt:
                self.state.status = "paused"
                self._emit("loop_paused", {"iteration": i})
                self._save_checkpoint(results_dir)
                break
            except Exception as e:
                self._emit("experiment_error", {"iteration": i, "error": str(e)})
                # Record failure but keep going
                exp = Experiment(iteration=i, variables={})
                exp.complete(ExperimentResult(error=str(e)))
                self.state.experiments.append(exp)
                continue

        if self.state.status == "running":
            self.state.status = "completed"

        self._save_results(results_dir)
        self._emit("loop_completed", {
            "total_iterations": len(self.state.experiments),
            "best_metrics": self.state.metrics.summary(),
        })

        return self.state

    def _run_iteration(self, strategy: Strategy, iteration: int) -> Experiment:
        """Run a single iteration of the loop."""
        self._emit("iteration_started", {"iteration": iteration})

        # 1. Agent designs the experiment
        if self.config.dry_run:
            design = _mock_design(strategy, iteration)
        else:
            design = self.agent.design_experiment(
                strategy=strategy,
                iteration=iteration,
                past_results=self.state.past_results(),
                best_metrics=self.state.metrics.summary(),
            )

        experiment = Experiment(
            iteration=iteration,
            variables=design.get("variables", {}),
            hypothesis=design.get("hypothesis", ""),
            approach=design.get("approach", ""),
        )
        experiment.start()

        artifact = design.get("artifact", "")

        # 2. Evaluate the experiment
        if self.config.dry_run:
            scores = _mock_evaluate(strategy, iteration)
        else:
            scores = self.agent.evaluate_experiment(strategy, experiment, artifact)

        result = ExperimentResult(
            metrics=scores,
            artifacts={"output": artifact},
        )
        experiment.complete(result)

        # 3. Clear metric decides what stays
        improved = False
        for m in strategy.metrics:
            if m.name in scores:
                is_new_best = self.state.metrics.record(
                    m.name, scores[m.name], iteration, m.direction
                )
                if is_new_best and m == strategy.primary_metric:
                    improved = True

        if improved:
            self.state.best_artifact = artifact
            self.state.best_iteration = iteration
            self._emit("new_best", {
                "iteration": iteration,
                "metrics": scores,
                "artifact_preview": artifact[:200],
            })
        else:
            experiment.reject()
            self._emit("experiment_rejected", {
                "iteration": iteration,
                "metrics": scores,
            })

        self._emit("iteration_completed", {
            "iteration": iteration,
            "status": experiment.status.value,
            "metrics": scores,
        })

        return experiment

    def _save_checkpoint(self, results_dir: Path):
        checkpoint = {
            "iteration": self.state.current_iteration,
            "status": self.state.status,
            "best_iteration": self.state.best_iteration,
            "metrics_summary": self.state.metrics.summary(),
            "experiments_count": len(self.state.experiments),
        }
        (results_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2))

    def _save_results(self, results_dir: Path):
        # Full experiment log
        experiments_data = [e.to_dict() for e in self.state.experiments]
        (results_dir / "experiments.json").write_text(
            json.dumps(experiments_data, indent=2)
        )

        # Best artifact
        if self.state.best_artifact:
            (results_dir / "best_artifact.txt").write_text(self.state.best_artifact)

        # Summary
        summary = {
            "strategy": self.state.strategy.name if self.state.strategy else "",
            "total_iterations": len(self.state.experiments),
            "completed": sum(
                1 for e in self.state.experiments
                if e.status == ExperimentStatus.COMPLETED
            ),
            "rejected": sum(
                1 for e in self.state.experiments
                if e.status == ExperimentStatus.REJECTED
            ),
            "failed": sum(
                1 for e in self.state.experiments
                if e.status == ExperimentStatus.FAILED
            ),
            "best_iteration": self.state.best_iteration,
            "best_metrics": self.state.metrics.summary(),
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
        }
        (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        # Metrics history for plotting
        history = [
            {"name": s.name, "value": s.value, "iteration": s.iteration}
            for s in self.state.metrics.history
        ]
        (results_dir / "metrics_history.json").write_text(
            json.dumps(history, indent=2)
        )


def _mock_design(strategy: Strategy, iteration: int) -> dict:
    """Mock experiment design for dry runs / testing."""
    import random

    variables = {}
    for var, options in strategy.variables.items():
        variables[var] = random.choice(options)

    return {
        "hypothesis": f"[DRY RUN] Iteration {iteration} exploring {variables}",
        "approach": "Mock experiment for testing the loop",
        "variables": variables,
        "artifact": f"Mock artifact for iteration {iteration} with vars: {variables}",
    }


def _mock_evaluate(strategy: Strategy, iteration: int) -> dict[str, float]:
    """Mock evaluation for dry runs. Simulates gradual improvement."""
    import random

    scores = {}
    for m in strategy.metrics:
        base = m.baseline or 0.5
        # Simulate noisy improvement over time
        noise = random.gauss(0, 0.05)
        trend = iteration / strategy.max_iterations * 0.2
        if m.direction == "higher":
            scores[m.name] = min(1.0, base + trend + noise)
        else:
            scores[m.name] = max(0.0, base - trend + noise)
    return scores
