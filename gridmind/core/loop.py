"""The autonomous research loop. This is the core pattern.

1. Human writes strategy doc (.md)
2. Agent runs experiments autonomously
3. Clear metric decides what stays
4. Repeat 100x while you sleep

Production features:
- Checkpoint save/load for crash recovery
- Convergence detection (early stopping)
- Graceful error handling with fallback to mock
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from gridmind.core.agent import AgentConfig, ResearchAgent
from gridmind.core.experiment import Experiment, ExperimentResult, ExperimentStatus
from gridmind.core.metrics import MetricTracker, MetricSnapshot
from gridmind.core.strategy import Strategy, StrategyLoader

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    """Configuration for the research loop."""

    strategy_path: str = ""
    results_dir: str = "results"
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    dry_run: bool = False
    verbose: bool = True
    # Convergence: stop if no improvement in N iterations
    convergence_window: int = 0  # 0 = disabled
    # Max consecutive failures before pausing
    max_consecutive_failures: int = 10
    # Fallback to mock on LLM failure (keeps loop alive)
    fallback_on_error: bool = True


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
    status: str = "idle"  # idle, running, paused, completed, converged
    consecutive_failures: int = 0
    last_improvement_iteration: int = 0

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
            try:
                cb(event_type, data)
            except Exception:
                pass  # Never let callback errors kill the loop

    def load_strategy(self, path: str | None = None) -> Strategy:
        path = path or self.config.strategy_path
        self.state.strategy = StrategyLoader.load(path)
        self._emit("strategy_loaded", {"name": self.state.strategy.name})
        return self.state.strategy

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> bool:
        """Resume loop from a checkpoint file. Returns True if resumed.

        Loads iteration count, metrics history, best artifact, and experiment
        log so the loop can pick up where it left off.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return False

        try:
            checkpoint = json.loads(checkpoint_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load checkpoint: %s", e)
            return False

        self.state.current_iteration = checkpoint.get("iteration", 0)
        self.state.best_iteration = checkpoint.get("best_iteration", 0)
        self.state.best_artifact = checkpoint.get("best_artifact", "")
        self.state.last_improvement_iteration = checkpoint.get(
            "last_improvement_iteration", 0
        )

        # Restore metrics history
        for entry in checkpoint.get("metrics_history", []):
            self.state.metrics.record(
                entry["name"], entry["value"], entry["iteration"],
                entry.get("direction", "higher"),
            )

        # Restore experiment log
        for exp_data in checkpoint.get("experiments", []):
            exp = Experiment(
                iteration=exp_data["iteration"],
                variables=exp_data.get("variables", {}),
                hypothesis=exp_data.get("hypothesis", ""),
                approach=exp_data.get("approach", ""),
            )
            exp.status = ExperimentStatus(exp_data.get("status", "completed"))
            if exp_data.get("metrics"):
                exp.result = ExperimentResult(metrics=exp_data["metrics"])
            self.state.experiments.append(exp)

        self.state.status = "running"
        logger.info(
            "Resumed from checkpoint at iteration %d", self.state.current_iteration
        )
        self._emit("checkpoint_resumed", {
            "iteration": self.state.current_iteration,
            "experiments_restored": len(self.state.experiments),
        })
        return True

    def run(self, strategy_path: str | None = None) -> LoopState:
        """Run the full autonomous loop."""
        if self.state.strategy and not strategy_path:
            strategy = self.state.strategy
        else:
            strategy = self.load_strategy(strategy_path)

        if not self.state.started_at:
            self.state.started_at = datetime.now(timezone.utc)
        self.state.status = "running"

        results_dir = Path(self.config.results_dir) / strategy.name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Try to resume from checkpoint
        checkpoint_path = results_dir / "checkpoint.json"
        resumed = False
        if self.state.current_iteration == 0 and checkpoint_path.exists():
            resumed = self.resume_from_checkpoint(checkpoint_path)

        start_iteration = self.state.current_iteration + 1

        self._emit("loop_started", {
            "name": strategy.name,
            "max_iterations": strategy.max_iterations,
            "resumed_from": start_iteration - 1 if resumed else 0,
        })

        # Set baselines (only if not resumed)
        if not resumed:
            for m in strategy.metrics:
                if m.baseline is not None:
                    self.state.metrics.record(
                        m.name, m.baseline, iteration=0, direction=m.direction
                    )

        for i in range(start_iteration, strategy.max_iterations + 1):
            self.state.current_iteration = i

            try:
                experiment = self._run_iteration(strategy, i)
                self.state.experiments.append(experiment)
                self.state.consecutive_failures = 0

                # Save checkpoint after each iteration
                self._save_checkpoint(results_dir)

            except KeyboardInterrupt:
                self.state.status = "paused"
                self._emit("loop_paused", {"iteration": i})
                self._save_checkpoint(results_dir)
                break

            except Exception as e:
                self.state.consecutive_failures += 1
                logger.error("Iteration %d failed: %s", i, e)
                self._emit("experiment_error", {"iteration": i, "error": str(e)})

                # Fallback: use mock data to keep loop alive
                if self.config.fallback_on_error and not self.config.dry_run:
                    logger.info("Falling back to mock for iteration %d", i)
                    try:
                        design = _mock_design(strategy, i)
                        scores = _mock_evaluate(strategy, i)
                        exp = Experiment(iteration=i, variables=design.get("variables", {}))
                        exp.start()
                        exp.complete(ExperimentResult(
                            metrics=scores,
                            artifacts={"output": design.get("artifact", "")},
                            logs=[f"Fallback mock: {e}"],
                        ))
                        self.state.experiments.append(exp)
                        self._save_checkpoint(results_dir)
                        continue
                    except Exception:
                        pass

                # Record failure
                exp = Experiment(iteration=i, variables={})
                exp.complete(ExperimentResult(error=str(e)))
                self.state.experiments.append(exp)
                self._save_checkpoint(results_dir)

                # Too many consecutive failures = pause
                if self.state.consecutive_failures >= self.config.max_consecutive_failures:
                    self.state.status = "paused"
                    self._emit("loop_paused_failures", {
                        "iteration": i,
                        "consecutive_failures": self.state.consecutive_failures,
                    })
                    break

                continue

            # Check convergence
            if self._check_convergence(strategy, i):
                self.state.status = "converged"
                self._emit("loop_converged", {
                    "iteration": i,
                    "last_improvement": self.state.last_improvement_iteration,
                    "window": self.config.convergence_window,
                })
                break

        if self.state.status == "running":
            self.state.status = "completed"

        self._save_results(results_dir)
        self._save_checkpoint(results_dir)  # Final checkpoint
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
            self.state.last_improvement_iteration = iteration
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

    def _check_convergence(self, strategy: Strategy, iteration: int) -> bool:
        """Check if the loop has converged (no improvement for N iterations)."""
        window = self.config.convergence_window
        if window <= 0:
            return False
        since_last = iteration - self.state.last_improvement_iteration
        return since_last >= window

    def _save_checkpoint(self, results_dir: Path):
        """Save full checkpoint for crash recovery."""
        checkpoint = {
            "iteration": self.state.current_iteration,
            "status": self.state.status,
            "best_iteration": self.state.best_iteration,
            "best_artifact": self.state.best_artifact,
            "last_improvement_iteration": self.state.last_improvement_iteration,
            "metrics_summary": self.state.metrics.summary(),
            "metrics_history": [
                {
                    "name": s.name,
                    "value": s.value,
                    "iteration": s.iteration,
                    "direction": s.direction.value,
                }
                for s in self.state.metrics.history
            ],
            "experiments": [e.to_dict() for e in self.state.experiments],
            "experiments_count": len(self.state.experiments),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            # Write to temp file first, then rename (atomic on most filesystems)
            tmp_path = results_dir / "checkpoint.json.tmp"
            tmp_path.write_text(json.dumps(checkpoint, indent=2))
            tmp_path.rename(results_dir / "checkpoint.json")
        except OSError as e:
            logger.error("Failed to save checkpoint: %s", e)

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
            "converged": self.state.status == "converged",
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
