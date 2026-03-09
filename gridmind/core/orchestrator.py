"""Multi-loop orchestrator. Run 17+ experiment loops simultaneously across
every growth surface. Different scoring speeds, same engine.

Signal speed tiers (from Eric Siu / Karpathy pattern):

    FAST (24-72h):   cold email, social DMs, YT thumbnails, warm outreach
    MEDIUM (7-14d):  ad creative, landing pages, blog intros, trial funnels, job postings, pricing
    SLOW (30-90d):   call scripts, proposals, onboarding, churn recovery, SEO/AEO, AP/AR, procurement

The orchestrator runs all loops in parallel, routes cross-loop learnings,
and tracks the compounding revenue impact.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from gridmind.core.agent import AgentConfig
from gridmind.core.loop import LoopConfig, LoopState, ResearchLoop
from gridmind.core.strategy import Strategy, StrategyLoader


class SignalSpeed(str, Enum):
    FAST = "fast"      # 24-72h scoring window
    MEDIUM = "medium"  # 7-14d scoring window
    SLOW = "slow"      # 30-90d scoring window


# Default scoring windows in hours
SIGNAL_WINDOWS = {
    SignalSpeed.FAST: 72,
    SignalSpeed.MEDIUM: 168,  # 7 days
    SignalSpeed.SLOW: 720,   # 30 days
}

# Map domains to their natural signal speed
DOMAIN_SIGNAL_SPEED = {
    # Fast signal (24-72h)
    "sales_pipeline": SignalSpeed.FAST,
    "warm_outreach": SignalSpeed.FAST,
    "yt_thumbnail": SignalSpeed.FAST,
    # Medium signal (7-14d)
    "ad_creative": SignalSpeed.MEDIUM,
    "lead_gen": SignalSpeed.MEDIUM,
    "landing_page": SignalSpeed.MEDIUM,
    "job_posting": SignalSpeed.MEDIUM,
    "pricing": SignalSpeed.MEDIUM,
    # Slow signal (30-90d)
    "client_onboarding": SignalSpeed.SLOW,
    "call_script": SignalSpeed.SLOW,
    "seo_aeo": SignalSpeed.SLOW,
    "ap_ar": SignalSpeed.SLOW,
    "procurement": SignalSpeed.SLOW,
}


@dataclass
class LoopHandle:
    """Reference to a running loop within the orchestrator."""

    strategy: Strategy
    loop: ResearchLoop
    signal_speed: SignalSpeed
    events: list[dict] = field(default_factory=list)
    state: LoopState | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class CrossLoopInsight:
    """A learning from one loop that may benefit another."""

    source_domain: str
    source_iteration: int
    insight_type: str  # "hook_works", "angle_resonates", "number_converts", etc.
    insight: str
    metric_impact: dict[str, float]
    target_domains: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CampaignConfig:
    """Configuration for a multi-loop campaign."""

    name: str = "default-campaign"
    strategies_dir: str = "strategies"
    results_dir: str = "results"
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    max_workers: int = 4  # concurrent loops
    dry_run: bool = False
    verbose: bool = True
    enable_cross_learning: bool = True


class Orchestrator:
    """Runs multiple research loops in parallel across growth surfaces.

    The marketing machine: 13 domains, 3 signal speeds, 1 engine.
    """

    def __init__(self, config: CampaignConfig):
        self.config = config
        self.loops: dict[str, LoopHandle] = {}
        self.insights: list[CrossLoopInsight] = []
        self._callbacks: list = []

    def on_event(self, callback):
        self._callbacks.append(callback)

    def _emit(self, event_type: str, data: dict):
        for cb in self._callbacks:
            cb(event_type, data)

    def add_strategy(self, path: str) -> str:
        """Add a strategy to the campaign. Returns the loop key."""
        strategy = StrategyLoader.load(path)
        signal_speed = DOMAIN_SIGNAL_SPEED.get(
            strategy.domain, SignalSpeed.MEDIUM
        )

        loop_config = LoopConfig(
            strategy_path=path,
            results_dir=f"{self.config.results_dir}/{self.config.name}",
            agent_config=self.config.agent_config,
            dry_run=self.config.dry_run,
            verbose=self.config.verbose,
        )
        loop = ResearchLoop(loop_config)
        loop.state.strategy = strategy

        key = strategy.name
        handle = LoopHandle(
            strategy=strategy,
            loop=loop,
            signal_speed=signal_speed,
        )

        # Wire up events
        def make_handler(k):
            def handler(etype, data):
                handle.events.append({"type": etype, **data})
                self._emit("loop_event", {"loop": k, "type": etype, **data})
            return handler

        loop.on_event(make_handler(key))
        self.loops[key] = handle

        self._emit("strategy_added", {
            "name": key,
            "domain": strategy.domain,
            "signal_speed": signal_speed.value,
            "max_iterations": strategy.max_iterations,
        })

        return key

    def add_strategies_from_dir(self, directory: str | None = None) -> list[str]:
        """Load all .md strategy files from a directory."""
        d = Path(directory or self.config.strategies_dir)
        keys = []
        for f in sorted(d.glob("*.md")):
            try:
                key = self.add_strategy(str(f))
                keys.append(key)
            except Exception as e:
                self._emit("strategy_error", {"path": str(f), "error": str(e)})
        return keys

    def run(self) -> dict[str, LoopState]:
        """Run all loops in parallel. Returns {name: final_state}."""
        if not self.loops:
            raise ValueError("No strategies added. Use add_strategy() first.")

        self._emit("campaign_started", {
            "name": self.config.name,
            "total_loops": len(self.loops),
            "loops_by_speed": self._count_by_speed(),
        })

        results: dict[str, LoopState] = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            for key, handle in self.loops.items():
                handle.started_at = datetime.now(timezone.utc)
                future = executor.submit(self._run_single_loop, key, handle)
                futures[future] = key

            for future in as_completed(futures):
                key = futures[future]
                handle = self.loops[key]
                try:
                    state = future.result()
                    handle.state = state
                    handle.completed_at = datetime.now(timezone.utc)
                    results[key] = state

                    # Extract cross-loop insights
                    if self.config.enable_cross_learning:
                        self._extract_insights(key, handle)

                    self._emit("loop_completed", {
                        "name": key,
                        "domain": handle.strategy.domain,
                        "iterations": len(state.experiments),
                        "best_iteration": state.best_iteration,
                        "best_metrics": state.metrics.summary(),
                    })

                except Exception as e:
                    handle.error = str(e)
                    handle.completed_at = datetime.now(timezone.utc)
                    self._emit("loop_failed", {"name": key, "error": str(e)})

        # Save campaign-level results
        self._save_campaign_results()

        self._emit("campaign_completed", {
            "name": self.config.name,
            "total_loops": len(self.loops),
            "completed": sum(1 for h in self.loops.values() if h.state),
            "failed": sum(1 for h in self.loops.values() if h.error),
            "total_experiments": sum(
                len(h.state.experiments) for h in self.loops.values() if h.state
            ),
            "cross_loop_insights": len(self.insights),
        })

        return results

    def _run_single_loop(self, key: str, handle: LoopHandle) -> LoopState:
        """Run a single loop (called from thread pool)."""
        return handle.loop.run()

    def _extract_insights(self, source_key: str, handle: LoopHandle):
        """Extract learnings from a completed loop that could help other loops."""
        if not handle.state or not handle.state.best_artifact:
            return

        best_metrics = handle.state.metrics.summary()
        source_domain = handle.strategy.domain

        # Find which metrics improved the most from baseline
        for m in handle.strategy.metrics:
            if m.baseline is None:
                continue
            best_info = best_metrics.get(m.name, {})
            best_val = best_info.get("best_value", m.baseline)

            if m.direction == "higher":
                improvement = (best_val - m.baseline) / max(m.baseline, 0.001)
            else:
                improvement = (m.baseline - best_val) / max(m.baseline, 0.001)

            # If >20% improvement, it's an insight worth sharing
            if improvement > 0.2:
                # Determine which other domains could benefit
                target_domains = self._find_related_domains(source_domain)

                insight = CrossLoopInsight(
                    source_domain=source_domain,
                    source_iteration=best_info.get("best_iteration", 0),
                    insight_type=f"{m.name}_improvement",
                    insight=(
                        f"{source_domain} improved {m.name} by {improvement:.0%} "
                        f"(from {m.baseline:.4f} to {best_val:.4f}). "
                        f"Best artifact: {handle.state.best_artifact[:200]}"
                    ),
                    metric_impact={m.name: improvement},
                    target_domains=target_domains,
                )
                self.insights.append(insight)

    def _find_related_domains(self, source: str) -> list[str]:
        """Find domains that could benefit from insights in the source domain."""
        # Mapping of which domains share learnings
        relations = {
            "sales_pipeline": ["warm_outreach", "call_script", "ad_creative"],
            "warm_outreach": ["sales_pipeline", "client_onboarding"],
            "ad_creative": ["landing_page", "yt_thumbnail", "lead_gen"],
            "landing_page": ["ad_creative", "lead_gen", "pricing"],
            "lead_gen": ["landing_page", "ad_creative", "sales_pipeline"],
            "seo_aeo": ["landing_page", "lead_gen", "ad_creative"],
            "pricing": ["landing_page", "lead_gen"],
            "client_onboarding": ["warm_outreach", "call_script"],
            "call_script": ["sales_pipeline", "warm_outreach", "client_onboarding"],
            "yt_thumbnail": ["ad_creative", "landing_page"],
            "job_posting": ["ad_creative", "landing_page"],
            "ap_ar": ["procurement"],
            "procurement": ["ap_ar"],
        }
        return relations.get(source, [])

    def _count_by_speed(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for handle in self.loops.values():
            speed = handle.signal_speed.value
            counts[speed] = counts.get(speed, 0) + 1
        return counts

    def _save_campaign_results(self):
        """Save campaign-level summary and cross-loop insights."""
        results_dir = Path(self.config.results_dir) / self.config.name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Campaign summary
        summary = {
            "name": self.config.name,
            "total_loops": len(self.loops),
            "loops": {},
            "cross_loop_insights": len(self.insights),
            "total_experiments": 0,
        }

        for key, handle in self.loops.items():
            loop_info = {
                "domain": handle.strategy.domain,
                "signal_speed": handle.signal_speed.value,
                "status": "completed" if handle.state else "failed",
            }
            if handle.state:
                loop_info["experiments"] = len(handle.state.experiments)
                loop_info["best_iteration"] = handle.state.best_iteration
                loop_info["best_metrics"] = handle.state.metrics.summary()
                summary["total_experiments"] += len(handle.state.experiments)
            if handle.error:
                loop_info["error"] = handle.error

            summary["loops"][key] = loop_info

        (results_dir / "campaign_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        # Cross-loop insights
        if self.insights:
            insights_data = [
                {
                    "source_domain": i.source_domain,
                    "insight_type": i.insight_type,
                    "insight": i.insight,
                    "metric_impact": i.metric_impact,
                    "target_domains": i.target_domains,
                    "timestamp": i.timestamp.isoformat(),
                }
                for i in self.insights
            ]
            (results_dir / "cross_loop_insights.json").write_text(
                json.dumps(insights_data, indent=2)
            )

    def status(self) -> dict:
        """Get current status of all loops."""
        return {
            "campaign": self.config.name,
            "total_loops": len(self.loops),
            "loops_by_speed": self._count_by_speed(),
            "loops": {
                key: {
                    "domain": h.strategy.domain,
                    "signal_speed": h.signal_speed.value,
                    "status": (
                        "completed" if h.completed_at
                        else "running" if h.started_at
                        else "pending"
                    ),
                    "iteration": (
                        h.loop.state.current_iteration if h.loop else 0
                    ),
                    "max_iterations": h.strategy.max_iterations,
                    "best_iteration": (
                        h.loop.state.best_iteration if h.loop else 0
                    ),
                }
                for key, h in self.loops.items()
            },
            "cross_loop_insights": len(self.insights),
            "total_experiments": sum(
                len(h.loop.state.experiments)
                for h in self.loops.values()
                if h.loop
            ),
        }


def estimate_experiments(strategies: list[str] | None = None, days: int = 365) -> dict:
    """Estimate total experiments per year across all loops.

    The math from Eric Siu:
    - 100 experiments per loop per night
    - 365 nights per year
    - 17 loops = 36,500+ experiments/year
    - Your competitor runs 30.
    """
    if strategies:
        loop_count = len(strategies)
        per_loop = sum(
            StrategyLoader.load(s).max_iterations for s in strategies
        ) / loop_count
    else:
        loop_count = len(DOMAIN_SIGNAL_SPEED)
        per_loop = 100

    daily = loop_count * per_loop
    yearly = daily * days

    return {
        "loops": loop_count,
        "experiments_per_loop_per_run": per_loop,
        "daily_experiments": daily,
        "yearly_experiments": yearly,
        "vs_traditional": f"{yearly / 30:.0f}x more than a typical team (30/year)",
        "signal_tiers": {
            "fast_loops": sum(
                1 for s in DOMAIN_SIGNAL_SPEED.values() if s == SignalSpeed.FAST
            ),
            "medium_loops": sum(
                1 for s in DOMAIN_SIGNAL_SPEED.values() if s == SignalSpeed.MEDIUM
            ),
            "slow_loops": sum(
                1 for s in DOMAIN_SIGNAL_SPEED.values() if s == SignalSpeed.SLOW
            ),
        },
    }
