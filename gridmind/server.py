"""FastAPI server for running research loops and campaigns via API.

Includes webhook endpoint for ingesting real-world metrics from production.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gridmind.core.agent import AgentConfig
from gridmind.core.loop import LoopConfig, ResearchLoop
from gridmind.core.orchestrator import (
    CampaignConfig,
    Orchestrator,
    DOMAIN_SIGNAL_SPEED,
    estimate_experiments,
)
from gridmind.core.strategy import StrategyLoader
from gridmind.domains.registry import DomainRegistry

app = FastAPI(
    title="GridMind",
    description="Autonomous research loops for any domain. Write strategy docs, run experiments while you sleep.",
    version="0.3.0",
)

# In-memory store
_loops: dict[str, dict] = {}
_campaigns: dict[str, dict] = {}
_webhook_events: list[dict] = []  # Real-world metrics from production


class RunRequest(BaseModel):
    strategy_content: str | None = None
    strategy_path: str | None = None
    provider: str = "anthropic"
    model: str | None = None
    dry_run: bool = False
    max_iterations: int | None = None
    results_dir: str = "results"


class CampaignRequest(BaseModel):
    name: str = "campaign"
    strategies_dir: str = "strategies"
    provider: str = "anthropic"
    model: str | None = None
    dry_run: bool = False
    max_workers: int = 4
    results_dir: str = "results"


class WebhookEvent(BaseModel):
    """Real-world metric data from production systems (SendGrid, GA, etc.)."""
    source: str  # e.g. "sendgrid", "google_analytics", "mixpanel"
    loop_id: str | None = None  # link to a specific loop
    experiment_iteration: int | None = None  # link to specific experiment
    metrics: dict[str, float]  # e.g. {"reply_rate": 0.05, "open_rate": 0.42}
    metadata: dict | None = None  # arbitrary context
    timestamp: str | None = None


# --- Root ---

@app.get("/")
async def root():
    return {
        "name": "GridMind",
        "version": "0.3.0",
        "description": "Autonomous research loops for any domain",
        "domains": len(DomainRegistry.list_domains()),
        "pattern": [
            "1. Human writes strategy doc (.md)",
            "2. Agent runs experiments autonomously",
            "3. Clear metric decides what stays",
            "4. Repeat 100x while you sleep",
        ],
    }


# --- Domains ---

@app.get("/domains")
async def list_domains():
    """List all 13 domain adapters with signal speed tiers."""
    result = {}
    for name, adapter in DomainRegistry.all().items():
        speed = DOMAIN_SIGNAL_SPEED.get(name)
        result[name] = {
            "description": adapter.description,
            "signal_speed": speed.value if speed else "medium",
        }
    return result


@app.post("/domains/{domain}/template")
async def get_domain_template(domain: str):
    """Get a strategy template for a specific domain."""
    adapter = DomainRegistry.get(domain)
    if not adapter:
        raise HTTPException(404, f"Unknown domain: {domain}")
    return {"domain": domain, "template": adapter.get_strategy_template()}


# --- Single Loops ---

@app.post("/loops")
async def start_loop(req: RunRequest):
    """Start a single autonomous research loop."""
    loop_id = str(uuid.uuid4())[:8]

    if not req.strategy_content and not req.strategy_path:
        raise HTTPException(400, "Provide strategy_content or strategy_path")

    if req.strategy_content:
        strategy = StrategyLoader.parse(req.strategy_content)
    else:
        strategy = StrategyLoader.load(req.strategy_path)

    if req.max_iterations:
        strategy.max_iterations = req.max_iterations

    default_models = {"anthropic": "claude-sonnet-4-20250514", "openai": "gpt-4o"}
    agent_config = AgentConfig(
        provider=req.provider,
        model=req.model or default_models.get(req.provider, "claude-sonnet-4-20250514"),
    )

    config = LoopConfig(
        results_dir=req.results_dir,
        agent_config=agent_config,
        dry_run=req.dry_run,
    )

    loop = ResearchLoop(config)
    loop.state.strategy = strategy

    events: list[dict] = []
    loop.on_event(lambda etype, data: events.append({"type": etype, **data}))

    _loops[loop_id] = {
        "id": loop_id,
        "strategy": strategy.name,
        "loop": loop,
        "events": events,
    }

    asyncio.get_event_loop().run_in_executor(None, loop.run)

    return {
        "loop_id": loop_id,
        "strategy": strategy.name,
        "status": "running",
        "max_iterations": strategy.max_iterations,
    }


@app.get("/loops")
async def list_loops():
    """List all research loops."""
    return [
        {
            "id": info["id"],
            "strategy": info["strategy"],
            "status": info["loop"].state.status,
            "iteration": info["loop"].state.current_iteration,
        }
        for info in _loops.values()
    ]


@app.get("/loops/{loop_id}")
async def get_loop(loop_id: str):
    """Get status and results of a research loop."""
    info = _loops.get(loop_id)
    if not info:
        raise HTTPException(404, f"Loop not found: {loop_id}")

    state = info["loop"].state
    return {
        "id": loop_id,
        "strategy": info["strategy"],
        "status": state.status,
        "current_iteration": state.current_iteration,
        "total_experiments": len(state.experiments),
        "best_iteration": state.best_iteration,
        "best_metrics": state.metrics.summary(),
        "recent_events": info["events"][-20:],
    }


@app.get("/loops/{loop_id}/best")
async def get_best_artifact(loop_id: str):
    """Get the best artifact from a loop."""
    info = _loops.get(loop_id)
    if not info:
        raise HTTPException(404, f"Loop not found: {loop_id}")

    state = info["loop"].state
    return {
        "best_iteration": state.best_iteration,
        "best_artifact": state.best_artifact,
        "best_metrics": state.metrics.summary(),
    }


# --- Campaigns (multi-loop) ---

@app.post("/campaigns")
async def start_campaign(req: CampaignRequest):
    """Start a multi-loop campaign across growth surfaces."""
    campaign_id = str(uuid.uuid4())[:8]

    default_models = {"anthropic": "claude-sonnet-4-20250514", "openai": "gpt-4o"}
    config = CampaignConfig(
        name=req.name,
        strategies_dir=req.strategies_dir,
        results_dir=req.results_dir,
        agent_config=AgentConfig(
            provider=req.provider,
            model=req.model or default_models.get(req.provider, "claude-sonnet-4-20250514"),
        ),
        max_workers=req.max_workers,
        dry_run=req.dry_run,
    )

    orchestrator = Orchestrator(config)
    events: list[dict] = []
    orchestrator.on_event(lambda etype, data: events.append({"type": etype, **data}))

    keys = orchestrator.add_strategies_from_dir()
    if not keys:
        raise HTTPException(400, f"No strategy files found in {req.strategies_dir}")

    _campaigns[campaign_id] = {
        "id": campaign_id,
        "name": req.name,
        "orchestrator": orchestrator,
        "events": events,
        "strategies": keys,
    }

    asyncio.get_event_loop().run_in_executor(None, orchestrator.run)

    return {
        "campaign_id": campaign_id,
        "name": req.name,
        "status": "running",
        "loops": len(keys),
        "strategies": keys,
    }


@app.get("/campaigns")
async def list_campaigns():
    """List all campaigns."""
    return [
        {
            "id": info["id"],
            "name": info["name"],
            "loops": len(info["strategies"]),
            "status": info["orchestrator"].status(),
        }
        for info in _campaigns.values()
    ]


@app.get("/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str):
    """Get campaign status with all loop details."""
    info = _campaigns.get(campaign_id)
    if not info:
        raise HTTPException(404, f"Campaign not found: {campaign_id}")

    return {
        "id": campaign_id,
        **info["orchestrator"].status(),
        "insights": [
            {
                "source": i.source_domain,
                "type": i.insight_type,
                "insight": i.insight[:200],
                "targets": i.target_domains,
            }
            for i in info["orchestrator"].insights
        ],
        "recent_events": info["events"][-30:],
    }


# --- Utils ---

@app.get("/math")
async def get_math():
    """The experiment math: you vs your competitor."""
    return estimate_experiments()


@app.post("/validate")
async def validate_strategy(req: RunRequest):
    """Validate a strategy document without running it."""
    if not req.strategy_content:
        raise HTTPException(400, "Provide strategy_content")

    try:
        strategy = StrategyLoader.parse(req.strategy_content)
        return {
            "valid": True,
            "name": strategy.name,
            "domain": strategy.domain,
            "objective": strategy.objective,
            "metrics": [{"name": m.name, "direction": m.direction} for m in strategy.metrics],
            "variables": strategy.variables,
            "max_iterations": strategy.max_iterations,
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


# --- Webhooks: Real-world metric ingestion ---

@app.post("/webhooks/metrics")
async def receive_metrics(event: WebhookEvent):
    """Receive real-world metrics from production systems.

    POST metrics from SendGrid, Google Analytics, Mixpanel, etc.
    These feed back into the loop to ground-truth LLM evaluations.

    Example payload:
        {
            "source": "sendgrid",
            "loop_id": "abc12345",
            "experiment_iteration": 42,
            "metrics": {"reply_rate": 0.05, "open_rate": 0.42},
            "metadata": {"campaign_id": "q1-outbound"}
        }
    """
    record = {
        "source": event.source,
        "loop_id": event.loop_id,
        "experiment_iteration": event.experiment_iteration,
        "metrics": event.metrics,
        "metadata": event.metadata,
        "timestamp": event.timestamp or datetime.now(timezone.utc).isoformat(),
        "received_at": datetime.now(timezone.utc).isoformat(),
    }
    _webhook_events.append(record)

    # If linked to a running loop, update its metrics
    if event.loop_id and event.loop_id in _loops:
        loop_info = _loops[event.loop_id]
        loop_state = loop_info["loop"].state
        for metric_name, value in event.metrics.items():
            loop_state.metrics.record(
                metric_name, value,
                iteration=event.experiment_iteration or loop_state.current_iteration,
                direction="higher",
            )
        loop_info["events"].append({"type": "webhook_metrics", **record})

    # Persist to disk
    _persist_webhook_events()

    return {
        "status": "received",
        "total_events": len(_webhook_events),
        "linked_to_loop": event.loop_id if event.loop_id in _loops else None,
    }


@app.get("/webhooks/metrics")
async def list_webhook_events(
    source: str | None = None,
    loop_id: str | None = None,
    limit: int = 100,
):
    """List received webhook metric events, optionally filtered."""
    events = _webhook_events
    if source:
        events = [e for e in events if e["source"] == source]
    if loop_id:
        events = [e for e in events if e.get("loop_id") == loop_id]
    return {"total": len(events), "events": events[-limit:]}


def _persist_webhook_events():
    """Save webhook events to disk for durability."""
    try:
        path = Path("results") / "webhook_events.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_webhook_events, indent=2))
    except OSError:
        pass
