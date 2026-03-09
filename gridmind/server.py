"""FastAPI server for running research loops via API."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gridmind.core.agent import AgentConfig
from gridmind.core.loop import LoopConfig, LoopState, ResearchLoop
from gridmind.core.strategy import StrategyLoader
from gridmind.domains.registry import DomainRegistry

app = FastAPI(
    title="GridMind",
    description="Autonomous research loops for any domain. Write strategy docs, run experiments while you sleep.",
    version="0.1.0",
)

# In-memory store of running/completed loops
_loops: dict[str, dict] = {}


class RunRequest(BaseModel):
    strategy_content: str | None = None
    strategy_path: str | None = None
    provider: str = "anthropic"
    model: str | None = None
    dry_run: bool = False
    max_iterations: int | None = None
    results_dir: str = "results"


class InitRequest(BaseModel):
    domain: str


# --- Endpoints ---


@app.get("/")
async def root():
    return {
        "name": "GridMind",
        "version": "0.1.0",
        "description": "Autonomous research loops for any domain",
        "pattern": [
            "1. Human writes strategy doc (.md)",
            "2. Agent runs experiments autonomously",
            "3. Clear metric decides what stays",
            "4. Repeat 100x while you sleep",
        ],
    }


@app.get("/domains")
async def list_domains():
    """List available domain adapters."""
    return {
        name: {"description": adapter.description}
        for name, adapter in DomainRegistry.all().items()
    }


@app.post("/domains/{domain}/template")
async def get_domain_template(domain: str):
    """Get a strategy template for a specific domain."""
    adapter = DomainRegistry.get(domain)
    if not adapter:
        raise HTTPException(404, f"Unknown domain: {domain}")
    return {"domain": domain, "template": adapter.get_strategy_template()}


@app.post("/loops")
async def start_loop(req: RunRequest):
    """Start an autonomous research loop."""
    loop_id = str(uuid.uuid4())[:8]

    if not req.strategy_content and not req.strategy_path:
        raise HTTPException(400, "Provide strategy_content or strategy_path")

    # Parse strategy
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
        "status": "running",
        "loop": loop,
        "events": events,
    }

    # Run in background
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
