"""CLI interface. Run research loops from the command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gridmind.core.agent import AgentConfig
from gridmind.core.loop import LoopConfig, ResearchLoop
from gridmind.core.orchestrator import (
    CampaignConfig,
    Orchestrator,
    DOMAIN_SIGNAL_SPEED,
    estimate_experiments,
)
from gridmind.domains.registry import DomainRegistry

console = Console()


SPEED_COLORS = {"fast": "green", "medium": "yellow", "slow": "blue"}


def _make_event_handler(verbose: bool):
    """Create a Rich-powered event handler for the loop."""

    def handle(event_type: str, data: dict):
        if event_type == "strategy_loaded":
            console.print(f"\n[bold green]Strategy loaded:[/] {data['name']}")

        elif event_type == "loop_started":
            console.print(
                Panel(
                    f"[bold]Running {data['max_iterations']} experiments autonomously[/]\n"
                    "Press Ctrl+C to pause and save progress",
                    title="Loop Started",
                    border_style="green",
                )
            )

        elif event_type == "iteration_started" and verbose:
            console.print(f"\n[dim]--- Iteration {data['iteration']} ---[/]")

        elif event_type == "new_best":
            metrics_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in data["metrics"].items()
            )
            console.print(
                f"  [bold green]NEW BEST[/] (iter {data['iteration']}): {metrics_str}"
            )
            if data.get("artifact_preview"):
                console.print(f"  [dim]{data['artifact_preview']}...[/]")

        elif event_type == "experiment_rejected" and verbose:
            metrics_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in data["metrics"].items()
            )
            console.print(f"  [dim]rejected[/]: {metrics_str}")

        elif event_type == "experiment_error":
            console.print(
                f"  [bold red]ERROR[/] (iter {data['iteration']}): {data['error']}"
            )

        elif event_type == "checkpoint_resumed":
            console.print(
                Panel(
                    f"[cyan]Resumed from checkpoint at iteration {data['iteration']}[/]\n"
                    f"Experiments restored: {data['experiments_restored']}",
                    border_style="cyan",
                )
            )

        elif event_type == "loop_converged":
            console.print(
                Panel(
                    f"[yellow]Converged at iteration {data['iteration']}[/]\n"
                    f"No improvement since iteration {data['last_improvement']}\n"
                    f"Window: {data['window']} iterations",
                    border_style="yellow",
                )
            )

        elif event_type == "loop_paused":
            console.print(
                Panel(
                    f"[yellow]Paused at iteration {data['iteration']}. Progress saved.[/]",
                    border_style="yellow",
                )
            )

        elif event_type == "loop_paused_failures":
            console.print(
                Panel(
                    f"[red]Paused at iteration {data['iteration']} after "
                    f"{data['consecutive_failures']} consecutive failures.[/]\n"
                    "Check logs, fix the issue, then resume with --resume.",
                    border_style="red",
                )
            )

        elif event_type == "loop_completed":
            table = Table(title="Final Results")
            table.add_column("Metric", style="bold")
            table.add_column("Best Value", style="green")
            table.add_column("Best Iteration")
            table.add_column("Total Measurements")
            for name, info in data["best_metrics"].items():
                table.add_row(
                    name,
                    f"{info['best_value']:.4f}",
                    str(info["best_iteration"]),
                    str(info["total_measurements"]),
                )
            console.print(table)

    return handle


def _make_campaign_handler(verbose: bool):
    """Event handler for multi-loop campaigns."""

    def handle(event_type: str, data: dict):
        if event_type == "strategy_added":
            speed = data["signal_speed"]
            color = SPEED_COLORS.get(speed, "white")
            console.print(
                f"  [{color}]{speed.upper():6s}[/] {data['name']} "
                f"({data['domain']}, {data['max_iterations']} iters)"
            )

        elif event_type == "campaign_started":
            by_speed = data.get("loops_by_speed", {})
            speed_str = " | ".join(f"{k}: {v}" for k, v in by_speed.items())
            console.print(
                Panel(
                    f"[bold]Running {data['total_loops']} loops in parallel[/]\n"
                    f"Signal tiers: {speed_str}\n"
                    "Press Ctrl+C to stop",
                    title="Campaign Started",
                    border_style="green",
                )
            )

        elif event_type == "loop_event":
            if data.get("type") == "new_best" and verbose:
                loop_name = data.get("loop", "?")
                metrics_str = " | ".join(
                    f"{k}: {v:.4f}" for k, v in data.get("metrics", {}).items()
                )
                console.print(
                    f"  [bold green]NEW BEST[/] [{loop_name}] "
                    f"iter {data.get('iteration', '?')}: {metrics_str}"
                )

        elif event_type == "loop_completed":
            name = data["name"]
            best = data.get("best_metrics", {})
            primary_metric = next(iter(best.values()), {}) if best else {}
            val = primary_metric.get("best_value", 0)
            console.print(
                f"  [green]DONE[/] {name} "
                f"({data.get('iterations', 0)} experiments, "
                f"best iter {data.get('best_iteration', 'N/A')}, "
                f"primary: {val:.4f})"
            )

        elif event_type == "loop_failed":
            console.print(
                f"  [red]FAILED[/] {data['name']}: {data['error']}"
            )

        elif event_type == "campaign_completed":
            console.print(
                Panel(
                    f"[bold]Campaign complete[/]\n"
                    f"Loops: {data['completed']}/{data['total_loops']} completed\n"
                    f"Total experiments: {data['total_experiments']}\n"
                    f"Cross-loop insights: {data['cross_loop_insights']}",
                    title="Campaign Results",
                    border_style="green",
                )
            )

    return handle


@click.group()
@click.version_option(version="0.2.0")
def main():
    """GridMind: Autonomous research loops for any domain.

    Write a strategy doc. Run experiments while you sleep.
    """
    pass


@main.command()
@click.argument("strategy_path", type=click.Path(exists=True))
@click.option("--results-dir", default="results", help="Directory to save results")
@click.option("--provider", default="anthropic", help="LLM provider: anthropic or openai")
@click.option("--model", default=None, help="Model name (default depends on provider)")
@click.option("--dry-run", is_flag=True, help="Run with mock data (no LLM calls)")
@click.option("--verbose", "-v", is_flag=True, help="Show all experiment details")
@click.option("--iterations", "-n", default=None, type=int, help="Override max iterations")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint")
@click.option(
    "--convergence-window", default=0, type=int,
    help="Stop if no improvement in N iterations (0 = disabled)",
)
@click.option(
    "--daemon", is_flag=True,
    help="Run as daemon with graceful shutdown, watchdog, and heartbeat",
)
@click.option(
    "--repeat", default=0, type=int,
    help="(Daemon) Repeat loop every N seconds (0 = run once)",
)
def run(
    strategy_path, results_dir, provider, model, dry_run, verbose,
    iterations, resume, convergence_window, daemon, repeat,
):
    """Run an autonomous research loop from a strategy document.

    STRATEGY_PATH is the path to your .md strategy file.

    Example:
        gridmind run strategies/cold-email.md --dry-run -v
        gridmind run strategies/cold-email.md --resume
        gridmind run strategies/cold-email.md --daemon --repeat 3600
        gridmind run strategies/ad-creative.md --convergence-window 20
    """
    default_models = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
    }

    agent_config = AgentConfig(
        provider=provider,
        model=model or default_models.get(provider, "claude-sonnet-4-20250514"),
    )

    config = LoopConfig(
        strategy_path=strategy_path,
        results_dir=results_dir,
        agent_config=agent_config,
        dry_run=dry_run,
        verbose=verbose,
        convergence_window=convergence_window,
    )

    loop = ResearchLoop(config)
    loop.on_event(_make_event_handler(verbose))

    # Override iterations if specified
    strategy = loop.load_strategy()
    if iterations:
        strategy.max_iterations = iterations

    mode = "daemon" if daemon else ("resume" if resume else "fresh")
    console.print(
        Panel(
            f"[bold]{strategy.name}[/]\n"
            f"Domain: {strategy.domain}\n"
            f"Objective: {strategy.objective}\n"
            f"Iterations: {strategy.max_iterations}\n"
            f"Provider: {provider} / {agent_config.model}\n"
            f"Dry run: {dry_run}\n"
            f"Mode: {mode}"
            + (f"\nConvergence window: {convergence_window}" if convergence_window else "")
            + (f"\nRepeat every: {repeat}s" if repeat else ""),
            title="GridMind Research Loop",
            border_style="blue",
        )
    )

    if daemon:
        from gridmind.core.daemon import DaemonConfig, LoopDaemon

        daemon_config = DaemonConfig(
            loop_config=config,
            heartbeat_path=f"{results_dir}/{strategy.name}/heartbeat.json",
            repeat_interval=repeat,
            pid_file=f"{results_dir}/{strategy.name}/gridmind.pid",
        )
        daemon_runner = LoopDaemon(daemon_config)
        console.print("[bold yellow]Running in daemon mode. SIGTERM to stop gracefully.[/]")
        state = daemon_runner.run()
    else:
        state = loop.run()

    if state:
        console.print(f"\n[bold]Results saved to:[/] {results_dir}/{strategy.name}/")
        if state.best_artifact:
            console.print(f"[bold]Best artifact:[/] {results_dir}/{strategy.name}/best_artifact.txt")
        if state.status == "converged":
            console.print(
                f"[bold yellow]Converged:[/] No improvement since iteration "
                f"{state.last_improvement_iteration} (window: {convergence_window})"
            )


@main.command()
@click.argument("strategies_dir", type=click.Path(exists=True))
@click.option("--name", default="campaign", help="Campaign name")
@click.option("--results-dir", default="results", help="Directory to save results")
@click.option("--provider", default="anthropic", help="LLM provider")
@click.option("--model", default=None, help="Model name")
@click.option("--workers", "-w", default=4, help="Max concurrent loops")
@click.option("--dry-run", is_flag=True, help="Run with mock data (no LLM calls)")
@click.option("--verbose", "-v", is_flag=True, help="Show all experiment details")
def campaign(strategies_dir, name, results_dir, provider, model, workers, dry_run, verbose):
    """Run multiple research loops in parallel across growth surfaces.

    STRATEGIES_DIR is a directory containing .md strategy files.

    Example:
        gridmind campaign strategies/ --dry-run -v
        gridmind campaign strategies/ --name q1-growth --workers 8
    """
    default_models = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
    }

    config = CampaignConfig(
        name=name,
        strategies_dir=strategies_dir,
        results_dir=results_dir,
        agent_config=AgentConfig(
            provider=provider,
            model=model or default_models.get(provider, "claude-sonnet-4-20250514"),
        ),
        max_workers=workers,
        dry_run=dry_run,
        verbose=verbose,
    )

    orchestrator = Orchestrator(config)
    orchestrator.on_event(_make_campaign_handler(verbose))

    console.print(
        Panel(
            f"[bold]{name}[/]\n"
            f"Loading strategies from: {strategies_dir}/\n"
            f"Workers: {workers} concurrent loops\n"
            f"Provider: {provider} / {config.agent_config.model}\n"
            f"Dry run: {dry_run}",
            title="GridMind Campaign",
            border_style="blue",
        )
    )

    keys = orchestrator.add_strategies_from_dir()
    if not keys:
        console.print("[red]No valid strategy files found.[/]")
        sys.exit(1)

    console.print(f"\n[bold]{len(keys)} strategies loaded[/]\n")

    results = orchestrator.run()

    # Show summary
    total_experiments = sum(
        len(s.experiments) for s in results.values()
    )
    console.print(f"\n[bold]Total experiments across all loops:[/] {total_experiments}")
    console.print(f"[bold]Results saved to:[/] {results_dir}/{name}/")

    # Show cross-loop insights
    if orchestrator.insights:
        console.print(f"\n[bold]Cross-loop insights discovered:[/] {len(orchestrator.insights)}")
        for insight in orchestrator.insights[:5]:
            console.print(
                f"  [cyan]{insight.source_domain}[/] -> "
                f"{', '.join(insight.target_domains)}: "
                f"{insight.insight_type}"
            )


@main.command()
@click.argument("domain", required=False)
@click.option("--output", "-o", default=None, help="Output path for generated template")
def init(domain, output):
    """Generate a strategy template for a domain.

    Example:
        gridmind init sales_pipeline -o strategies/my-sales-strategy.md
        gridmind init  # lists available domains
    """
    if not domain:
        console.print("[bold]Available domains (13):[/]\n")
        for name, adapter in DomainRegistry.all().items():
            speed = DOMAIN_SIGNAL_SPEED.get(name, "medium")
            if hasattr(speed, "value"):
                speed = speed.value
            color = SPEED_COLORS.get(speed, "white")
            console.print(f"  [{color}]{speed.upper():6s}[/] [green]{name}[/] - {adapter.description}")
        console.print(
            "\nUsage: [bold]gridmind init <domain> -o strategies/my-strategy.md[/]"
        )
        return

    adapter = DomainRegistry.get(domain)
    if not adapter:
        console.print(f"[red]Unknown domain:[/] {domain}")
        console.print(f"Available: {', '.join(DomainRegistry.list_domains())}")
        sys.exit(1)

    template = adapter.get_strategy_template()

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(template)
        console.print(f"[green]Strategy template written to:[/] {output}")
        console.print("Edit the file, then run: [bold]gridmind run {output}[/]")
    else:
        console.print(template)


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
def report(results_dir):
    """View results from a completed research loop or campaign.

    Example:
        gridmind report results/cold-email-optimizer/
        gridmind report results/campaign/
    """
    results_path = Path(results_dir)

    # Check if this is a campaign (has campaign_summary.json)
    campaign_summary_path = results_path / "campaign_summary.json"
    if campaign_summary_path.exists():
        _report_campaign(results_path)
        return

    summary_path = results_path / "summary.json"
    if not summary_path.exists():
        console.print(f"[red]No summary.json found in {results_dir}[/]")
        sys.exit(1)

    summary = json.loads(summary_path.read_text())

    console.print(
        Panel(
            f"[bold]{summary.get('strategy', 'Unknown')}[/]\n"
            f"Total iterations: {summary.get('total_iterations', 0)}\n"
            f"Completed: {summary.get('completed', 0)}\n"
            f"Rejected: {summary.get('rejected', 0)}\n"
            f"Failed: {summary.get('failed', 0)}\n"
            f"Best iteration: {summary.get('best_iteration', 'N/A')}",
            title="Research Loop Results",
            border_style="blue",
        )
    )

    best_metrics = summary.get("best_metrics", {})
    if best_metrics:
        table = Table(title="Best Metrics")
        table.add_column("Metric", style="bold")
        table.add_column("Best Value", style="green")
        table.add_column("Direction")
        table.add_column("Best Iteration")
        for name, info in best_metrics.items():
            table.add_row(
                name,
                f"{info['best_value']:.4f}",
                info["direction"],
                str(info["best_iteration"]),
            )
        console.print(table)

    artifact_path = results_path / "best_artifact.txt"
    if artifact_path.exists():
        console.print(
            Panel(
                artifact_path.read_text()[:1000],
                title="Best Artifact",
                border_style="green",
            )
        )


def _report_campaign(results_path: Path):
    """Report on a multi-loop campaign."""
    summary = json.loads((results_path / "campaign_summary.json").read_text())

    console.print(
        Panel(
            f"[bold]{summary.get('name', 'Unknown Campaign')}[/]\n"
            f"Total loops: {summary.get('total_loops', 0)}\n"
            f"Total experiments: {summary.get('total_experiments', 0)}\n"
            f"Cross-loop insights: {summary.get('cross_loop_insights', 0)}",
            title="Campaign Results",
            border_style="blue",
        )
    )

    # Loops table
    loops = summary.get("loops", {})
    if loops:
        table = Table(title="Loop Results")
        table.add_column("Loop", style="bold")
        table.add_column("Domain")
        table.add_column("Signal")
        table.add_column("Experiments", justify="right")
        table.add_column("Best Iter", justify="right")
        table.add_column("Status")

        for name, info in loops.items():
            speed = info.get("signal_speed", "?")
            color = SPEED_COLORS.get(speed, "white")
            status_style = "green" if info.get("status") == "completed" else "red"
            table.add_row(
                name,
                info.get("domain", "?"),
                f"[{color}]{speed}[/]",
                str(info.get("experiments", 0)),
                str(info.get("best_iteration", "-")),
                f"[{status_style}]{info.get('status', '?')}[/]",
            )
        console.print(table)

    # Cross-loop insights
    insights_path = results_path / "cross_loop_insights.json"
    if insights_path.exists():
        insights = json.loads(insights_path.read_text())
        if insights:
            console.print(f"\n[bold]Cross-Loop Insights ({len(insights)}):[/]")
            for i in insights[:10]:
                console.print(
                    f"  [cyan]{i['source_domain']}[/] -> "
                    f"{', '.join(i['target_domains'])}: "
                    f"{i['insight_type']}"
                )
                console.print(f"    [dim]{i['insight'][:150]}...[/]")


@main.command()
def domains():
    """List all available domain adapters with signal speed tiers."""
    console.print("[bold]Available Domains (13)[/]\n")

    # Group by signal speed
    by_speed: dict[str, list] = {"fast": [], "medium": [], "slow": []}
    for name, adapter in DomainRegistry.all().items():
        speed = DOMAIN_SIGNAL_SPEED.get(name)
        if speed:
            by_speed[speed.value].append((name, adapter))
        else:
            by_speed["medium"].append((name, adapter))

    labels = {
        "fast": "FAST SIGNAL (24-72h scoring)",
        "medium": "MEDIUM SIGNAL (7-14d scoring)",
        "slow": "SLOW SIGNAL (30-90d scoring)",
    }

    for speed, domains_list in by_speed.items():
        if not domains_list:
            continue
        color = SPEED_COLORS[speed]
        console.print(f"  [{color}]{labels[speed]}[/]")
        for name, adapter in domains_list:
            console.print(f"    [green]{name}[/] - {adapter.description}")
        console.print()


@main.command()
def math():
    """Show the experiment math: how many experiments you'll run vs competitors."""
    est = estimate_experiments()

    console.print(
        Panel(
            f"[bold]Loops:[/] {est['loops']} domains\n"
            f"[bold]Per loop:[/] {est['experiments_per_loop_per_run']:.0f} experiments/run\n"
            f"[bold]Daily:[/] {est['daily_experiments']:.0f} experiments\n"
            f"[bold]Yearly:[/] {est['yearly_experiments']:,.0f} experiments\n"
            f"\n[bold green]{est['vs_traditional']}[/]\n"
            f"\n[dim]Signal tiers:[/]\n"
            f"  Fast (24-72h):  {est['signal_tiers']['fast_loops']} loops\n"
            f"  Medium (7-14d): {est['signal_tiers']['medium_loops']} loops\n"
            f"  Slow (30-90d):  {est['signal_tiers']['slow_loops']} loops",
            title="The Math",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
