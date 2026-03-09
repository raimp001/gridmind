"""CLI interface. Run research loops from the command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gridmind.core.agent import AgentConfig
from gridmind.core.loop import LoopConfig, ResearchLoop
from gridmind.domains.registry import DomainRegistry

console = Console()


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

        elif event_type == "loop_paused":
            console.print(
                Panel(
                    f"[yellow]Paused at iteration {data['iteration']}. Progress saved.[/]",
                    border_style="yellow",
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


@click.group()
@click.version_option(version="0.1.0")
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
def run(strategy_path, results_dir, provider, model, dry_run, verbose, iterations):
    """Run an autonomous research loop from a strategy document.

    STRATEGY_PATH is the path to your .md strategy file.

    Example:
        gridmind run strategies/cold-email.md --dry-run -v
        gridmind run strategies/ad-creative.md --provider anthropic -n 50
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
    )

    loop = ResearchLoop(config)
    loop.on_event(_make_event_handler(verbose))

    # Override iterations if specified
    strategy = loop.load_strategy()
    if iterations:
        strategy.max_iterations = iterations

    console.print(
        Panel(
            f"[bold]{strategy.name}[/]\n"
            f"Domain: {strategy.domain}\n"
            f"Objective: {strategy.objective}\n"
            f"Iterations: {strategy.max_iterations}\n"
            f"Provider: {provider} / {agent_config.model}\n"
            f"Dry run: {dry_run}",
            title="GridMind Research Loop",
            border_style="blue",
        )
    )

    state = loop.run()

    console.print(f"\n[bold]Results saved to:[/] {results_dir}/{strategy.name}/")
    if state.best_artifact:
        console.print(f"[bold]Best artifact:[/] {results_dir}/{strategy.name}/best_artifact.txt")


@main.command()
@click.argument("domain", required=False)
@click.option("--output", "-o", default=None, help="Output path for generated template")
def init(domain, output):
    """Generate a strategy template for a domain.

    Available domains: sales_pipeline, ad_creative, lead_gen, client_onboarding

    Example:
        gridmind init sales_pipeline -o strategies/my-sales-strategy.md
        gridmind init  # lists available domains
    """
    if not domain:
        console.print("[bold]Available domains:[/]\n")
        for name, adapter in DomainRegistry.all().items():
            console.print(f"  [green]{name}[/] - {adapter.description}")
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
    """View results from a completed research loop.

    Example:
        gridmind report results/cold-email-optimizer/
    """
    results_path = Path(results_dir)

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

    # Metrics table
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

    # Best artifact
    artifact_path = results_path / "best_artifact.txt"
    if artifact_path.exists():
        console.print(
            Panel(
                artifact_path.read_text()[:1000],
                title="Best Artifact",
                border_style="green",
            )
        )


@main.command()
def domains():
    """List all available domain adapters."""
    console.print("[bold]Available Domains[/]\n")
    for name, adapter in DomainRegistry.all().items():
        console.print(f"  [green]{name}[/]")
        console.print(f"    {adapter.description}\n")


if __name__ == "__main__":
    main()
