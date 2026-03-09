"""AI agent interface. The agent reads the strategy doc and runs experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from gridmind.core.experiment import Experiment, ExperimentResult
from gridmind.core.strategy import Strategy


SYSTEM_PROMPT = """You are an autonomous research agent running experiments in a loop.
You are given a strategy document that defines:
- The domain and objective
- The metrics to optimize
- The variables to experiment with
- The experiment template / instructions

Your job for each iteration:
1. Review past results and the current best metrics
2. Form a hypothesis about what change will improve the primary metric
3. Design a concrete experiment (choose variable values, write the artifact)
4. Predict expected metric values

You must respond with valid JSON matching this schema:
{
  "hypothesis": "string - why you think this will work",
  "approach": "string - what you're doing differently",
  "variables": {"var_name": "chosen_value", ...},
  "artifact": "string - the actual content (email copy, ad creative, script, etc.)",
  "predicted_metrics": {"metric_name": predicted_value, ...}
}

Be creative. Take risks. Learn from failures. The loop runs 100x - early experiments
should explore broadly, later experiments should exploit what works."""


def _build_experiment_prompt(
    strategy: Strategy,
    iteration: int,
    past_results: list[dict],
    best_metrics: dict[str, dict],
) -> str:
    parts = [
        f"# Strategy: {strategy.name}",
        f"**Domain:** {strategy.domain}",
        f"**Objective:** {strategy.objective}",
        f"**Iteration:** {iteration} / {strategy.max_iterations}",
        "",
        "## Metrics to optimize",
    ]
    for m in strategy.metrics:
        best = best_metrics.get(m.name, {})
        parts.append(
            f"- **{m.name}** ({m.direction} is better) | "
            f"baseline: {m.baseline} | best so far: {best.get('best_value', 'N/A')}"
        )

    if strategy.constraints:
        parts.append("\n## Constraints")
        for c in strategy.constraints:
            parts.append(f"- {c}")

    parts.append("\n## Variables to experiment with")
    for var, options in strategy.variables.items():
        parts.append(f"- **{var}**: {options}")

    if past_results:
        parts.append("\n## Recent experiment results (last 10)")
        for r in past_results[-10:]:
            status = r.get("status", "unknown")
            metrics_str = json.dumps(r.get("metrics", {}))
            parts.append(
                f"- Iter {r['iteration']} [{status}]: {metrics_str} | "
                f"vars={json.dumps(r.get('variables', {}))}"
            )

    parts.append(f"\n## Experiment Instructions\n\n{strategy.experiment_template}")
    parts.append(
        "\n\nRespond with a JSON object. "
        "Be bold in early iterations, refine in later ones."
    )

    return "\n".join(parts)


@dataclass
class AgentConfig:
    provider: str = "anthropic"  # "anthropic" or "openai"
    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None
    temperature: float = 0.8  # high creativity for exploration
    max_tokens: int = 4096


class ResearchAgent:
    """Calls an LLM to design and run each experiment."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self._client = None

    def _get_client(self):
        if self._client:
            return self._client

        if self.config.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
        elif self.config.provider == "openai":
            import openai

            self._client = openai.OpenAI(
                api_key=self.config.api_key or os.environ.get("OPENAI_API_KEY")
            )
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
        return self._client

    def design_experiment(
        self,
        strategy: Strategy,
        iteration: int,
        past_results: list[dict],
        best_metrics: dict[str, dict],
    ) -> dict:
        """Ask the LLM to design the next experiment."""
        prompt = _build_experiment_prompt(strategy, iteration, past_results, best_metrics)
        client = self._get_client()

        if self.config.provider == "anthropic":
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        else:
            response = client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content

        # Extract JSON from response (handle markdown code blocks)
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        return json.loads(text)

    def evaluate_experiment(
        self,
        strategy: Strategy,
        experiment: Experiment,
        artifact: str,
    ) -> dict[str, float]:
        """Ask the LLM to evaluate/score the experiment artifact against metrics.

        For domains without real-world execution (simulated mode), the LLM
        acts as both generator and evaluator.
        """
        client = self._get_client()
        eval_prompt = (
            f"You are evaluating an experiment artifact for: {strategy.objective}\n\n"
            f"## Artifact\n\n{artifact}\n\n"
            f"## Variables used\n{json.dumps(experiment.variables)}\n\n"
            f"## Metrics to score (0.0 to 1.0 scale unless otherwise noted)\n"
        )
        for m in strategy.metrics:
            eval_prompt += f"- {m.name} ({m.direction} is better)\n"

        eval_prompt += (
            "\nScore this artifact realistically. Be critical. "
            "Return ONLY a JSON object: {\"metric_name\": score, ...}"
        )

        if self.config.provider == "anthropic":
            response = client.messages.create(
                model=self.config.model,
                max_tokens=1024,
                temperature=0.3,  # low temp for evaluation
                system="You are a critical evaluator. Score artifacts honestly.",
                messages=[{"role": "user", "content": eval_prompt}],
            )
            text = response.content[0].text
        else:
            response = client.chat.completions.create(
                model=self.config.model,
                max_tokens=1024,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "You are a critical evaluator. Score artifacts honestly."},
                    {"role": "user", "content": eval_prompt},
                ],
            )
            text = response.choices[0].message.content

        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        return json.loads(text)
