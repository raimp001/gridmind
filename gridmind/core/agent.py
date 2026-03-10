"""AI agent interface. The agent reads the strategy doc and runs experiments.

Production-hardened: retries, rate limiting, robust JSON parsing,
adaptive temperature, timeout handling.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

from gridmind.core.experiment import Experiment
from gridmind.core.strategy import Strategy

logger = logging.getLogger(__name__)


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
    bandit_suggestion: dict[str, str] | None = None,
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

    # Only include last 10 results to stay within token limits
    if past_results:
        recent = past_results[-10:]
        parts.append(f"\n## Recent experiment results (last {len(recent)})")
        for r in recent:
            status = r.get("status", "unknown")
            metrics_str = json.dumps(r.get("metrics", {}))
            parts.append(
                f"- Iter {r['iteration']} [{status}]: {metrics_str} | "
                f"vars={json.dumps(r.get('variables', {}))}"
            )

    # Bandit-recommended variables (from Thompson Sampling)
    if bandit_suggestion:
        parts.append("\n## Recommended variables (from optimization algorithm)")
        parts.append(
            "The following variable values are statistically likely to perform well "
            "based on past results. You may use them or deviate if you have a strong hypothesis."
        )
        for var, val in bandit_suggestion.items():
            parts.append(f"- **{var}**: {val}")

    parts.append(f"\n## Experiment Instructions\n\n{strategy.experiment_template}")
    parts.append(
        "\n\nRespond with a JSON object. "
        "Be bold in early iterations, refine in later ones."
    )

    return "\n".join(parts)


# --- Robust JSON extraction ---

def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response text. Handles common issues:
    - Markdown code blocks
    - Leading/trailing text around JSON
    - Missing closing braces
    - Single quotes instead of double
    """
    text = text.strip()

    # Strip markdown code blocks
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Try fixing common issues: single quotes, trailing commas
        fixed = candidate.replace("'", '"')
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Try adding missing closing braces
        open_braces = candidate.count("{") - candidate.count("}")
        if open_braces > 0:
            try:
                return json.loads(candidate + "}" * open_braces)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}...")


def _validate_design(data: dict) -> dict:
    """Validate and normalize a design response."""
    return {
        "hypothesis": data.get("hypothesis", ""),
        "approach": data.get("approach", ""),
        "variables": data.get("variables", {}),
        "artifact": data.get("artifact", ""),
        "predicted_metrics": data.get("predicted_metrics", {}),
    }


def _validate_scores(data: dict, strategy: Strategy) -> dict[str, float]:
    """Validate and normalize evaluation scores."""
    scores = {}
    for m in strategy.metrics:
        val = data.get(m.name)
        if val is not None:
            try:
                scores[m.name] = float(val)
            except (TypeError, ValueError):
                logger.warning("Invalid score for %s: %s", m.name, val)
    return scores


# --- Retry logic ---

class RetryError(Exception):
    """All retry attempts exhausted."""

    def __init__(self, message: str, last_error: Exception | None = None):
        super().__init__(message)
        self.last_error = last_error


def _retry_call(fn, max_attempts: int = 4, base_delay: float = 2.0) -> str:
    """Call fn() with exponential backoff. Returns the LLM response text.

    Handles:
    - Network errors (retry)
    - Rate limits / 429 (retry with longer backoff)
    - Server errors / 500 (retry)
    - Auth errors (fail fast, no retry)
    """
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            error_type = type(e).__name__

            # Don't retry auth errors
            if "auth" in error_str or "api key" in error_str or "401" in error_str:
                raise

            # Rate limit: longer backoff
            if "429" in error_str or "rate" in error_str:
                delay = base_delay * (4 ** attempt)  # 8s, 32s, 128s
                logger.warning(
                    "Rate limited (attempt %d/%d), waiting %.0fs: %s",
                    attempt, max_attempts, delay, error_type,
                )
            else:
                delay = base_delay * (2 ** (attempt - 1))  # 2s, 4s, 8s
                logger.warning(
                    "API error (attempt %d/%d), retrying in %.0fs: %s: %s",
                    attempt, max_attempts, delay, error_type, str(e)[:100],
                )

            if attempt < max_attempts:
                time.sleep(delay)

    raise RetryError(
        f"All {max_attempts} attempts failed. Last error: {last_error}",
        last_error=last_error,
    )


# --- Adaptive temperature ---

def adaptive_temperature(
    iteration: int,
    max_iterations: int,
    base_temp: float = 0.9,
    min_temp: float = 0.3,
) -> float:
    """Decay temperature from exploration (high) to exploitation (low).

    Early iterations: high temp = creative, diverse experiments.
    Late iterations: low temp = refine what works.
    """
    progress = iteration / max(max_iterations, 1)
    return base_temp - (base_temp - min_temp) * progress


# --- Agent ---

@dataclass
class AgentConfig:
    provider: str = "anthropic"  # "anthropic" or "openai"
    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None
    temperature: float = 0.8  # base temperature (will be adapted per iteration)
    max_tokens: int = 4096
    max_retries: int = 4
    retry_base_delay: float = 2.0
    timeout_seconds: float = 60.0
    adaptive_temp: bool = True  # enable temperature decay


class ResearchAgent:
    """Calls an LLM to design and run each experiment.

    Production-hardened with retries, rate limiting, robust JSON parsing,
    and adaptive temperature.
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self._client = None

    def _get_client(self):
        if self._client:
            return self._client

        if self.config.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self.config.api_key or os.environ.get("ANTHROPIC_API_KEY"),
                timeout=self.config.timeout_seconds,
            )
        elif self.config.provider == "openai":
            import openai

            self._client = openai.OpenAI(
                api_key=self.config.api_key or os.environ.get("OPENAI_API_KEY"),
                timeout=self.config.timeout_seconds,
            )
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
        return self._client

    def _get_temperature(self, iteration: int, max_iterations: int) -> float:
        """Get temperature for this iteration (adaptive or fixed)."""
        if self.config.adaptive_temp:
            return adaptive_temperature(
                iteration, max_iterations,
                base_temp=min(self.config.temperature + 0.1, 1.0),
                min_temp=max(self.config.temperature - 0.5, 0.2),
            )
        return self.config.temperature

    def _call_llm(
        self,
        system: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the LLM with retries. Returns raw response text."""
        client = self._get_client()

        def _do_call() -> str:
            if self.config.provider == "anthropic":
                response = client.messages.create(
                    model=self.config.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=self.config.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content

        return _retry_call(
            _do_call,
            max_attempts=self.config.max_retries,
            base_delay=self.config.retry_base_delay,
        )

    def design_experiment(
        self,
        strategy: Strategy,
        iteration: int,
        past_results: list[dict],
        best_metrics: dict[str, dict],
        bandit_suggestion: dict[str, str] | None = None,
    ) -> dict:
        """Ask the LLM to design the next experiment.

        Returns validated design dict with retries and JSON repair.
        """
        prompt = _build_experiment_prompt(
            strategy, iteration, past_results, best_metrics, bandit_suggestion,
        )
        temp = self._get_temperature(iteration, strategy.max_iterations)

        text = self._call_llm(SYSTEM_PROMPT, prompt, temp, self.config.max_tokens)
        data = _extract_json(text)
        return _validate_design(data)

    def evaluate_experiment(
        self,
        strategy: Strategy,
        experiment: Experiment,
        artifact: str,
    ) -> dict[str, float]:
        """Ask the LLM to evaluate/score the experiment artifact."""
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
            'Return ONLY a JSON object: {"metric_name": score, ...}'
        )

        text = self._call_llm(
            "You are a critical evaluator. Score artifacts honestly.",
            eval_prompt,
            temperature=0.3,
            max_tokens=1024,
        )

        data = _extract_json(text)
        scores = _validate_scores(data, strategy)

        if not scores:
            raise ValueError(
                f"LLM returned no valid scores. Expected metrics: "
                f"{[m.name for m in strategy.metrics]}, got: {data}"
            )

        return scores
