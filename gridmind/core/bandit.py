"""Multi-armed bandit for variable selection.

Implements Thompson Sampling to balance exploration (trying new variable combos)
vs exploitation (repeating what works). Replaces the naive "let the LLM pick
randomly" approach with a principled optimization algorithm.

The bandit tracks which variable values produce high scores and biases future
selections toward winners while still exploring alternatives.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class ArmStats:
    """Statistics for a single arm (variable value)."""

    name: str
    successes: float = 1.0  # Beta prior alpha (start at 1 for uniform prior)
    failures: float = 1.0   # Beta prior beta
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.total_reward / self.pulls

    def sample_thompson(self) -> float:
        """Draw from Beta(successes, failures) for Thompson Sampling."""
        return random.betavariate(max(self.successes, 0.01), max(self.failures, 0.01))

    def ucb1_score(self, total_pulls: int) -> float:
        """Upper Confidence Bound score for UCB1 algorithm."""
        if self.pulls == 0:
            return float("inf")
        exploitation = self.mean_reward
        exploration = math.sqrt(2 * math.log(max(total_pulls, 1)) / self.pulls)
        return exploitation + exploration


class VariableBandit:
    """Multi-armed bandit for a single variable (e.g., "subject_style").

    Each possible value is an arm. Thompson Sampling is used to select
    the next value to try.
    """

    def __init__(self, variable_name: str, options: list[str]):
        self.variable_name = variable_name
        self.arms: dict[str, ArmStats] = {
            opt: ArmStats(name=opt) for opt in options
        }
        self.total_pulls = 0

    def select(self, method: str = "thompson") -> str:
        """Select the next value to try.

        Methods:
        - "thompson": Thompson Sampling (default, best for this use case)
        - "ucb1": Upper Confidence Bound
        - "epsilon_greedy": Epsilon-greedy with epsilon=0.1
        """
        if method == "thompson":
            return self._select_thompson()
        elif method == "ucb1":
            return self._select_ucb1()
        elif method == "epsilon_greedy":
            return self._select_epsilon_greedy()
        else:
            return self._select_thompson()

    def _select_thompson(self) -> str:
        """Thompson Sampling: sample from each arm's Beta distribution."""
        best_value = ""
        best_sample = -1.0
        for name, arm in self.arms.items():
            sample = arm.sample_thompson()
            if sample > best_sample:
                best_sample = sample
                best_value = name
        return best_value

    def _select_ucb1(self) -> str:
        """UCB1: pick arm with highest upper confidence bound."""
        best_value = ""
        best_score = -1.0
        for name, arm in self.arms.items():
            score = arm.ucb1_score(self.total_pulls)
            if score > best_score:
                best_score = score
                best_value = name
        return best_value

    def _select_epsilon_greedy(self, epsilon: float = 0.1) -> str:
        """Epsilon-greedy: exploit best arm with prob 1-epsilon, explore otherwise."""
        if random.random() < epsilon or self.total_pulls == 0:
            return random.choice(list(self.arms.keys()))
        # Exploit: pick arm with highest mean reward
        return max(self.arms.values(), key=lambda a: a.mean_reward).name

    def update(self, value: str, reward: float):
        """Update arm statistics after observing a reward.

        Args:
            value: The variable value that was used.
            reward: Reward signal in [0, 1]. Higher = better.
        """
        if value not in self.arms:
            return

        arm = self.arms[value]
        arm.pulls += 1
        arm.total_reward += reward
        self.total_pulls += 1

        # Update Beta distribution parameters
        # Treat reward as probability of success
        arm.successes += reward
        arm.failures += (1.0 - reward)

    def stats(self) -> dict[str, dict]:
        """Get statistics for all arms."""
        return {
            name: {
                "pulls": arm.pulls,
                "mean_reward": round(arm.mean_reward, 4),
                "successes": round(arm.successes, 2),
                "failures": round(arm.failures, 2),
                "thompson_sample": round(arm.sample_thompson(), 4),
            }
            for name, arm in self.arms.items()
        }


class ExperimentBandit:
    """Multi-armed bandit across all variables in a strategy.

    Maintains a separate bandit per variable. When selecting the next
    experiment, each variable is independently sampled from its bandit.
    """

    def __init__(
        self,
        variables: dict[str, list[str]],
        method: str = "thompson",
    ):
        self.bandits: dict[str, VariableBandit] = {
            var_name: VariableBandit(var_name, options)
            for var_name, options in variables.items()
        }
        self.method = method
        self._explored_combos: set[tuple] = set()
        self.total_experiments = 0

    def select_variables(self, force_explore: bool = False) -> dict[str, str]:
        """Select variable values for the next experiment.

        Args:
            force_explore: If True, force a combo we haven't tried yet.

        Returns:
            Dict of {variable_name: selected_value}.
        """
        if force_explore:
            return self._select_unexplored()

        selected = {}
        for var_name, bandit in self.bandits.items():
            selected[var_name] = bandit.select(self.method)
        return selected

    def _select_unexplored(self, max_attempts: int = 50) -> dict[str, str]:
        """Try to find a variable combo we haven't explored yet."""
        for _ in range(max_attempts):
            selected = {}
            for var_name, bandit in self.bandits.items():
                selected[var_name] = bandit.select(self.method)
            combo_key = tuple(sorted(selected.items()))
            if combo_key not in self._explored_combos:
                return selected

        # Fallback: just use bandit selection
        return {
            var_name: bandit.select(self.method)
            for var_name, bandit in self.bandits.items()
        }

    def update(self, variables: dict[str, str], reward: float):
        """Update bandits after observing experiment results.

        Args:
            variables: The variable values used in the experiment.
            reward: Overall reward in [0, 1].
        """
        self.total_experiments += 1
        combo_key = tuple(sorted(variables.items()))
        self._explored_combos.add(combo_key)

        for var_name, value in variables.items():
            if var_name in self.bandits:
                self.bandits[var_name].update(value, reward)

    @property
    def exploration_coverage(self) -> float:
        """Fraction of total possible combos that have been explored."""
        total_combos = 1
        for bandit in self.bandits.values():
            total_combos *= len(bandit.arms)
        if total_combos == 0:
            return 1.0
        return len(self._explored_combos) / total_combos

    def stats(self) -> dict[str, dict]:
        """Get statistics for all variables."""
        return {
            var_name: bandit.stats()
            for var_name, bandit in self.bandits.items()
        }

    def top_values(self, n: int = 3) -> dict[str, list[str]]:
        """Get top-performing values for each variable."""
        result = {}
        for var_name, bandit in self.bandits.items():
            sorted_arms = sorted(
                bandit.arms.values(),
                key=lambda a: a.mean_reward,
                reverse=True,
            )
            result[var_name] = [a.name for a in sorted_arms[:n]]
        return result

    def to_dict(self) -> dict:
        """Serialize bandit state for checkpointing."""
        return {
            "method": self.method,
            "total_experiments": self.total_experiments,
            "explored_combos": [list(c) for c in self._explored_combos],
            "bandits": {
                var_name: {
                    name: {
                        "successes": arm.successes,
                        "failures": arm.failures,
                        "pulls": arm.pulls,
                        "total_reward": arm.total_reward,
                    }
                    for name, arm in bandit.arms.items()
                }
                for var_name, bandit in self.bandits.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict, variables: dict[str, list[str]]) -> ExperimentBandit:
        """Restore bandit state from checkpoint."""
        bandit = cls(variables, method=data.get("method", "thompson"))
        bandit.total_experiments = data.get("total_experiments", 0)
        bandit._explored_combos = {
            tuple(tuple(pair) for pair in c) for c in data.get("explored_combos", [])
        }

        for var_name, arms_data in data.get("bandits", {}).items():
            if var_name not in bandit.bandits:
                continue
            for arm_name, arm_data in arms_data.items():
                if arm_name in bandit.bandits[var_name].arms:
                    arm = bandit.bandits[var_name].arms[arm_name]
                    arm.successes = arm_data.get("successes", 1.0)
                    arm.failures = arm_data.get("failures", 1.0)
                    arm.pulls = arm_data.get("pulls", 0)
                    arm.total_reward = arm_data.get("total_reward", 0.0)
                    bandit.bandits[var_name].total_pulls += arm.pulls

        return bandit
