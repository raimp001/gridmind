"""Tests for multi-armed bandit variable selection."""

from gridmind.core.bandit import ArmStats, VariableBandit, ExperimentBandit


def test_arm_stats_initial():
    arm = ArmStats(name="test")
    assert arm.pulls == 0
    assert arm.mean_reward == 0.0
    # Thompson sample should work with prior
    sample = arm.sample_thompson()
    assert 0.0 <= sample <= 1.0


def test_arm_stats_update():
    arm = ArmStats(name="test")
    arm.pulls = 5
    arm.total_reward = 4.0
    assert arm.mean_reward == 0.8


def test_arm_ucb1_untried():
    arm = ArmStats(name="test")
    assert arm.ucb1_score(10) == float("inf")


def test_arm_ucb1_tried():
    arm = ArmStats(name="test")
    arm.pulls = 5
    arm.total_reward = 4.0
    score = arm.ucb1_score(100)
    # Should be mean + exploration bonus
    assert score > arm.mean_reward


def test_variable_bandit_select():
    bandit = VariableBandit("tone", ["formal", "casual", "urgent"])
    # Should return one of the options
    for _ in range(20):
        val = bandit.select("thompson")
        assert val in ["formal", "casual", "urgent"]


def test_variable_bandit_converges():
    """After many updates, bandit should prefer the best arm."""
    bandit = VariableBandit("tone", ["good", "bad"])
    # Train: "good" always scores 0.9, "bad" always 0.1
    for _ in range(50):
        bandit.update("good", 0.9)
        bandit.update("bad", 0.1)

    # Sample 100 times, should pick "good" most of the time
    good_count = sum(1 for _ in range(100) if bandit.select("thompson") == "good")
    assert good_count > 70  # Should be heavily biased toward "good"


def test_variable_bandit_epsilon_greedy():
    bandit = VariableBandit("x", ["a", "b"])
    for _ in range(20):
        bandit.update("a", 0.9)
        bandit.update("b", 0.1)

    val = bandit.select("epsilon_greedy")
    assert val in ["a", "b"]


def test_experiment_bandit_select():
    variables = {
        "tone": ["formal", "casual"],
        "length": ["short", "long"],
    }
    bandit = ExperimentBandit(variables)
    selected = bandit.select_variables()
    assert "tone" in selected
    assert "length" in selected
    assert selected["tone"] in ["formal", "casual"]
    assert selected["length"] in ["short", "long"]


def test_experiment_bandit_update():
    variables = {"x": ["a", "b"], "y": ["1", "2"]}
    bandit = ExperimentBandit(variables)

    bandit.update({"x": "a", "y": "1"}, 0.8)
    assert bandit.total_experiments == 1
    assert bandit.exploration_coverage > 0


def test_experiment_bandit_force_explore():
    variables = {"x": ["a", "b"]}
    bandit = ExperimentBandit(variables)
    bandit.update({"x": "a"}, 0.5)

    # Force explore should try to find untried combos
    selected = bandit.select_variables(force_explore=True)
    assert selected["x"] in ["a", "b"]


def test_experiment_bandit_serialization():
    variables = {"tone": ["formal", "casual"], "length": ["short", "long"]}
    bandit = ExperimentBandit(variables, method="thompson")

    # Do some updates
    bandit.update({"tone": "formal", "length": "short"}, 0.8)
    bandit.update({"tone": "casual", "length": "long"}, 0.3)

    # Serialize
    data = bandit.to_dict()
    assert data["method"] == "thompson"
    assert data["total_experiments"] == 2

    # Deserialize
    restored = ExperimentBandit.from_dict(data, variables)
    assert restored.total_experiments == 2
    assert restored.method == "thompson"


def test_experiment_bandit_top_values():
    variables = {"x": ["a", "b", "c"]}
    bandit = ExperimentBandit(variables)
    for _ in range(20):
        bandit.update({"x": "a"}, 0.9)
        bandit.update({"x": "b"}, 0.5)
        bandit.update({"x": "c"}, 0.1)

    top = bandit.top_values(n=2)
    assert "x" in top
    assert top["x"][0] == "a"  # Best value should be first


def test_experiment_bandit_stats():
    variables = {"x": ["a", "b"]}
    bandit = ExperimentBandit(variables)
    bandit.update({"x": "a"}, 0.7)
    stats = bandit.stats()
    assert "x" in stats
    assert "a" in stats["x"]
    assert stats["x"]["a"]["pulls"] == 1
