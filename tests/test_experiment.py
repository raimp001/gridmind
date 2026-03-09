"""Tests for experiment execution."""

from gridmind.core.experiment import Experiment, ExperimentResult, ExperimentStatus


def test_experiment_lifecycle():
    exp = Experiment(iteration=1, variables={"tone": "casual"})
    assert exp.status == ExperimentStatus.PENDING

    exp.start()
    assert exp.status == ExperimentStatus.RUNNING
    assert exp.started_at is not None

    result = ExperimentResult(
        metrics={"reply_rate": 0.05},
        artifacts={"email": "Hey there!"},
    )
    exp.complete(result)
    assert exp.status == ExperimentStatus.COMPLETED
    assert exp.duration_seconds >= 0


def test_experiment_failure():
    exp = Experiment(iteration=1, variables={})
    exp.start()
    result = ExperimentResult(error="LLM returned invalid JSON")
    exp.complete(result)
    assert exp.status == ExperimentStatus.FAILED
    assert not result.succeeded


def test_experiment_rejection():
    exp = Experiment(iteration=1, variables={"tone": "casual"})
    exp.start()
    result = ExperimentResult(metrics={"reply_rate": 0.01})
    exp.complete(result)
    exp.reject()
    assert exp.status == ExperimentStatus.REJECTED


def test_experiment_to_dict():
    exp = Experiment(iteration=5, variables={"tone": "casual", "length": "short"})
    exp.start()
    result = ExperimentResult(metrics={"reply_rate": 0.08})
    exp.complete(result)

    d = exp.to_dict()
    assert d["iteration"] == 5
    assert d["variables"] == {"tone": "casual", "length": "short"}
    assert d["metrics"] == {"reply_rate": 0.08}
    assert d["status"] == "completed"


def test_result_succeeded():
    r1 = ExperimentResult(metrics={"reply_rate": 0.05})
    assert r1.succeeded is True

    r2 = ExperimentResult(error="something broke")
    assert r2.succeeded is False

    r3 = ExperimentResult()  # no metrics, no error
    assert r3.succeeded is False
