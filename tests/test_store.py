"""Tests for SQLite persistent state store."""

import tempfile
from pathlib import Path

from gridmind.core.store import StateStore


def _make_store():
    """Create a store with a temp database."""
    tmpdir = tempfile.mkdtemp()
    return StateStore(Path(tmpdir) / "test.db")


def test_store_init():
    store = _make_store()
    assert store.db_path.exists()
    store.close()


def test_save_and_get_loop():
    store = _make_store()
    store.save_loop("loop1", "my-strategy", status="running")
    loop = store.get_loop("loop1")
    assert loop is not None
    assert loop["strategy_name"] == "my-strategy"
    assert loop["status"] == "running"
    store.close()


def test_update_loop_status():
    store = _make_store()
    store.save_loop("loop1", "strat", status="running")
    store.update_loop_status("loop1", "completed", iteration=50, best_iteration=42)
    loop = store.get_loop("loop1")
    assert loop["status"] == "completed"
    assert loop["current_iteration"] == 50
    assert loop["best_iteration"] == 42
    store.close()


def test_list_loops():
    store = _make_store()
    store.save_loop("a", "strat-a", status="running")
    store.save_loop("b", "strat-b", status="completed")
    loops = store.list_loops()
    assert len(loops) == 2
    store.close()


def test_save_and_get_experiments():
    store = _make_store()
    store.save_loop("loop1", "strat")
    store.save_experiment(
        "loop1", iteration=1,
        variables={"tone": "formal"},
        hypothesis="test",
        status="completed",
        metrics={"reply_rate": 0.05},
    )
    store.save_experiment(
        "loop1", iteration=2,
        variables={"tone": "casual"},
        status="rejected",
        metrics={"reply_rate": 0.03},
    )
    exps = store.get_experiments("loop1")
    assert len(exps) == 2
    assert exps[0]["variables"] == {"tone": "formal"}
    assert exps[0]["metrics"] == {"reply_rate": 0.05}
    store.close()


def test_save_and_get_metrics():
    store = _make_store()
    store.save_loop("loop1", "strat")
    store.save_metric("loop1", "reply_rate", 0.02, 0)
    store.save_metric("loop1", "reply_rate", 0.05, 1)
    store.save_metric("loop1", "reply_rate", 0.04, 2)

    history = store.get_metric_history("loop1", "reply_rate")
    assert len(history) == 3

    values = store.get_metric_values("loop1", "reply_rate")
    assert values == [0.02, 0.05, 0.04]
    store.close()


def test_webhook_events():
    store = _make_store()
    store.save_webhook_event(
        source="sendgrid",
        metrics={"open_rate": 0.42},
        loop_id="loop1",
        metadata={"campaign": "q1"},
    )
    store.save_webhook_event(
        source="mixpanel",
        metrics={"conversions": 100},
    )

    # All events
    events = store.get_webhook_events()
    assert len(events) == 2

    # Filter by source
    sg_events = store.get_webhook_events(source="sendgrid")
    assert len(sg_events) == 1
    assert sg_events[0]["metrics"]["open_rate"] == 0.42

    # Filter by loop
    loop_events = store.get_webhook_events(loop_id="loop1")
    assert len(loop_events) == 1
    store.close()


def test_bandit_state():
    store = _make_store()
    store.save_loop("loop1", "strat")

    state = {"method": "thompson", "total_experiments": 10, "bandits": {}}
    store.save_bandit_state("loop1", state)

    restored = store.get_bandit_state("loop1")
    assert restored is not None
    assert restored["method"] == "thompson"
    assert restored["total_experiments"] == 10

    # Update
    state["total_experiments"] = 20
    store.save_bandit_state("loop1", state)
    restored = store.get_bandit_state("loop1")
    assert restored["total_experiments"] == 20
    store.close()


def test_loop_summary():
    store = _make_store()
    store.save_loop("loop1", "strat", status="completed", best_iteration=5)
    store.save_experiment("loop1", 1, {"x": "a"}, status="completed",
                          metrics={"score": 0.5})
    store.save_experiment("loop1", 2, {"x": "b"}, status="rejected",
                          metrics={"score": 0.3})
    store.save_metric("loop1", "score", 0.5, 1, "higher")
    store.save_metric("loop1", "score", 0.3, 2, "higher")

    summary = store.get_loop_summary("loop1")
    assert summary is not None
    assert summary["total_experiments"] == 2
    assert "score" in summary["best_metrics"]
    store.close()


def test_upsert_loop():
    """save_loop should update on conflict."""
    store = _make_store()
    store.save_loop("loop1", "strat", status="running")
    store.save_loop("loop1", "strat", status="completed", best_iteration=10)
    loop = store.get_loop("loop1")
    assert loop["status"] == "completed"
    assert loop["best_iteration"] == 10
    store.close()


def test_nonexistent_loop():
    store = _make_store()
    assert store.get_loop("nope") is None
    assert store.get_bandit_state("nope") is None
    assert store.get_loop_summary("nope") is None
    store.close()
