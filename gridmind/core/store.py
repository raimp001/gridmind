"""SQLite persistent state store.

Replaces in-memory dicts in the server. Survives server restarts,
stores loop state, experiment history, webhook events, and bandit state.

Zero external dependencies — uses Python's built-in sqlite3.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS loops (
    id TEXT PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    strategy_content TEXT,
    status TEXT DEFAULT 'idle',
    current_iteration INTEGER DEFAULT 0,
    best_iteration INTEGER DEFAULT 0,
    best_artifact TEXT DEFAULT '',
    config_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    loop_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    variables_json TEXT,
    hypothesis TEXT DEFAULT '',
    approach TEXT DEFAULT '',
    status TEXT DEFAULT 'pending',
    metrics_json TEXT DEFAULT '{}',
    artifacts_json TEXT DEFAULT '{}',
    error TEXT,
    duration_seconds REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (loop_id) REFERENCES loops(id)
);

CREATE TABLE IF NOT EXISTS metrics_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    loop_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    iteration INTEGER NOT NULL,
    direction TEXT DEFAULT 'higher',
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (loop_id) REFERENCES loops(id)
);

CREATE TABLE IF NOT EXISTS webhook_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    loop_id TEXT,
    experiment_iteration INTEGER,
    metrics_json TEXT NOT NULL,
    metadata_json TEXT,
    timestamp TEXT NOT NULL,
    received_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bandit_state (
    loop_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (loop_id) REFERENCES loops(id)
);

CREATE INDEX IF NOT EXISTS idx_experiments_loop ON experiments(loop_id, iteration);
CREATE INDEX IF NOT EXISTS idx_metrics_loop ON metrics_history(loop_id, name);
CREATE INDEX IF NOT EXISTS idx_webhook_source ON webhook_events(source);
CREATE INDEX IF NOT EXISTS idx_webhook_loop ON webhook_events(loop_id);
"""


class StateStore:
    """SQLite-backed persistent state store.

    Thread-safe. Uses WAL mode for concurrent read/write.

    Usage:
        store = StateStore("results/gridmind.db")
        store.save_loop("abc", "my-strategy", "running", ...)
        loop = store.get_loop("abc")
    """

    def __init__(self, db_path: str | Path = "results/gridmind.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _tx(self):
        """Transaction context manager."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self):
        with self._tx() as conn:
            conn.executescript(_SCHEMA)

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # --- Loops ---

    def save_loop(
        self,
        loop_id: str,
        strategy_name: str,
        status: str = "idle",
        current_iteration: int = 0,
        best_iteration: int = 0,
        best_artifact: str = "",
        strategy_content: str = "",
        config_json: str = "{}",
    ):
        now = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO loops (id, strategy_name, strategy_content, status,
                   current_iteration, best_iteration, best_artifact, config_json,
                   created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                   status=excluded.status,
                   current_iteration=excluded.current_iteration,
                   best_iteration=excluded.best_iteration,
                   best_artifact=excluded.best_artifact,
                   updated_at=excluded.updated_at""",
                (loop_id, strategy_name, strategy_content, status,
                 current_iteration, best_iteration, best_artifact, config_json,
                 now, now),
            )

    def update_loop_status(self, loop_id: str, status: str, iteration: int | None = None,
                           best_iteration: int | None = None, best_artifact: str | None = None):
        now = datetime.now(timezone.utc).isoformat()
        updates = ["status=?", "updated_at=?"]
        params: list = [status, now]

        if iteration is not None:
            updates.append("current_iteration=?")
            params.append(iteration)
        if best_iteration is not None:
            updates.append("best_iteration=?")
            params.append(best_iteration)
        if best_artifact is not None:
            updates.append("best_artifact=?")
            params.append(best_artifact)

        params.append(loop_id)
        with self._tx() as conn:
            conn.execute(
                f"UPDATE loops SET {', '.join(updates)} WHERE id=?",
                params,
            )

    def get_loop(self, loop_id: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM loops WHERE id=?", (loop_id,)).fetchone()
        return dict(row) if row else None

    def list_loops(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, strategy_name, status, current_iteration, best_iteration, "
            "created_at, updated_at FROM loops ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Experiments ---

    def save_experiment(
        self,
        loop_id: str,
        iteration: int,
        variables: dict,
        hypothesis: str = "",
        approach: str = "",
        status: str = "pending",
        metrics: dict | None = None,
        artifacts: dict | None = None,
        error: str | None = None,
        duration_seconds: float = 0.0,
    ):
        now = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO experiments
                   (loop_id, iteration, variables_json, hypothesis, approach,
                    status, metrics_json, artifacts_json, error, duration_seconds, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (loop_id, iteration, json.dumps(variables), hypothesis, approach,
                 status, json.dumps(metrics or {}), json.dumps(artifacts or {}),
                 error, duration_seconds, now),
            )

    def get_experiments(self, loop_id: str, limit: int = 0) -> list[dict]:
        conn = self._get_conn()
        query = "SELECT * FROM experiments WHERE loop_id=? ORDER BY iteration"
        if limit > 0:
            query += f" LIMIT {limit}"
        rows = conn.execute(query, (loop_id,)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["variables"] = json.loads(d.pop("variables_json", "{}"))
            d["metrics"] = json.loads(d.pop("metrics_json", "{}"))
            d["artifacts"] = json.loads(d.pop("artifacts_json", "{}"))
            result.append(d)
        return result

    # --- Metrics History ---

    def save_metric(self, loop_id: str, name: str, value: float,
                    iteration: int, direction: str = "higher"):
        now = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO metrics_history
                   (loop_id, name, value, iteration, direction, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (loop_id, name, value, iteration, direction, now),
            )

    def get_metric_history(self, loop_id: str, name: str | None = None) -> list[dict]:
        conn = self._get_conn()
        if name:
            rows = conn.execute(
                "SELECT * FROM metrics_history WHERE loop_id=? AND name=? ORDER BY iteration",
                (loop_id, name),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM metrics_history WHERE loop_id=? ORDER BY iteration",
                (loop_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_metric_values(self, loop_id: str, name: str) -> list[float]:
        """Get raw metric values for statistical analysis."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT value FROM metrics_history WHERE loop_id=? AND name=? ORDER BY iteration",
            (loop_id, name),
        ).fetchall()
        return [r["value"] for r in rows]

    # --- Webhook Events ---

    def save_webhook_event(
        self,
        source: str,
        metrics: dict,
        loop_id: str | None = None,
        experiment_iteration: int | None = None,
        metadata: dict | None = None,
        timestamp: str | None = None,
    ):
        now = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO webhook_events
                   (source, loop_id, experiment_iteration, metrics_json,
                    metadata_json, timestamp, received_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (source, loop_id, experiment_iteration,
                 json.dumps(metrics), json.dumps(metadata),
                 timestamp or now, now),
            )

    def get_webhook_events(
        self,
        source: str | None = None,
        loop_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        conn = self._get_conn()
        conditions = []
        params: list = []
        if source:
            conditions.append("source=?")
            params.append(source)
        if loop_id:
            conditions.append("loop_id=?")
            params.append(loop_id)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = conn.execute(
            f"SELECT * FROM webhook_events {where} ORDER BY received_at DESC LIMIT ?",
            params + [limit],
        ).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            d["metrics"] = json.loads(d.pop("metrics_json", "{}"))
            d["metadata"] = json.loads(d.pop("metadata_json", "null"))
            result.append(d)
        return result

    # --- Bandit State ---

    def save_bandit_state(self, loop_id: str, state: dict):
        now = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO bandit_state (loop_id, state_json, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(loop_id) DO UPDATE SET
                   state_json=excluded.state_json, updated_at=excluded.updated_at""",
                (loop_id, json.dumps(state), now),
            )

    def get_bandit_state(self, loop_id: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT state_json FROM bandit_state WHERE loop_id=?", (loop_id,)
        ).fetchone()
        if row:
            return json.loads(row["state_json"])
        return None

    # --- Aggregate Queries ---

    def get_loop_summary(self, loop_id: str) -> dict | None:
        """Get a full summary of a loop including experiment stats."""
        loop = self.get_loop(loop_id)
        if not loop:
            return None

        conn = self._get_conn()

        # Experiment counts by status
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM experiments WHERE loop_id=? GROUP BY status",
            (loop_id,),
        ).fetchall()
        status_counts = {r["status"]: r["cnt"] for r in rows}

        # Best metrics
        rows = conn.execute(
            """SELECT name, MAX(value) as best_value, direction
               FROM metrics_history WHERE loop_id=? AND direction='higher'
               GROUP BY name
               UNION ALL
               SELECT name, MIN(value) as best_value, direction
               FROM metrics_history WHERE loop_id=? AND direction='lower'
               GROUP BY name""",
            (loop_id, loop_id),
        ).fetchall()
        best_metrics = {r["name"]: {"best_value": r["best_value"], "direction": r["direction"]}
                        for r in rows}

        return {
            **loop,
            "experiments": status_counts,
            "total_experiments": sum(status_counts.values()),
            "best_metrics": best_metrics,
        }
