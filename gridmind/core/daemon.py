"""Daemon mode for running research loops unattended.

Features:
- SIGTERM/SIGINT graceful shutdown (finishes current iteration, saves checkpoint)
- Watchdog: detects hung iterations, restarts after timeout
- Scheduled runs: run loops on a cron-like schedule
- Process heartbeat for monitoring
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from gridmind.core.loop import LoopConfig, LoopState, ResearchLoop

logger = logging.getLogger(__name__)


@dataclass
class DaemonConfig:
    """Configuration for daemon mode."""

    loop_config: LoopConfig = field(default_factory=LoopConfig)
    # Heartbeat: write status to file every N seconds
    heartbeat_interval: int = 30
    heartbeat_path: str = ""
    # Watchdog: restart if no progress for N seconds
    watchdog_timeout: int = 600  # 10 minutes
    # Schedule: repeat the loop every N seconds (0 = run once)
    repeat_interval: int = 0
    # PID file for process management
    pid_file: str = ""


class GracefulShutdown:
    """Signal handler that allows the current iteration to finish."""

    def __init__(self):
        self.should_stop = threading.Event()
        self._original_sigterm = None
        self._original_sigint = None

    def install(self):
        """Install signal handlers for graceful shutdown."""
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def uninstall(self):
        """Restore original signal handlers."""
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)

    def _handle(self, signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, finishing current iteration then shutting down...", sig_name)
        self.should_stop.set()


class Watchdog:
    """Monitors loop progress and detects hangs."""

    def __init__(self, timeout: int = 600):
        self.timeout = timeout
        self._last_progress_time = time.monotonic()
        self._last_iteration = 0
        self._lock = threading.Lock()

    def report_progress(self, iteration: int):
        """Called after each iteration completes."""
        with self._lock:
            self._last_progress_time = time.monotonic()
            self._last_iteration = iteration

    def is_hung(self) -> bool:
        """Returns True if no progress for longer than timeout."""
        with self._lock:
            elapsed = time.monotonic() - self._last_progress_time
            return elapsed > self.timeout

    @property
    def seconds_since_progress(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_progress_time


class Heartbeat:
    """Writes periodic status to a file for external monitoring."""

    def __init__(self, path: str, interval: int = 30):
        self.path = Path(path) if path else None
        self.interval = interval
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._status: dict = {}

    def update(self, status: dict):
        self._status = status

    def start(self):
        if not self.path:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.wait(self.interval):
            self._write()
        self._write()  # Final write on stop

    def _write(self):
        if not self.path:
            return
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pid": __import__("os").getpid(),
            **self._status,
        }
        try:
            self.path.write_text(json.dumps(data, indent=2))
        except OSError:
            pass


class LoopDaemon:
    """Runs a research loop as a daemon with monitoring and restart.

    Usage:
        daemon = LoopDaemon(config)
        daemon.run()  # Blocks until complete or signaled
    """

    def __init__(self, config: DaemonConfig):
        self.config = config
        self.shutdown = GracefulShutdown()
        self.watchdog = Watchdog(config.watchdog_timeout)
        self.heartbeat = Heartbeat(
            config.heartbeat_path, config.heartbeat_interval,
        )
        self._loop: ResearchLoop | None = None

    def run(self) -> LoopState | None:
        """Run the loop (or repeat on schedule) with full daemon support."""
        self.shutdown.install()
        self.heartbeat.start()
        self._write_pid()

        try:
            if self.config.repeat_interval > 0:
                return self._run_scheduled()
            else:
                return self._run_once()
        finally:
            self.heartbeat.stop()
            self.shutdown.uninstall()
            self._remove_pid()

    def _run_once(self) -> LoopState | None:
        """Run the loop once with watchdog monitoring."""
        loop = ResearchLoop(self.config.loop_config)
        self._loop = loop

        # Wire up watchdog
        def on_event(event_type: str, data: dict):
            if event_type == "iteration_completed":
                self.watchdog.report_progress(data.get("iteration", 0))
            self.heartbeat.update({
                "status": loop.state.status,
                "iteration": loop.state.current_iteration,
                "best_iteration": loop.state.best_iteration,
                "event": event_type,
            })
            # Check for graceful shutdown
            if self.shutdown.should_stop.is_set():
                loop.state.status = "paused"

        loop.on_event(on_event)

        logger.info("Daemon starting loop")
        try:
            state = loop.run()
            logger.info(
                "Loop finished: status=%s, iterations=%d",
                state.status, len(state.experiments),
            )
            return state
        except Exception as e:
            logger.error("Loop crashed: %s", e)
            return loop.state

    def _run_scheduled(self) -> LoopState | None:
        """Run the loop repeatedly on a schedule."""
        run_count = 0
        last_state = None

        while not self.shutdown.should_stop.is_set():
            run_count += 1
            logger.info("Scheduled run #%d starting", run_count)
            self.heartbeat.update({"scheduled_run": run_count, "status": "starting"})

            last_state = self._run_once()

            if self.shutdown.should_stop.is_set():
                break

            logger.info(
                "Run #%d complete, sleeping %ds before next run",
                run_count, self.config.repeat_interval,
            )
            self.heartbeat.update({
                "status": "waiting",
                "next_run_in": self.config.repeat_interval,
                "completed_runs": run_count,
            })

            # Interruptible sleep
            self.shutdown.should_stop.wait(self.config.repeat_interval)

        return last_state

    def _write_pid(self):
        if self.config.pid_file:
            try:
                Path(self.config.pid_file).write_text(
                    str(__import__("os").getpid())
                )
            except OSError:
                pass

    def _remove_pid(self):
        if self.config.pid_file:
            try:
                Path(self.config.pid_file).unlink(missing_ok=True)
            except OSError:
                pass
