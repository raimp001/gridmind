"""Base class for all integration connectors."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for an integration connector."""

    api_key: str = ""
    api_secret: str = ""
    base_url: str = ""
    poll_interval_seconds: int = 300  # 5 minutes
    timeout_seconds: float = 30.0
    extra: dict = field(default_factory=dict)


@dataclass
class MetricEvent:
    """A metric event from an external source."""

    source: str
    metrics: dict[str, float]
    loop_id: str | None = None
    experiment_iteration: int | None = None
    metadata: dict | None = None
    timestamp: str | None = None


class Integration(ABC):
    """Base class for integration connectors.

    Each integration knows how to:
    1. Connect to an external service
    2. Fetch relevant metrics
    3. Return them as MetricEvents for the loop
    """

    name: str = "base"

    def __init__(self, config: IntegrationConfig):
        self.config = config

    @abstractmethod
    def fetch_metrics(self, **kwargs) -> list[MetricEvent]:
        """Fetch current metrics from the external service.

        Returns a list of MetricEvent objects ready to push into the loop.
        """
        ...

    @abstractmethod
    def validate_connection(self) -> bool:
        """Test that the integration is properly configured and can connect."""
        ...

    def _make_request(self, method: str, url: str, **kwargs) -> dict:
        """Make an HTTP request with retries. Uses urllib to avoid dependencies."""
        import json
        import urllib.request
        import urllib.error

        headers = kwargs.pop("headers", {})
        data = kwargs.pop("data", None)

        if data and isinstance(data, dict):
            data = json.dumps(data).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            logger.error("%s request to %s failed: %s %s", method, url, e.code, body[:200])
            raise
        except urllib.error.URLError as e:
            logger.error("%s request to %s failed: %s", method, url, e.reason)
            raise
