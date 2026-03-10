"""SendGrid integration for email campaign metrics.

Fetches open rates, click rates, bounce rates, and spam reports
from the SendGrid Stats API. Maps them to GridMind metrics.

Requires: SENDGRID_API_KEY environment variable or config.api_key.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta

from gridmind.integrations.base import Integration, IntegrationConfig, MetricEvent

logger = logging.getLogger(__name__)


class SendGridIntegration(Integration):
    """Fetch email campaign metrics from SendGrid.

    Metrics produced:
    - open_rate: unique opens / delivered
    - click_rate: unique clicks / delivered
    - reply_rate: estimated from click-through (SendGrid doesn't track replies natively)
    - bounce_rate: bounces / requests
    - spam_rate: spam reports / delivered
    """

    name = "sendgrid"

    def __init__(self, config: IntegrationConfig | None = None):
        config = config or IntegrationConfig()
        config.api_key = config.api_key or os.environ.get("SENDGRID_API_KEY", "")
        config.base_url = config.base_url or "https://api.sendgrid.com/v3"
        super().__init__(config)

    def validate_connection(self) -> bool:
        if not self.config.api_key:
            logger.error("SendGrid API key not configured")
            return False
        try:
            self._sg_request("GET", "/scopes")
            return True
        except Exception as e:
            logger.error("SendGrid connection failed: %s", e)
            return False

    def fetch_metrics(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        category: str | None = None,
        loop_id: str | None = None,
        **kwargs,
    ) -> list[MetricEvent]:
        """Fetch email stats from SendGrid.

        Args:
            start_date: Start date (YYYY-MM-DD). Defaults to yesterday.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            category: SendGrid category to filter by.
            loop_id: GridMind loop to link metrics to.
        """
        now = datetime.now(timezone.utc)
        if not start_date:
            start_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = now.strftime("%Y-%m-%d")

        url = f"/stats?start_date={start_date}&end_date={end_date}"
        if category:
            url = f"/categories/stats?start_date={start_date}&end_date={end_date}&categories={category}"

        data = self._sg_request("GET", url)

        events = []
        for day_stats in data:
            date = day_stats.get("date", start_date)
            for stat_group in day_stats.get("stats", []):
                raw = stat_group.get("metrics", {})
                metrics = self._compute_rates(raw)
                if metrics:
                    events.append(MetricEvent(
                        source="sendgrid",
                        metrics=metrics,
                        loop_id=loop_id,
                        metadata={
                            "date": date,
                            "category": category,
                            "raw": raw,
                        },
                        timestamp=f"{date}T00:00:00Z",
                    ))

        return events

    def fetch_category_comparison(
        self,
        categories: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compare metrics across SendGrid categories (useful for A/B testing).

        Returns {category: {metric: value}}.
        """
        result = {}
        for cat in categories:
            events = self.fetch_metrics(
                start_date=start_date, end_date=end_date, category=cat
            )
            if events:
                # Average across days
                all_metrics: dict[str, list[float]] = {}
                for e in events:
                    for k, v in e.metrics.items():
                        all_metrics.setdefault(k, []).append(v)
                result[cat] = {
                    k: sum(v) / len(v) for k, v in all_metrics.items()
                }
        return result

    def _compute_rates(self, raw: dict) -> dict[str, float]:
        """Compute rate metrics from raw SendGrid counts."""
        requests = raw.get("requests", 0)
        delivered = raw.get("delivered", 0)
        opens = raw.get("unique_opens", 0)
        clicks = raw.get("unique_clicks", 0)
        bounces = raw.get("bounces", 0)
        spam = raw.get("spam_reports", 0)

        if delivered <= 0:
            return {}

        return {
            "open_rate": opens / delivered,
            "click_rate": clicks / delivered,
            "bounce_rate": bounces / max(requests, 1),
            "spam_rate": spam / delivered,
            "delivered_count": float(delivered),
        }

    def _sg_request(self, method: str, path: str) -> dict | list:
        url = f"{self.config.base_url}{path}"
        return self._make_request(
            method, url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Accept": "application/json",
            },
        )
