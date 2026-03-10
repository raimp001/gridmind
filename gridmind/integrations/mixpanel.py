"""Mixpanel integration for product analytics metrics.

Fetches event counts, funnel conversion rates, and retention data
from the Mixpanel Data Export API.

Requires: MIXPANEL_PROJECT_ID and MIXPANEL_API_SECRET (service account).
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone, timedelta

from gridmind.integrations.base import Integration, IntegrationConfig, MetricEvent

logger = logging.getLogger(__name__)


class MixpanelIntegration(Integration):
    """Fetch product analytics from Mixpanel.

    Metrics produced:
    - event_count: total events for the tracked event
    - unique_users: unique users who triggered the event
    - funnel_conversion: conversion rate through a funnel
    - retention_rate: day-N retention rate
    """

    name = "mixpanel"

    def __init__(self, config: IntegrationConfig | None = None):
        config = config or IntegrationConfig()
        config.base_url = config.base_url or "https://data.mixpanel.com/api/2.0"
        super().__init__(config)
        self.project_id = config.extra.get(
            "project_id", os.environ.get("MIXPANEL_PROJECT_ID", "")
        )
        self._api_secret = config.api_secret or os.environ.get("MIXPANEL_API_SECRET", "")

    def validate_connection(self) -> bool:
        if not self._api_secret:
            logger.error("Mixpanel API secret not configured")
            return False
        try:
            # Test with a simple query
            now = datetime.now(timezone.utc)
            self._mp_request(
                "GET",
                "/events",
                params={
                    "event": '["$pageview"]',
                    "type": "general",
                    "unit": "day",
                    "from_date": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "to_date": now.strftime("%Y-%m-%d"),
                },
            )
            return True
        except Exception as e:
            logger.error("Mixpanel connection failed: %s", e)
            return False

    def fetch_metrics(
        self,
        event_name: str = "$pageview",
        start_date: str | None = None,
        end_date: str | None = None,
        loop_id: str | None = None,
        **kwargs,
    ) -> list[MetricEvent]:
        """Fetch event metrics from Mixpanel.

        Args:
            event_name: Mixpanel event name (e.g., "Sign Up", "Purchase").
            start_date: Start date (YYYY-MM-DD). Defaults to 7 days ago.
            end_date: End date. Defaults to today.
            loop_id: GridMind loop to link metrics to.
        """
        now = datetime.now(timezone.utc)
        if not start_date:
            start_date = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = now.strftime("%Y-%m-%d")

        data = self._mp_request(
            "GET",
            "/events",
            params={
                "event": f'["{event_name}"]',
                "type": "general",
                "unit": "day",
                "from_date": start_date,
                "to_date": end_date,
            },
        )

        return self._parse_event_response(data, event_name, loop_id, start_date, end_date)

    def fetch_funnel(
        self,
        funnel_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        loop_id: str | None = None,
    ) -> list[MetricEvent]:
        """Fetch funnel conversion metrics.

        Args:
            funnel_id: Mixpanel funnel ID.
            start_date: Start date. Defaults to 30 days ago.
            end_date: End date. Defaults to today.
            loop_id: GridMind loop to link metrics to.
        """
        now = datetime.now(timezone.utc)
        if not start_date:
            start_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = now.strftime("%Y-%m-%d")

        data = self._mp_request(
            "GET",
            f"/funnels/{funnel_id}",
            params={
                "from_date": start_date,
                "to_date": end_date,
            },
        )

        return self._parse_funnel_response(data, funnel_id, loop_id, start_date, end_date)

    def _parse_event_response(
        self, data: dict, event_name: str, loop_id: str | None,
        start_date: str, end_date: str,
    ) -> list[MetricEvent]:
        events = []
        values = data.get("data", {}).get("values", {}).get(event_name, {})

        # Aggregate across days
        total_count = sum(values.values()) if values else 0

        if total_count > 0:
            events.append(MetricEvent(
                source="mixpanel",
                metrics={
                    "event_count": float(total_count),
                    "daily_average": total_count / max(len(values), 1),
                },
                loop_id=loop_id,
                metadata={
                    "event_name": event_name,
                    "date_range": f"{start_date} to {end_date}",
                    "days": len(values),
                    "project_id": self.project_id,
                },
            ))

        return events

    def _parse_funnel_response(
        self, data: dict, funnel_id: str, loop_id: str | None,
        start_date: str, end_date: str,
    ) -> list[MetricEvent]:
        events = []
        meta = data.get("meta", {})
        funnel_data = data.get("data", {})

        # Extract step-by-step conversion
        steps = []
        for date_data in funnel_data.values():
            if isinstance(date_data, dict) and "steps" in date_data:
                steps = date_data["steps"]
                break

        if steps and len(steps) >= 2:
            first_step_count = steps[0].get("count", 0)
            last_step_count = steps[-1].get("count", 0)

            if first_step_count > 0:
                overall_conversion = last_step_count / first_step_count
            else:
                overall_conversion = 0.0

            metrics = {
                "funnel_conversion": overall_conversion,
                "funnel_entries": float(first_step_count),
                "funnel_completions": float(last_step_count),
            }

            # Step-by-step drop-off
            for i, step in enumerate(steps):
                step_count = step.get("count", 0)
                if i > 0 and steps[i - 1].get("count", 0) > 0:
                    step_rate = step_count / steps[i - 1]["count"]
                    metrics[f"step_{i + 1}_rate"] = step_rate

            events.append(MetricEvent(
                source="mixpanel",
                metrics=metrics,
                loop_id=loop_id,
                metadata={
                    "funnel_id": funnel_id,
                    "funnel_name": meta.get("name", ""),
                    "date_range": f"{start_date} to {end_date}",
                    "steps": len(steps),
                },
            ))

        return events

    def _mp_request(self, method: str, path: str, params: dict | None = None) -> dict:
        # Build URL with query params
        url = f"{self.config.base_url}{path}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"

        # Mixpanel uses HTTP Basic Auth with API secret as username
        auth_str = base64.b64encode(f"{self._api_secret}:".encode()).decode()

        return self._make_request(
            method, url,
            headers={
                "Authorization": f"Basic {auth_str}",
                "Accept": "application/json",
            },
        )
