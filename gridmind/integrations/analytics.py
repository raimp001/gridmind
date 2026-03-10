"""Google Analytics 4 integration for web metrics.

Fetches page views, conversion rates, bounce rates, and session metrics
from the GA4 Data API. Maps them to GridMind metrics.

Requires: GA4_PROPERTY_ID and a service account key or OAuth token.

Note: Uses the GA4 Data API (not Universal Analytics).
API docs: https://developers.google.com/analytics/devguides/reporting/data/v1
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta

from gridmind.integrations.base import Integration, IntegrationConfig, MetricEvent

logger = logging.getLogger(__name__)


class GoogleAnalyticsIntegration(Integration):
    """Fetch web analytics metrics from Google Analytics 4.

    Metrics produced:
    - sessions: total sessions
    - page_views: total page views
    - bounce_rate: % of single-page sessions
    - avg_session_duration: average session length in seconds
    - conversion_rate: conversions / sessions
    - new_user_rate: new users / total users
    """

    name = "google_analytics"

    def __init__(self, config: IntegrationConfig | None = None):
        config = config or IntegrationConfig()
        config.base_url = config.base_url or "https://analyticsdata.googleapis.com/v1beta"
        super().__init__(config)
        self.property_id = config.extra.get(
            "property_id", os.environ.get("GA4_PROPERTY_ID", "")
        )
        self._access_token = config.api_key or os.environ.get("GA4_ACCESS_TOKEN", "")

    def validate_connection(self) -> bool:
        if not self.property_id:
            logger.error("GA4 property ID not configured")
            return False
        if not self._access_token:
            logger.error("GA4 access token not configured")
            return False
        try:
            self._ga_request({
                "dateRanges": [{"startDate": "yesterday", "endDate": "today"}],
                "metrics": [{"name": "sessions"}],
            })
            return True
        except Exception as e:
            logger.error("GA4 connection failed: %s", e)
            return False

    def fetch_metrics(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        page_path: str | None = None,
        loop_id: str | None = None,
        **kwargs,
    ) -> list[MetricEvent]:
        """Fetch analytics metrics from GA4.

        Args:
            start_date: Start date (YYYY-MM-DD or "7daysAgo"). Defaults to "7daysAgo".
            end_date: End date. Defaults to "today".
            page_path: Filter to a specific page path.
            loop_id: GridMind loop to link metrics to.
        """
        start_date = start_date or "7daysAgo"
        end_date = end_date or "today"

        request_body = {
            "dateRanges": [{"startDate": start_date, "endDate": end_date}],
            "metrics": [
                {"name": "sessions"},
                {"name": "screenPageViews"},
                {"name": "bounceRate"},
                {"name": "averageSessionDuration"},
                {"name": "conversions"},
                {"name": "newUsers"},
                {"name": "totalUsers"},
            ],
        }

        if page_path:
            request_body["dimensionFilter"] = {
                "filter": {
                    "fieldName": "pagePath",
                    "stringFilter": {"value": page_path, "matchType": "EXACT"},
                }
            }

        data = self._ga_request(request_body)
        return self._parse_response(data, loop_id, start_date, end_date, page_path)

    def fetch_page_comparison(
        self,
        page_paths: list[str],
        start_date: str = "7daysAgo",
        end_date: str = "today",
    ) -> dict[str, dict[str, float]]:
        """Compare metrics across different pages (useful for landing page A/B tests)."""
        result = {}
        for path in page_paths:
            events = self.fetch_metrics(
                start_date=start_date, end_date=end_date, page_path=path,
            )
            if events:
                result[path] = events[0].metrics
        return result

    def _parse_response(
        self, data: dict, loop_id: str | None,
        start_date: str, end_date: str, page_path: str | None,
    ) -> list[MetricEvent]:
        events = []
        for row in data.get("rows", []):
            values = row.get("metricValues", [])
            if len(values) < 7:
                continue

            sessions = float(values[0].get("value", 0))
            page_views = float(values[1].get("value", 0))
            bounce_rate = float(values[2].get("value", 0))
            avg_duration = float(values[3].get("value", 0))
            conversions = float(values[4].get("value", 0))
            new_users = float(values[5].get("value", 0))
            total_users = float(values[6].get("value", 0))

            metrics = {
                "sessions": sessions,
                "page_views": page_views,
                "bounce_rate": bounce_rate,
                "avg_session_duration": avg_duration,
            }

            if sessions > 0:
                metrics["conversion_rate"] = conversions / sessions

            if total_users > 0:
                metrics["new_user_rate"] = new_users / total_users

            events.append(MetricEvent(
                source="google_analytics",
                metrics=metrics,
                loop_id=loop_id,
                metadata={
                    "date_range": f"{start_date} to {end_date}",
                    "page_path": page_path,
                    "property_id": self.property_id,
                },
            ))

        return events

    def _ga_request(self, body: dict) -> dict:
        url = f"{self.config.base_url}/properties/{self.property_id}:runReport"
        return self._make_request(
            "POST", url,
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
            },
            data=body,
        )
