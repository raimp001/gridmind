"""Integration connectors for real-world metric ingestion.

Each connector polls or receives data from an external service
and pushes metrics into GridMind via the webhook/store interface.
"""

from gridmind.integrations.base import Integration, IntegrationConfig
from gridmind.integrations.sendgrid import SendGridIntegration
from gridmind.integrations.analytics import GoogleAnalyticsIntegration
from gridmind.integrations.mixpanel import MixpanelIntegration

__all__ = [
    "Integration",
    "IntegrationConfig",
    "SendGridIntegration",
    "GoogleAnalyticsIntegration",
    "MixpanelIntegration",
]
