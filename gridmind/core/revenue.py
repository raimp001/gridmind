"""Revenue impact tracking. The compounding effect across all loops.

Each loop produces learning. But the learning spreads:

    Outreach learns "revenue at risk" gets 3x replies
    -> Call scripts lead with $ amounts
    -> Ad creative uses case study numbers
    -> Blog intros highlight ROI proof
    -> Everything converts better

This module tracks the cumulative revenue impact of experiment improvements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RevenueModel:
    """Maps metric improvements to estimated revenue impact.

    Configure per domain: "if reply_rate improves by X%, that means Y more
    pipeline dollars per month."
    """

    domain: str
    monthly_volume: float = 0      # emails sent, visitors, calls made, etc.
    baseline_rate: float = 0       # current conversion/response rate
    revenue_per_conversion: float = 0  # $ value of each conversion
    metric_name: str = ""          # which metric drives revenue

    def impact(self, new_rate: float) -> dict:
        """Calculate revenue impact of a rate improvement."""
        old_conversions = self.monthly_volume * self.baseline_rate
        new_conversions = self.monthly_volume * new_rate
        delta_conversions = new_conversions - old_conversions
        delta_revenue = delta_conversions * self.revenue_per_conversion

        return {
            "domain": self.domain,
            "metric": self.metric_name,
            "baseline_rate": self.baseline_rate,
            "new_rate": new_rate,
            "improvement_pct": (
                (new_rate - self.baseline_rate) / max(self.baseline_rate, 0.001) * 100
            ),
            "old_conversions_monthly": old_conversions,
            "new_conversions_monthly": new_conversions,
            "delta_conversions_monthly": delta_conversions,
            "delta_revenue_monthly": delta_revenue,
            "delta_revenue_annual": delta_revenue * 12,
        }


@dataclass
class CampaignRevenue:
    """Aggregate revenue impact across all loops in a campaign."""

    models: list[RevenueModel] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)

    def add_model(self, model: RevenueModel):
        self.models.append(model)

    def calculate(self, current_metrics: dict[str, dict]) -> dict:
        """Calculate total revenue impact from current best metrics.

        Args:
            current_metrics: {domain: {metric_name: {best_value: x, ...}}}
        """
        total_monthly = 0.0
        total_annual = 0.0
        by_domain: list[dict] = []

        for model in self.models:
            domain_metrics = current_metrics.get(model.domain, {})
            metric_info = domain_metrics.get(model.metric_name, {})
            new_rate = metric_info.get("best_value", model.baseline_rate)

            impact = model.impact(new_rate)
            total_monthly += impact["delta_revenue_monthly"]
            total_annual += impact["delta_revenue_annual"]
            by_domain.append(impact)

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_delta_revenue_monthly": total_monthly,
            "total_delta_revenue_annual": total_annual,
            "by_domain": by_domain,
        }
        self.snapshots.append(snapshot)

        return snapshot

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "revenue_impact.json").write_text(
            json.dumps(self.snapshots, indent=2)
        )


# --- Preset revenue models for common scenarios ---

def cold_outreach_revenue(
    monthly_sends: int = 9000,
    baseline_reply_rate: float = 0.01,
    close_rate: float = 0.1,
    deal_size: float = 50000,
) -> RevenueModel:
    """Revenue model for cold email outreach.

    Eric Siu's example:
    - 9,000 sends/month
    - 1% -> 2% reply rate = 90 -> 180 interested replies
    - At typical close rates = $500K+ additional pipeline
    """
    return RevenueModel(
        domain="sales_pipeline",
        monthly_volume=monthly_sends,
        baseline_rate=baseline_reply_rate,
        revenue_per_conversion=close_rate * deal_size,
        metric_name="reply_rate",
    )


def ad_creative_revenue(
    monthly_impressions: int = 500000,
    baseline_ctr: float = 0.012,
    conversion_rate: float = 0.03,
    revenue_per_conversion: float = 200,
) -> RevenueModel:
    """Revenue model for ad creative optimization."""
    return RevenueModel(
        domain="ad_creative",
        monthly_volume=monthly_impressions,
        baseline_rate=baseline_ctr,
        revenue_per_conversion=conversion_rate * revenue_per_conversion,
        metric_name="click_rate",
    )


def landing_page_revenue(
    monthly_visitors: int = 50000,
    baseline_conversion: float = 0.03,
    revenue_per_signup: float = 500,
) -> RevenueModel:
    """Revenue model for landing page optimization."""
    return RevenueModel(
        domain="landing_page",
        monthly_volume=monthly_visitors,
        baseline_rate=baseline_conversion,
        revenue_per_signup=revenue_per_signup,
        revenue_per_conversion=revenue_per_signup,
        metric_name="conversion_rate",
    )


def lead_gen_revenue(
    monthly_visitors: int = 20000,
    baseline_conversion: float = 0.15,
    lead_to_customer_rate: float = 0.05,
    customer_ltv: float = 10000,
) -> RevenueModel:
    """Revenue model for lead generation."""
    return RevenueModel(
        domain="lead_gen",
        monthly_volume=monthly_visitors,
        baseline_rate=baseline_conversion,
        revenue_per_conversion=lead_to_customer_rate * customer_ltv,
        metric_name="conversion_rate",
    )
