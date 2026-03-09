"""Tests for revenue impact tracking."""

from gridmind.core.revenue import (
    RevenueModel,
    CampaignRevenue,
    cold_outreach_revenue,
    ad_creative_revenue,
    landing_page_revenue,
    lead_gen_revenue,
)


def test_revenue_model_basic():
    model = RevenueModel(
        domain="sales_pipeline",
        monthly_volume=9000,
        baseline_rate=0.01,
        revenue_per_conversion=5000,
        metric_name="reply_rate",
    )
    impact = model.impact(0.02)
    assert impact["delta_conversions_monthly"] == 90  # 9000 * 0.01
    assert impact["delta_revenue_monthly"] == 450000  # 90 * 5000
    assert impact["improvement_pct"] == 100.0


def test_revenue_model_no_improvement():
    model = RevenueModel(
        domain="test",
        monthly_volume=1000,
        baseline_rate=0.05,
        revenue_per_conversion=100,
        metric_name="rate",
    )
    impact = model.impact(0.05)
    assert impact["delta_conversions_monthly"] == 0
    assert impact["delta_revenue_monthly"] == 0


def test_cold_outreach_preset():
    model = cold_outreach_revenue()
    assert model.domain == "sales_pipeline"
    assert model.monthly_volume == 9000
    # 1% -> 2% = $500K+ pipeline (from Eric Siu's example)
    impact = model.impact(0.02)
    assert impact["delta_revenue_monthly"] > 0


def test_campaign_revenue_aggregate():
    campaign = CampaignRevenue()
    campaign.add_model(cold_outreach_revenue(
        monthly_sends=9000,
        baseline_reply_rate=0.01,
        close_rate=0.1,
        deal_size=50000,
    ))
    campaign.add_model(ad_creative_revenue(
        monthly_impressions=500000,
        baseline_ctr=0.012,
    ))

    metrics = {
        "sales_pipeline": {"reply_rate": {"best_value": 0.02}},
        "ad_creative": {"click_rate": {"best_value": 0.018}},
    }

    result = campaign.calculate(metrics)
    assert result["total_delta_revenue_monthly"] > 0
    assert result["total_delta_revenue_annual"] > 0
    assert len(result["by_domain"]) == 2


def test_campaign_revenue_save(tmp_path):
    campaign = CampaignRevenue()
    campaign.add_model(cold_outreach_revenue())
    campaign.calculate({"sales_pipeline": {"reply_rate": {"best_value": 0.02}}})

    campaign.save(tmp_path / "revenue")
    assert (tmp_path / "revenue" / "revenue_impact.json").exists()
