"""Domain adapter registry. Domains customize how experiments are evaluated."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class DomainAdapter(Protocol):
    """Interface for domain-specific experiment evaluation."""

    name: str
    description: str

    def get_evaluation_prompt(self, artifact: str, variables: dict) -> str:
        """Return a domain-specific evaluation prompt."""
        ...

    def get_strategy_template(self) -> str:
        """Return a starter strategy .md template for this domain."""
        ...


class DomainRegistry:
    """Registry of available domain adapters."""

    _domains: dict[str, DomainAdapter] = {}

    @classmethod
    def register(cls, adapter: DomainAdapter):
        cls._domains[adapter.name] = adapter

    @classmethod
    def get(cls, name: str) -> DomainAdapter | None:
        return cls._domains.get(name)

    @classmethod
    def list_domains(cls) -> list[str]:
        return list(cls._domains.keys())

    @classmethod
    def all(cls) -> dict[str, DomainAdapter]:
        return dict(cls._domains)


# --- Built-in domain adapters ---


@dataclass
class SalesPipelineAdapter:
    name: str = "sales_pipeline"
    description: str = "Optimize sales outreach sequences, scripts, and cadences"

    def get_evaluation_prompt(self, artifact: str, variables: dict) -> str:
        return (
            "Evaluate this sales outreach artifact. Score realistically (0.0-1.0):\n"
            "- reply_rate: likelihood of getting a response\n"
            "- meeting_rate: likelihood of booking a meeting\n"
            "- personalization: how tailored it feels\n"
            "- clarity: how clear the value prop is\n"
            f"\n## Artifact\n{artifact}\n"
            f"\n## Variables\n{variables}\n"
            "\nReturn JSON: {\"reply_rate\": x, \"meeting_rate\": x, \"personalization\": x, \"clarity\": x}"
        )

    def get_strategy_template(self) -> str:
        return '''---
name: sales-outreach-optimizer
domain: sales_pipeline
objective: Maximize meeting booking rate from cold outreach
metrics:
  - name: meeting_rate
    direction: higher
    baseline: 0.03
  - name: reply_rate
    direction: higher
    baseline: 0.08
  - name: personalization
    direction: higher
    baseline: 0.4
constraints:
  - Keep emails under 150 words
  - No clickbait or misleading subject lines
  - Must include clear value proposition
  - Professional tone appropriate for B2B
variables:
  subject_style: [question, statistic, mutual_connection, pain_point, curiosity_gap]
  opener: [compliment, observation, shared_experience, direct_pitch, story]
  value_prop: [roi_focused, time_saving, competitive_advantage, social_proof, risk_reduction]
  cta: [soft_ask, specific_time, value_offer, question, calendar_link]
  length: [ultra_short, short, medium]
  tone: [casual, professional, consultative, urgent]
max_iterations: 100
time_budget_seconds: 120
---

# Experiment Instructions

Generate a cold outreach email using the assigned variable combination.

The email should:
1. Hook attention with the subject line (using the subject_style)
2. Open with the chosen opener approach
3. Present the value proposition clearly
4. End with the specified CTA style

Target persona: VP of Engineering at mid-market SaaS companies (200-1000 employees).
Product: AI-powered code review tool that reduces PR cycle time by 40%.

Output the complete email including subject line.
'''


@dataclass
class AdCreativeAdapter:
    name: str = "ad_creative"
    description: str = "Optimize ad copy, headlines, and creative variations"

    def get_evaluation_prompt(self, artifact: str, variables: dict) -> str:
        return (
            "Evaluate this ad creative. Score realistically (0.0-1.0):\n"
            "- click_rate: likelihood of getting a click\n"
            "- conversion_intent: how likely it drives action\n"
            "- brand_alignment: professional, on-brand feel\n"
            "- scroll_stop: how attention-grabbing it is\n"
            f"\n## Artifact\n{artifact}\n"
            f"\n## Variables\n{variables}\n"
            "\nReturn JSON: {\"click_rate\": x, \"conversion_intent\": x, \"brand_alignment\": x, \"scroll_stop\": x}"
        )

    def get_strategy_template(self) -> str:
        return '''---
name: ad-creative-optimizer
domain: ad_creative
objective: Maximize click-through rate on paid social ads
metrics:
  - name: click_rate
    direction: higher
    baseline: 0.012
  - name: conversion_intent
    direction: higher
    baseline: 0.3
  - name: scroll_stop
    direction: higher
    baseline: 0.4
constraints:
  - Headlines under 40 characters
  - Body copy under 125 characters
  - Must include a clear CTA
  - No misleading claims
variables:
  headline_style: [question, number, how_to, bold_claim, social_proof]
  emotion: [fomo, curiosity, aspiration, pain, relief]
  cta_type: [learn_more, get_started, try_free, see_demo, claim_offer]
  format: [single_image, carousel_concept, video_script_hook]
  hook: [statistic, story, question, contradiction, before_after]
max_iterations: 100
time_budget_seconds: 90
---

# Experiment Instructions

Create an ad creative variation for a LinkedIn/Meta paid campaign.

Product: [Define your product here]
Target audience: [Define your audience here]

Generate:
1. Headline (under 40 chars)
2. Primary text / body copy (under 125 chars)
3. CTA button text
4. Visual direction (brief description of imagery)
5. Hook strategy explanation
'''


@dataclass
class LeadGenAdapter:
    name: str = "lead_gen"
    description: str = "Optimize lead generation: landing pages, forms, lead magnets"

    def get_evaluation_prompt(self, artifact: str, variables: dict) -> str:
        return (
            "Evaluate this lead generation artifact. Score realistically (0.0-1.0):\n"
            "- conversion_rate: likelihood of capturing a lead\n"
            "- lead_quality: how qualified the lead would be\n"
            "- perceived_value: how valuable the offer seems\n"
            "- friction: how much friction in the process (lower is better)\n"
            f"\n## Artifact\n{artifact}\n"
            f"\n## Variables\n{variables}\n"
            "\nReturn JSON: {\"conversion_rate\": x, \"lead_quality\": x, \"perceived_value\": x, \"friction\": x}"
        )

    def get_strategy_template(self) -> str:
        return '''---
name: lead-magnet-optimizer
domain: lead_gen
objective: Maximize lead capture conversion rate
metrics:
  - name: conversion_rate
    direction: higher
    baseline: 0.15
  - name: lead_quality
    direction: higher
    baseline: 0.4
  - name: perceived_value
    direction: higher
    baseline: 0.5
  - name: friction
    direction: lower
    baseline: 0.6
constraints:
  - Headline must be immediately clear
  - Form should ask for minimal info
  - Must deliver genuine value
  - Mobile-friendly design descriptions
variables:
  lead_magnet_type: [checklist, template, calculator, mini_course, swipe_file, report]
  headline_formula: [how_to, number_list, question, challenge, secret]
  social_proof: [testimonials, user_count, logos, case_study, none]
  form_fields: [email_only, email_name, email_name_company, email_name_role]
  urgency: [none, limited_time, limited_spots, countdown, exclusive]
max_iterations: 100
time_budget_seconds: 120
---

# Experiment Instructions

Design a lead capture landing page concept.

Product/Service: [Define here]
Target audience: [Define here]

Generate:
1. Page headline and subheadline
2. Lead magnet title and description
3. Form layout (fields and CTA button text)
4. Above-the-fold copy
5. Social proof element
6. Urgency mechanism (if any)
'''


@dataclass
class ClientOnboardingAdapter:
    name: str = "client_onboarding"
    description: str = "Optimize client onboarding flows, welcome sequences, activation"

    def get_evaluation_prompt(self, artifact: str, variables: dict) -> str:
        return (
            "Evaluate this onboarding artifact. Score realistically (0.0-1.0):\n"
            "- activation_rate: likelihood user completes key action\n"
            "- time_to_value: how quickly user gets value (lower is better)\n"
            "- clarity: how clear the next steps are\n"
            "- engagement: how engaging/motivating the experience is\n"
            f"\n## Artifact\n{artifact}\n"
            f"\n## Variables\n{variables}\n"
            "\nReturn JSON: {\"activation_rate\": x, \"time_to_value\": x, \"clarity\": x, \"engagement\": x}"
        )

    def get_strategy_template(self) -> str:
        return '''---
name: onboarding-flow-optimizer
domain: client_onboarding
objective: Maximize 7-day activation rate for new users
metrics:
  - name: activation_rate
    direction: higher
    baseline: 0.25
  - name: time_to_value
    direction: lower
    baseline: 0.7
  - name: clarity
    direction: higher
    baseline: 0.5
  - name: engagement
    direction: higher
    baseline: 0.4
constraints:
  - Must reach "aha moment" within 3 steps
  - No overwhelming with features
  - Progress should feel tangible
  - Respect user's time
variables:
  first_action: [guided_tour, template_start, import_data, quick_win, watch_video]
  communication: [in_app_only, email_drip, slack_bot, combo_light, combo_heavy]
  personalization: [none, role_based, goal_based, industry_based, usage_based]
  pacing: [self_guided, daily_drip, sprint_3day, milestone_based]
  motivation: [progress_bar, celebration, social, streak, unlock]
max_iterations: 100
time_budget_seconds: 150
---

# Experiment Instructions

Design an onboarding flow for the first 7 days after signup.

Product: [Define your SaaS product here]
Key activation metric: [What action = activated user?]

Generate:
1. Day-by-day onboarding sequence (Day 0 through Day 7)
2. First-run experience (what happens immediately after signup)
3. Email/notification cadence
4. Key "aha moment" and how you get users there
5. How you handle users who go quiet on Day 2-3
'''


# Auto-register built-in domains
for _adapter_cls in [SalesPipelineAdapter, AdCreativeAdapter, LeadGenAdapter, ClientOnboardingAdapter]:
    DomainRegistry.register(_adapter_cls())
