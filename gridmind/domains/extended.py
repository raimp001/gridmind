"""Extended domain adapters: landing pages, SEO/AEO, pricing, warm outreach,
AP/AR, procurement, job postings, YouTube thumbnails, call scripts."""

from __future__ import annotations

from dataclasses import dataclass

from gridmind.domains.registry import DomainRegistry


@dataclass
class LandingPageAdapter:
    name: str = "landing_page"
    description: str = "Optimize landing page copy, layout, and conversion flow"

    def get_evaluation_prompt(self, artifact: str, variables: dict) -> str:
        return (
            "Evaluate this landing page concept. Score realistically (0.0-1.0):\n"
            "- conversion_rate: likelihood visitor completes desired action\n"
            "- bounce_rate: likelihood visitor leaves immediately (lower is better)\n"
            "- clarity: how quickly visitor understands the offer\n"
            "- trust: how credible and trustworthy it feels\n"
            "- mobile_score: how well it works on mobile\n"
            f"\n## Artifact\n{artifact}\n"
            f"\n## Variables\n{variables}\n"
            "\nReturn JSON: {\"conversion_rate\": x, \"bounce_rate\": x, \"clarity\": x, "
            "\"trust\": x, \"mobile_score\": x}"
        )

    def get_strategy_template(self) -> str:
        return '''---
name: landing-page-optimizer
domain: landing_page
objective: Maximize visitor-to-signup conversion rate
signal_speed: medium
scoring_window_hours: 168
metrics:
  - name: conversion_rate
    direction: higher
    baseline: 0.03
  - name: bounce_rate
    direction: lower
    baseline: 0.65
  - name: clarity
    direction: higher
    baseline: 0.5
  - name: trust
    direction: higher
    baseline: 0.4
constraints:
  - Page must load under 3 seconds
  - Single clear CTA above the fold
  - Must work on mobile
  - No misleading claims or dark patterns
variables:
  headline_type: [benefit, pain_point, social_proof, question, how_to, number]
  hero_layout: [text_left_image_right, centered_text, video_hero, testimonial_hero, product_demo]
  social_proof: [logos, testimonials, stats, case_study, user_count, awards]
  cta_text: [get_started, try_free, see_demo, start_trial, claim_offer, learn_more]
  urgency: [none, limited_time, limited_spots, social_proof_live, countdown]
  form_length: [email_only, email_name, email_name_company, full_form]
  above_fold: [benefit_focused, problem_focused, demo_focused, testimonial_focused]
max_iterations: 100
time_budget_seconds: 120
---

# Experiment Instructions

Design a landing page concept optimized for conversion.

Product: [Define your product]
Target visitor: [Who is landing on this page and from where?]
Desired action: [Signup, demo request, purchase, etc.]

Generate:
1. Page headline and subheadline
2. Above-the-fold layout description
3. Hero section content
4. Social proof section
5. CTA button text and placement
6. Form fields (if applicable)
7. Below-the-fold content structure
8. Mobile layout considerations
'''


@dataclass
class SEOAEOAdapter:
    name: str = "seo_aeo"
    description: str = "Optimize content for search engines and AI answer engines (AEO/GEO)"

    def get_evaluation_prompt(self, artifact: str, variables: dict) -> str:
        return (
            "Evaluate this SEO/AEO content artifact. Score realistically (0.0-1.0):\n"
            "- search_rank_potential: likelihood of ranking on page 1\n"
            "- ai_citation_likelihood: chance of being cited by AI answer engines\n"
            "- content_depth: thoroughness and expertise demonstrated\n"
            "- user_intent_match: how well it matches search intent\n"
            "- engagement: likelihood of being read, shared, linked to\n"
            f"\n## Artifact\n{artifact}\n"
            f"\n## Variables\n{variables}\n"
            "\nReturn JSON: {\"search_rank_potential\": x, \"ai_citation_likelihood\": x, "
            "\"content_depth\": x, \"user_intent_match\": x, \"engagement\": x}"
        )

    def get_strategy_template(self) -> str:
        return '''---
name: seo-aeo-content-optimizer
domain: seo_aeo
objective: Maximize organic search ranking and AI answer engine citations
signal_speed: slow
scoring_window_hours: 720
metrics:
  - name: search_rank_potential
    direction: higher
    baseline: 0.3
  - name: ai_citation_likelihood
    direction: higher
    baseline: 0.2
  - name: content_depth
    direction: higher
    baseline: 0.5
  - name: user_intent_match
    direction: higher
    baseline: 0.5
constraints:
  - No keyword stuffing
  - Must provide genuine value to the reader
  - Content must be factually accurate
  - Include structured data / schema markup recommendations
  - Optimize for both traditional search and AI answer engines
variables:
  content_format: [how_to_guide, listicle, comparison, deep_dive, faq, data_study, case_study]
  structure: [inverted_pyramid, problem_solution, chronological, hub_spoke, skyscraper]
  ai_optimization: [direct_answer_block, structured_faq, entity_markup, citation_bait, definition_first]
  authority_signal: [original_data, expert_quotes, case_studies, methodology, benchmarks]
  intent_match: [informational, commercial, navigational, transactional]
  title_formula: [number_list, how_to, question, year_updated, vs_comparison, ultimate_guide]
max_iterations: 100
time_budget_seconds: 180
---

# Experiment Instructions

Create an SEO/AEO content piece optimized for both search engines and AI answer engines.

Target keyword: [Define primary keyword]
Search intent: [What is the searcher trying to accomplish?]
Competitor benchmark: [What currently ranks #1?]

Generate:
1. Title tag and meta description
2. Content outline with H2/H3 structure
3. Opening paragraph (optimized for featured snippets / AI answers)
4. Key sections with topic coverage
5. Structured data / FAQ schema recommendations
6. Internal linking strategy
7. AEO optimization notes (how to get cited by AI)
'''


@dataclass
class PricingAdapter:
    name: str = "pricing"
    description: str = "Optimize pricing pages, tier structures, and packaging"

    def get_evaluation_prompt(self, artifact: str, variables: dict) -> str:
        return (
            "Evaluate this pricing strategy artifact. Score realistically (0.0-1.0):\n"
            "- conversion_rate: likelihood of plan selection\n"
            "- revenue_per_visitor: expected revenue yield per pricing page visit\n"
            "- plan_clarity: how easily users understand what they get\n"
            "- upgrade_potential: likelihood of future upsell/expansion\n"
            "- competitive_positioning: how well it positions against alternatives\n"
            f"\n## Artifact\n{artifact}\n"
            f"\n## Variables\n{variables}\n"
            "\nReturn JSON: {\"conversion_rate\": x, \"revenue_per_visitor\": x, "
            "\"plan_clarity\": x, \"upgrade_potential\": x, \"competitive_positioning\": x}"
        )

    def get_strategy_template(self) -> str:
        return '''---
name: pricing-page-optimizer
domain: pricing
objective: Maximize revenue per pricing page visitor
signal_speed: medium
scoring_window_hours: 336
metrics:
  - name: revenue_per_visitor
    direction: higher
    baseline: 0.15
  - name: conversion_rate
    direction: higher
    baseline: 0.04
  - name: plan_clarity
    direction: higher
    baseline: 0.5
  - name: upgrade_potential
    direction: higher
    baseline: 0.3
constraints:
  - Pricing must be transparent (no hidden fees)
  - Must include a free or low-commitment entry point
  - Enterprise plan must have "Contact sales" option
  - Comparison between plans must be clear
variables:
  tier_count: [two, three, four]
  anchor: [highest_first, recommended_middle, progressive]
  free_tier: [freemium, free_trial_7d, free_trial_14d, free_trial_30d, none]
  pricing_model: [per_seat, usage_based, flat_rate, hybrid, feature_gated]
  social_proof: [customer_logos, testimonials, user_count, case_study_roi]
  cta_style: [start_free, buy_now, get_started, talk_to_sales, see_demo]
  discount: [annual_save_20, annual_save_2months, none, startup_discount]
  highlight: [most_popular_badge, recommended_badge, best_value_badge, enterprise_badge]
max_iterations: 100
time_budget_seconds: 150
---

# Experiment Instructions

Design a pricing page and tier structure.

Product: [Define your SaaS product]
Current pricing: [What do you charge today?]
Target customer segments: [Startup, SMB, Mid-market, Enterprise?]
Key competitor pricing: [What do alternatives charge?]

Generate:
1. Tier names and positioning
2. Price points for each tier (monthly/annual)
3. Feature matrix (what's in each tier)
4. Recommended/highlighted plan and why
5. CTA button text for each tier
6. Social proof placement
7. FAQ section (addressing pricing objections)
'''


@dataclass
class WarmOutreachAdapter:
    name: str = "warm_outreach"
    description: str = "Optimize warm outreach, nurture sequences, and re-engagement"

    def get_evaluation_prompt(self, artifact: str, variables: dict) -> str:
        return (
            "Evaluate this warm outreach artifact. Score realistically (0.0-1.0):\n"
            "- response_rate: likelihood of getting a response\n"
            "- relationship_score: how much it strengthens the relationship\n"
            "- conversion_rate: likelihood of advancing to next stage\n"
            "- authenticity: how genuine and non-spammy it feels\n"
            f"\n## Artifact\n{artifact}\n"
            f"\n## Variables\n{variables}\n"
            "\nReturn JSON: {\"response_rate\": x, \"relationship_score\": x, "
            "\"conversion_rate\": x, \"authenticity\": x}"
        )

    def get_strategy_template(self) -> str:
        return '''---
name: warm-outreach-optimizer
domain: warm_outreach
objective: Maximize response rate from warm leads and existing contacts
signal_speed: fast
scoring_window_hours: 72
metrics:
  - name: response_rate
    direction: higher
    baseline: 0.15
  - name: relationship_score
    direction: higher
    baseline: 0.5
  - name: conversion_rate
    direction: higher
    baseline: 0.08
  - name: authenticity
    direction: higher
    baseline: 0.6
constraints:
  - Must feel personal, not automated
  - Reference specific shared context or history
  - No hard sell on first touch
  - Appropriate follow-up cadence (not too aggressive)
variables:
  trigger: [content_engagement, company_news, role_change, anniversary, mutual_connection, event_attendance]
  channel: [email, linkedin_dm, twitter_dm, slack, text]
  opener: [congratulate, share_resource, ask_opinion, offer_help, reference_past]
  ask: [coffee_chat, intro_request, feedback_request, collaboration, soft_pitch, no_ask]
  follow_up: [one_touch, two_touch, three_touch_sequence]
  personalization_depth: [light, medium, deep_research]
max_iterations: 100
time_budget_seconds: 120
---

# Experiment Instructions

Create a warm outreach message for re-engaging an existing contact.

Relationship context: [How do you know this person?]
Trigger event: [What prompted the outreach?]
Ultimate goal: [Meeting, referral, partnership, deal?]

Generate:
1. Channel selection rationale
2. Subject line / opening hook
3. Message body (personalized to the trigger)
4. The ask (or non-ask for pure nurture)
5. Follow-up sequence (if no response)
'''


@dataclass
class APARAdapter:
    name: str = "ap_ar"
    description: str = "Optimize accounts payable/receivable: invoicing, collections, payment terms"

    def get_evaluation_prompt(self, artifact: str, variables: dict) -> str:
        return (
            "Evaluate this AP/AR artifact. Score realistically (0.0-1.0):\n"
            "- collection_rate: likelihood of on-time payment\n"
            "- days_sales_outstanding: speed of payment (lower is better)\n"
            "- relationship_preservation: maintaining good vendor/client relations\n"
            "- compliance: adherence to financial regulations\n"
            f"\n## Artifact\n{artifact}\n"
            f"\n## Variables\n{variables}\n"
            "\nReturn JSON: {\"collection_rate\": x, \"days_sales_outstanding\": x, "
            "\"relationship_preservation\": x, \"compliance\": x}"
        )

    def get_strategy_template(self) -> str:
        return '''---
name: ar-collections-optimizer
domain: ap_ar
objective: Minimize days sales outstanding while preserving client relationships
signal_speed: slow
scoring_window_hours: 720
metrics:
  - name: collection_rate
    direction: higher
    baseline: 0.75
  - name: days_sales_outstanding
    direction: lower
    baseline: 0.65
  - name: relationship_preservation
    direction: higher
    baseline: 0.7
  - name: compliance
    direction: higher
    baseline: 0.9
constraints:
  - Must comply with fair debt collection practices
  - Professional tone at all times
  - Escalation path must be clear
  - Never threaten or use aggressive language
  - Include clear payment instructions
variables:
  reminder_timing: [3_days_before, on_due_date, 3_days_after, 7_days_after, 14_days_after]
  tone: [friendly, professional, firm, urgent, collaborative]
  channel: [email, phone_script, letter, sms, multi_channel]
  incentive: [early_payment_discount, payment_plan, none, convenience_fee_waiver]
  escalation: [gentle_reminder, manager_cc, formal_notice, collections_warning]
  personalization: [automated_template, semi_personalized, fully_personalized]
max_iterations: 100
time_budget_seconds: 120
---

# Experiment Instructions

Design an accounts receivable collection communication.

Business context: [B2B SaaS, professional services, etc.]
Invoice profile: [Average size, typical payment terms]
Client segment: [Enterprise, SMB, startup]

Generate:
1. Communication timeline (pre-due to 60+ days overdue)
2. Email/message for this specific touchpoint
3. Subject line
4. Tone and escalation level
5. Payment facilitation (links, options, plans)
6. Next step if no response
'''


@dataclass
class ProcurementAdapter:
    name: str = "procurement"
    description: str = "Optimize procurement: RFPs, vendor negotiations, cost reduction"

    def get_evaluation_prompt(self, artifact: str, variables: dict) -> str:
        return (
            "Evaluate this procurement artifact. Score realistically (0.0-1.0):\n"
            "- cost_reduction: potential savings vs current spend\n"
            "- vendor_quality: likelihood of getting quality responses/vendors\n"
            "- process_efficiency: how streamlined the process is\n"
            "- risk_mitigation: how well it addresses procurement risks\n"
            f"\n## Artifact\n{artifact}\n"
            f"\n## Variables\n{variables}\n"
            "\nReturn JSON: {\"cost_reduction\": x, \"vendor_quality\": x, "
            "\"process_efficiency\": x, \"risk_mitigation\": x}"
        )

    def get_strategy_template(self) -> str:
        return '''---
name: procurement-optimizer
domain: procurement
objective: Maximize cost savings while maintaining vendor quality
signal_speed: slow
scoring_window_hours: 720
metrics:
  - name: cost_reduction
    direction: higher
    baseline: 0.05
  - name: vendor_quality
    direction: higher
    baseline: 0.6
  - name: process_efficiency
    direction: higher
    baseline: 0.4
  - name: risk_mitigation
    direction: higher
    baseline: 0.5
constraints:
  - Must comply with company procurement policies
  - Maintain competitive bidding where required
  - Include evaluation criteria transparency
  - Preserve strategic vendor relationships
variables:
  rfp_structure: [streamlined, detailed, phased, reverse_auction, negotiated]
  evaluation_weight: [price_heavy, quality_heavy, balanced, innovation_weighted, risk_weighted]
  negotiation_style: [collaborative, competitive, bundled, long_term_commitment, volume_based]
  vendor_pool: [incumbent_only, open_market, pre_qualified, referral_based]
  contract_terms: [standard, flexible, performance_based, milestone_based]
  communication: [formal_rfp, informal_inquiry, hybrid, platform_based]
max_iterations: 100
time_budget_seconds: 180
---

# Experiment Instructions

Design a procurement approach for a specific category.

Category: [Software, services, hardware, etc.]
Annual spend: [Current budget]
Current pain: [Too expensive, slow process, poor quality?]

Generate:
1. Sourcing strategy
2. RFP/RFQ structure (or alternative approach)
3. Evaluation criteria and weighting
4. Negotiation playbook
5. Contract terms to optimize
6. Timeline and milestones
'''


@dataclass
class JobPostingAdapter:
    name: str = "job_posting"
    description: str = "Optimize job postings, recruitment messaging, and candidate attraction"

    def get_evaluation_prompt(self, artifact: str, variables: dict) -> str:
        return (
            "Evaluate this job posting artifact. Score realistically (0.0-1.0):\n"
            "- application_rate: likelihood of attracting qualified applicants\n"
            "- candidate_quality: caliber of candidates it would attract\n"
            "- inclusivity: how welcoming to diverse candidates\n"
            "- clarity: how clear the role and expectations are\n"
            f"\n## Artifact\n{artifact}\n"
            f"\n## Variables\n{variables}\n"
            "\nReturn JSON: {\"application_rate\": x, \"candidate_quality\": x, "
            "\"inclusivity\": x, \"clarity\": x}"
        )

    def get_strategy_template(self) -> str:
        return '''---
name: job-posting-optimizer
domain: job_posting
objective: Maximize qualified application rate
signal_speed: medium
scoring_window_hours: 168
metrics:
  - name: application_rate
    direction: higher
    baseline: 0.02
  - name: candidate_quality
    direction: higher
    baseline: 0.4
  - name: inclusivity
    direction: higher
    baseline: 0.5
  - name: clarity
    direction: higher
    baseline: 0.5
constraints:
  - No discriminatory language
  - Realistic requirements (no "10 years of 5-year-old tech")
  - Salary range must be included
  - Clear about remote/hybrid/onsite
variables:
  title_style: [standard, creative, seniority_clear, impact_focused, team_focused]
  opening: [mission_first, team_culture, problem_to_solve, growth_opportunity, impact_statement]
  requirements: [strict_list, flexible_nice_to_have, skills_matrix, competency_based, minimal]
  compensation: [range_upfront, range_bottom, competitive_plus_equity, total_comp_breakdown]
  culture_signal: [values_list, day_in_life, team_quotes, perks_list, growth_stories]
  format: [traditional, conversational, challenge_based, story_driven, data_driven]
max_iterations: 100
time_budget_seconds: 120
---

# Experiment Instructions

Create a job posting optimized for attracting top candidates.

Role: [Title and level]
Team: [What team, how big, who they report to]
Key challenge: [The main problem this hire will solve]

Generate:
1. Job title
2. Opening paragraph (the hook)
3. Role description and impact
4. Requirements (must-have vs nice-to-have)
5. Compensation and benefits
6. Application CTA
'''


@dataclass
class YouTubeThumbnailAdapter:
    name: str = "yt_thumbnail"
    description: str = "Optimize YouTube thumbnails, titles, and click-through concepts"

    def get_evaluation_prompt(self, artifact: str, variables: dict) -> str:
        return (
            "Evaluate this YouTube thumbnail/title concept. Score realistically (0.0-1.0):\n"
            "- click_through_rate: likelihood of getting clicked in search/browse\n"
            "- curiosity_gap: how much it makes viewers want to know more\n"
            "- brand_recognition: how recognizable and consistent with channel brand\n"
            "- scroll_stop: how much it stands out in a feed of thumbnails\n"
            f"\n## Artifact\n{artifact}\n"
            f"\n## Variables\n{variables}\n"
            "\nReturn JSON: {\"click_through_rate\": x, \"curiosity_gap\": x, "
            "\"brand_recognition\": x, \"scroll_stop\": x}"
        )

    def get_strategy_template(self) -> str:
        return '''---
name: yt-thumbnail-optimizer
domain: yt_thumbnail
objective: Maximize click-through rate on YouTube videos
signal_speed: fast
scoring_window_hours: 48
metrics:
  - name: click_through_rate
    direction: higher
    baseline: 0.04
  - name: curiosity_gap
    direction: higher
    baseline: 0.4
  - name: scroll_stop
    direction: higher
    baseline: 0.4
  - name: brand_recognition
    direction: higher
    baseline: 0.5
constraints:
  - No misleading thumbnails (clickbait that doesn't deliver)
  - Must be readable at small sizes (mobile)
  - Maximum 5 words of text on thumbnail
  - Face must be visible and expressive (if using face)
variables:
  face_expression: [shocked, excited, curious, serious, laughing, pointing]
  text_style: [bold_number, question, one_word, before_after, none, emoji_accent]
  color_scheme: [high_contrast, brand_colors, complementary, monochrome_pop, dark_dramatic]
  composition: [face_left_text_right, centered, split_screen, product_focus, reaction_style]
  hook_type: [number, question, controversy, transformation, secret, challenge]
  background: [solid_color, gradient, contextual, blurred, pattern]
max_iterations: 100
time_budget_seconds: 90
---

# Experiment Instructions

Design a YouTube thumbnail and title combination.

Channel: [Channel name and niche]
Video topic: [What the video is about]
Target viewer: [Who should click this?]

Generate:
1. Video title (under 60 chars, optimized for CTR)
2. Thumbnail concept description (visual layout)
3. Text overlay on thumbnail (max 5 words)
4. Face/expression direction
5. Color palette
6. Why this combination should outperform baseline
'''


@dataclass
class CallScriptAdapter:
    name: str = "call_script"
    description: str = "Optimize discovery calls, sales scripts, and demo presentations"

    def get_evaluation_prompt(self, artifact: str, variables: dict) -> str:
        return (
            "Evaluate this call script artifact. Score realistically (0.0-1.0):\n"
            "- meeting_to_opportunity: likelihood of advancing to next stage\n"
            "- discovery_depth: how well it uncovers real pain points\n"
            "- rapport: how naturally it builds connection\n"
            "- objection_handling: how well it addresses common objections\n"
            f"\n## Artifact\n{artifact}\n"
            f"\n## Variables\n{variables}\n"
            "\nReturn JSON: {\"meeting_to_opportunity\": x, \"discovery_depth\": x, "
            "\"rapport\": x, \"objection_handling\": x}"
        )

    def get_strategy_template(self) -> str:
        return '''---
name: discovery-call-optimizer
domain: call_script
objective: Maximize meeting-to-opportunity conversion rate
signal_speed: slow
scoring_window_hours: 336
metrics:
  - name: meeting_to_opportunity
    direction: higher
    baseline: 0.25
  - name: discovery_depth
    direction: higher
    baseline: 0.4
  - name: rapport
    direction: higher
    baseline: 0.5
  - name: objection_handling
    direction: higher
    baseline: 0.4
constraints:
  - Call must be under 30 minutes
  - Must ask at least 3 discovery questions
  - Never badmouth competitors directly
  - Always confirm next steps before ending
variables:
  opener: [rapport_first, agenda_set, bold_question, shared_research, customer_story]
  discovery_framework: [spin, meddic, bant, challenger, consultative, gap_selling]
  demo_style: [no_demo, micro_demo, full_walkthrough, use_case_specific, roi_calculator]
  objection_approach: [acknowledge_redirect, feel_felt_found, reframe, social_proof, data_driven]
  close: [next_step_close, trial_offer, mutual_action_plan, urgency_based, value_summary]
  pacing: [fast_direct, slow_consultative, adaptive, question_heavy, story_driven]
max_iterations: 100
time_budget_seconds: 180
---

# Experiment Instructions

Design a discovery/sales call script.

Product: [What are you selling?]
Buyer persona: [Title, company size, typical pain points]
Sales cycle: [How long, how many stakeholders?]

Generate:
1. Opening (first 2 minutes)
2. Discovery questions sequence (ordered by depth)
3. Transition to value presentation
4. Demo/presentation approach (if applicable)
5. Objection handling playbook (top 3 objections)
6. Close and next steps
7. Post-call follow-up message
'''


# Auto-register all extended domain adapters
for _cls in [
    LandingPageAdapter,
    SEOAEOAdapter,
    PricingAdapter,
    WarmOutreachAdapter,
    APARAdapter,
    ProcurementAdapter,
    JobPostingAdapter,
    YouTubeThumbnailAdapter,
    CallScriptAdapter,
]:
    DomainRegistry.register(_cls())
