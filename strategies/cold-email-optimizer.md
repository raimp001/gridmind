---
name: cold-email-optimizer
domain: sales_pipeline
objective: Maximize reply rate for cold outreach emails to VP Engineering at mid-market SaaS
metrics:
  - name: reply_rate
    direction: higher
    baseline: 0.02
  - name: meeting_rate
    direction: higher
    baseline: 0.005
  - name: personalization
    direction: higher
    baseline: 0.3
constraints:
  - Keep emails under 150 words
  - No clickbait or misleading subject lines
  - Must include a clear, specific value proposition
  - Professional tone - this is B2B outreach
  - No fake urgency or manipulation
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

Generate a cold outreach email targeting VP of Engineering at mid-market SaaS companies (200-1000 employees).

**Product:** AI-powered code review tool that reduces PR cycle time by 40%.

## Requirements

Using the assigned variable combination, create a complete email including:

1. **Subject line** - using the `subject_style` approach
2. **Opening line** - using the `opener` approach, make it feel like a human wrote it
3. **Body** - present the `value_prop` naturally, not as a pitch
4. **CTA** - use the `cta` style, make it easy to say yes
5. **P.S. line** (optional) - only if it adds value

## What makes a great cold email
- Subject that earns the open (not tricks it)
- First line proves you did research
- Body creates genuine curiosity about the product
- CTA has low friction - easy to say yes to
- Reads like a message from a smart colleague, not a sales template
