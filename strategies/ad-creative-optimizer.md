---
name: ad-creative-optimizer
domain: ad_creative
objective: Maximize click-through rate on LinkedIn paid ads for developer tools
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
  - Body copy under 125 characters for primary text
  - Must include a clear CTA
  - No misleading claims or clickbait
  - Must feel native to LinkedIn feed
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

Create a LinkedIn ad creative variation for a developer productivity tool.

**Product:** AI code review tool - reduces PR review time from 2 hours to 12 minutes.
**Audience:** Engineering managers and senior developers at companies with 50+ engineers.
**Platform:** LinkedIn Sponsored Content (single image or carousel)

## Generate

1. **Headline** (under 40 chars) - stop the scroll
2. **Primary text** (under 125 chars) - the body copy above the image
3. **CTA button text** - what the button says
4. **Visual direction** - describe the image concept (1-2 sentences)
5. **Hook rationale** - why this combination should work

## Principles
- Engineers hate being sold to - lead with value
- Specific numbers beat vague claims
- Show the problem they already feel
- Make the CTA feel like the obvious next step
