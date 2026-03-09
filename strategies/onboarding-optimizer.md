---
name: onboarding-flow-optimizer
domain: client_onboarding
objective: Maximize 7-day activation rate - get new users to their first successful code review
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
  - Must reach "aha moment" within first 3 interactions
  - Do not overwhelm with features
  - Every step should feel like progress
  - Respect the user's time - devs are busy
variables:
  first_action: [guided_tour, template_start, connect_repo, quick_win, watch_video]
  communication: [in_app_only, email_drip, slack_bot, combo_light, combo_heavy]
  personalization: [none, role_based, goal_based, team_size_based, usage_based]
  pacing: [self_guided, daily_drip, sprint_3day, milestone_based]
  motivation: [progress_bar, celebration, social_proof, streak, unlock_features]
max_iterations: 100
time_budget_seconds: 150
---

# Experiment Instructions

Design a 7-day onboarding flow for a SaaS code review tool.

**Product:** AI code review platform (think: GitHub PR integration)
**Activation metric:** User completes their first AI-assisted code review
**Target user:** Senior developer or engineering manager, just signed up

## Generate

1. **Minute 0-5:** What happens immediately after signup
2. **Day 1:** First-run experience and initial setup
3. **Day 2-3:** Deepening engagement (or re-engagement if they went quiet)
4. **Day 4-7:** Building the habit, showing advanced value
5. **Rescue flow:** What happens if user goes dark after Day 1
6. **Communication plan:** What messages go out, when, and through which channel

## Key insight
The best onboarding doesn't feel like onboarding. It feels like the product
is so intuitive that you're just... using it. Every "onboarding step" should
be disguised as genuine product value.
