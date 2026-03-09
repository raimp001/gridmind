# GridMind

> Autonomous research loops for any domain. Write a strategy doc. Run 100 experiments while you sleep.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — but generalized beyond ML. The same loop that optimizes LLM training can optimize sales pipelines, ad creative, lead generation, client onboarding, and anything else with a clear metric.

## The Pattern

```
1. Human writes a strategy doc (.md)
2. Agent runs experiments autonomously
3. Clear metric decides what stays
4. Repeat 100x while you sleep
```

You don't write code anymore. You write the `.md` file that tells AI how to think.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Generate a strategy template for your domain
gridmind init sales_pipeline -o strategies/my-sales-strategy.md

# Edit the strategy doc - this is YOUR job as a human
# Define the objective, metrics, variables, and experiment instructions

# Run the loop (dry run - no LLM calls, tests the system)
gridmind run strategies/cold-email-optimizer.md --dry-run -v

# Run for real (needs ANTHROPIC_API_KEY or OPENAI_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
gridmind run strategies/cold-email-optimizer.md -n 50

# View results
gridmind report results/cold-email-optimizer/
```

## How It Works

### Strategy Docs

The strategy doc is a markdown file with YAML frontmatter. It defines everything the AI agent needs to run experiments autonomously:

```yaml
---
name: cold-email-optimizer
domain: sales_pipeline
objective: Maximize reply rate for cold outreach emails
metrics:
  - name: reply_rate
    direction: higher
    baseline: 0.02
variables:
  subject_style: [question, statistic, pain_point, curiosity_gap]
  length: [short, medium, long]
  cta: [soft_ask, direct_ask, value_offer]
max_iterations: 100
---

# Experiment Instructions

Generate a cold email variant using the assigned variables.
Target: VP Engineering at mid-market SaaS companies.
Product: AI code review tool that cuts PR time by 40%.
```

### The Loop

Each iteration:
1. **Agent reads** the strategy doc + past results
2. **Designs** an experiment (picks variables, forms hypothesis)
3. **Generates** the artifact (email, ad copy, onboarding flow, etc.)
4. **Evaluates** against the defined metrics
5. **Keeps or discards** based on whether it beats the current best
6. **Repeats** — learning from every iteration

### Built-in Domains

| Domain | What it optimizes |
|--------|------------------|
| `sales_pipeline` | Cold emails, outreach sequences, sales scripts |
| `ad_creative` | Ad copy, headlines, CTAs, creative concepts |
| `lead_gen` | Landing pages, lead magnets, form optimization |
| `client_onboarding` | Welcome flows, activation sequences, email drips |

```bash
# List all domains
gridmind domains

# Generate a template for any domain
gridmind init ad_creative -o strategies/my-ads.md
gridmind init lead_gen -o strategies/my-leads.md
gridmind init client_onboarding -o strategies/my-onboarding.md
```

## API Server

Run GridMind as an API for programmatic access:

```bash
uvicorn gridmind.server:app --reload

# Start a loop
curl -X POST http://localhost:8000/loops \
  -H "Content-Type: application/json" \
  -d '{"strategy_path": "strategies/cold-email-optimizer.md", "dry_run": true}'

# Check status
curl http://localhost:8000/loops/{loop_id}

# Get best result
curl http://localhost:8000/loops/{loop_id}/best
```

## Project Structure

```
gridmind/
├── core/
│   ├── strategy.py    # Strategy doc parser (the human-agent interface)
│   ├── experiment.py  # Experiment execution and tracking
│   ├── metrics.py     # Metric evaluation (what stays, what goes)
│   ├── agent.py       # LLM agent (designs + evaluates experiments)
│   └── loop.py        # The autonomous loop (the pattern)
├── domains/
│   └── registry.py    # Domain adapters (sales, ads, lead gen, onboarding)
├── cli.py             # Command-line interface
└── server.py          # FastAPI server
strategies/            # Your strategy docs live here
results/               # Experiment results saved here
tests/                 # Test suite
```

## Configuration

### LLM Providers

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY=sk-ant-...
gridmind run strategy.md --provider anthropic --model claude-sonnet-4-20250514

# OpenAI
export OPENAI_API_KEY=sk-...
gridmind run strategy.md --provider openai --model gpt-4o
```

### Key Options

```bash
gridmind run strategy.md \
  --dry-run          # Mock mode, no LLM calls
  --verbose          # Show all experiment details
  --iterations 50    # Override max iterations
  --results-dir out  # Custom output directory
  --provider openai  # Switch LLM provider
```

## Running Tests

```bash
pytest tests/ -v
```

## The Insight

This isn't an ML research trick. This is the pattern.

The person who figures out how to apply this loop to sales pipelines, client onboarding, ad creative, lead gen — not just LLM training — is going to build something massive.

The strategy doc is the new code. The metric is the new test suite. The loop is the new CI/CD.
