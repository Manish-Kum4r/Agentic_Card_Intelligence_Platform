---
id: KB-14
title: Budget Allocation Framework
---

# Budget Allocation Framework

## The 70 / 20 / 10 rule
- **70%** of budget to proven campaigns (ROAS ≥ 4.0, measured for 2+ months).
- **20%** to promising tests (ROAS 3.0–4.0, or new channels with a clear hypothesis).
- **10%** to experiments (new segments, new creatives, new channels; judged on learning, not ROAS).

## Quarterly reallocation
1. Compute ROAS by campaign, channel, and segment for the trailing 2 months.
2. Pause campaigns below ROAS 3.0 (except one win-back test cycle, KB-10).
3. Move paused budget to the top-quartile ROAS campaigns, capped at +40% of their current budget per quarter.
4. Reserve the experiment bucket for the highest-LTV personas first (KB-02).

## Decision inputs the analytics platform must provide
- Blended ROAS and ROAS by channel (dashboard: Campaigns page).
- Predicted-decline cardholder counts by segment (dashboard: ML Predictions page).
- A/B test verdicts with confidence intervals (KB-07).
- Win-back pipeline: at-risk high-spend cardholders and their expected recovery revenue.

## Reporting
The monthly marketing review pack contains: blended ROAS trend, channel rollup, campaign table sorted by ROAS, A/B verdicts, and the at-risk intervention list. The analytics platform generates all five from a single pipeline run.
