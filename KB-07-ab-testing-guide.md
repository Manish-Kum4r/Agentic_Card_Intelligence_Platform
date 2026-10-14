---
id: KB-07
title: A/B Testing Guide
---

# A/B Testing Guide

## Design checklist
1. **One variable per test** — creative, offer, CTA label, or send time. Never two at once.
2. **Random assignment** — split the audience 50/50 at the cardholder level, not the session level.
3. **Pre-register the metric** — decide the primary metric (usually CVR) and the minimum detectable effect before launch.
4. **Sample size** — for a baseline CVR of 4% and a minimum detectable effect of 8% relative, you need roughly 55,000 impressions per arm. Use the z-test for two proportions.
5. **Duration** — run at least 14 days and always complete full weeks to neutralize day-of-week effects.

## Reading results
- Compute conversion rate per arm: CVR = conversions / impressions.
- Two-proportion z-test: pooled standard error, two-sided p-value.
- Report the **relative lift** with a 95% confidence interval, not just the p-value.
- **Decision rule:** if p < 0.05 and lift > 0, roll the winner out to 100% and log the result in the campaign playbook.
- If p ≥ 0.05, the test is inconclusive — do not "peek" daily and stop early on a lucky day.

## Common mistakes
- Peeking and stopping early inflates false positives.
- Testing on unequal arms without a reason biases the lift estimate.
- Forgetting to exclude internal/test cardholders from both arms.
