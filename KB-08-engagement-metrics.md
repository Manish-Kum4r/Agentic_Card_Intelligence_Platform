---
id: KB-08
title: Engagement Score Definition
---

# Engagement Score Definition

## What it is
The engagement score is a **0–100 monthly health score** per cardholder. It is the single metric used to decide who gets a retention intervention.

## Formula (version 1.0, used by the analytics pipeline)
```
engagement_score = clip( 2.2 × logins
                       + 1.6 × app_sessions
                       + 1.1 × email_clicks
                       + 1.5 × min(monthly_spend / 1000, 20), 0, 100 )
```
where logins, app_sessions and email_clicks are the cardholder's counts in the calendar month and monthly_spend is total card spend in the month.

## Interpretation bands
| Score | Band | Action |
|---|---|---|
| 70–100 | Highly engaged | Protect; VIP treatment; no intervention needed |
| 45–69 | Healthy | Standard lifecycle messaging |
| 30–44 | Fading | Trigger engagement nudge (In-App + email) |
| 0–29 | At risk | Trigger win-back playbook (KB-09) |

## Decline definition (model target)
A cardholder is labeled **"engagement decline"** in month t when:
```
engagement_score(t + 3) < 0.80 × engagement_score(t)
```
i.e. the score is projected to fall more than 20% within the next 3 months. The prediction model scores every cardholder monthly so interventions start **before** the decline, not after.

## Data notes
- The score is computed from activity data only; it never uses transaction line-item merchant names.
- Scores are refreshed monthly, on the first business day after month close.
- The score is not a credit metric and must not be used for credit decisions (see KB-11).
