---
id: KB-09
title: Engagement Decline Intervention Playbook
---

# Engagement Decline Intervention Playbook

## Trigger
The prediction model flags a cardholder when predicted decline probability ≥ 0.5 (High risk band) in the latest month, or when the engagement score band drops to "Fading" or "At risk" (KB-08).

## Step 1 — Diagnose (days 0–2)
Pull the cardholder's 6-month trend: logins, app sessions, email clicks, spend, transaction count. Identify the **first metric that rolled over** — that points to the cause:
- Logins/sessions down first → app experience or awareness problem → In-App + email re-engagement.
- Email clicks down first → creative fatigue → new creative test (KB-07).
- Spend/txn count down first → business softness → retention offer, not marketing.

## Step 2 — Choose the intervention
| Profile | Recommended intervention | Channel |
|---|---|---|
| High spend, fading engagement | Personal retention offer (bonus earn or cashback) | Statement insert + email |
| Mid spend, fading engagement | 2x points booster for 60 days | In-App + email |
| Low spend, fading engagement | Automated 3-email re-engagement series | Email only |
| No login 45+ days | Win-back offer (KB-03 win-back bonus) | Email + direct mail if LTV high |

## Step 3 — Execute and measure
- Launch within 5 business days of the flag.
- Measure: engagement score 3 months later, spend change, campaign-attributed revenue.
- A cardholder who recovers above score 45 exits the playbook.

## Guardrails
- Max **2 interventions per cardholder per quarter**.
- Never intervene on cardholders in the Low-Engagement SMB persona with paid channels.
- Log every intervention with campaign_id so ROAS can be attributed.
