---
id: KB-02
title: Segmentation Guide
---

# Segmentation Guide

## Official company segments
| Segment | Definition | Share of portfolio | Typical annual spend |
|---|---|---|---|
| SMB | 2–50 employees | ~55% | 15k–80k |
| Mid-Market | 51–500 employees | ~30% | 80k–600k |
| Large | 500+ employees | ~15% | 600k–5M+ |

## Behavioral personas (analytics layer)
Analytics runs K-Means clustering on RFM (recency, frequency, monetary) plus average engagement score. The five personas used in the dashboard are:

- **High-Value Engaged** — high spend, high engagement, recent activity. Protect with VIP treatment; do not over-message.
- **Steady Core** — average spend, stable engagement. Growth segment for upsell campaigns.
- **Growing Mid-Market** — rising spend trend, moderate engagement. Prioritize for In-App adoption campaigns.
- **At-Risk High-Spend** — high historical spend but rising recency and falling engagement. First priority for win-back.
- **Low-Engagement SMB** — low spend, low engagement. Serve with low-cost automated email only.

## Targeting rules
- Win-back campaigns target At-Risk High-Spend and dormant High-Value Engaged cardholders.
- Acquisition-style offers never target Low-Engagement SMB with paid channels (CPA too high).
- Personalization must use segment + persona + industry only — never individual transaction line items in creative.
