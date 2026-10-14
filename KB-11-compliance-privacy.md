---
id: KB-11
title: Compliance, Consent & Data Privacy
---

# Compliance, Consent & Data Privacy

## Data minimization
- Analytics datasets contain **aggregated activity and synthetic or tokenized identifiers only**.
- Never store full card numbers (PAN) in analytics tables. PANs live only in the payment vault; analytics uses a cardholder_id token.
- Free-text fields are scanned for PAN patterns (13–19 consecutive digits) and emails before any export. The evaluation suite includes an automated PII scan that must pass before release.

## Consent & marketing communications
- Marketing messages (email, In-App, SMS, mail) go only to cardholders with **marketing consent = Yes**.
- Every marketing message must include a working opt-out; opt-outs are honored within 24 hours.
- Transactional messages (statements, security alerts) are exempt from marketing consent but must never contain marketing content.

## Retention & access
- Raw engagement data retained 24 months; aggregates retained 5 years.
- Access to cardholder-level data is role-based and logged.
- Models must not use protected attributes; the engagement-decline model uses activity and firmographic features only.

## Model governance (responsible AI)
- Every model release ships with: metrics on a held-out time window, feature importance, and a fairness check across segments (SMB / Mid-Market / Large).
- Predictions are decision **support** for marketers — a human approves every campaign action.
- The engagement score must never be used for credit decisions, pricing, or eligibility.
