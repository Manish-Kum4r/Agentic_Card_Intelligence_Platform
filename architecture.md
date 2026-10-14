# Architecture — Agentic AI-Powered Commercial Card Intelligence Platform

## 1. System overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT DASHBOARD (8 pages)                   │
│  Overview · Transactions · Segments · Campaigns & A/B · ML Predictions    │
│  RAG Assistant (chat) · Agent · Evaluation & Responsible AI               │
└───────────────┬──────────────────────────────────────┬───────────────────┘
                │ reads artifacts                       │ runs live
                ▼                                       ▼
┌──────────────────────────┐              ┌──────────────────────────────┐
│  outputs/ (artifacts)    │              │  LANGGRAPH AGENT             │
│  run_summary.json        │              │  analyze → retrieve →        │
│  model_metrics.csv       │              │  recommend → validate (retry)│
│  campaign_*.csv/json     │              │  tools: analytics / retrieval │
│  segment_*.csv           │              │  / recommendation / validator │
│  at_risk_cardholders.csv │              └──────────┬───────────────────┘
│  evaluation/*.json       │                         │ uses
│  *.png charts            │                         ▼
└─────────────▲────────────┘              ┌──────────────────────────────┐
              │                           │  RAG PIPELINE (src/rag)      │
              │                           │  kb/*.md → chunks → TF-IDF   │
              │                           │  (word ∪ char n-grams) →     │
              │                           │  cosine top-k → LocalLLM     │
              │                           │  extractive answer + [KB-xx] │
              │                           └──────────────────────────────┘
   ┌──────────┴───────────────────────────────────────────────────────────┐
   │                    RUN_PIPELINE.PY (orchestrator, ~36 s)             │
   │  1 data_generation → 2 db (SQLite) → 3 features → 4 models           │
   │  → 5 segmentation → 6 campaigns+A/B → 7 RAG build → 8 agent smoke   │
   │  → 9 evaluation suite → 10 run summary                               │
   └──────────┬───────────────────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                               │
│  data/*.csv (synthetic, seeded)  +  data/demo.db (SQLite)                 │
│  Production: MySQL 8 — sql/01_schema_mysql.sql + 02_analytics_views      │
└──────────────────────────────────────────────────────────────────────────┘
```

## 2. Data model (5 tables)
| Table | Rows (this run) | Contents |
|---|---|---|
| `cardholders` | 3,000 | firmographics: industry, region, segment, employees, tenure, spend baseline |
| `engagement_monthly` | 72,000 | per cardholder-month: logins, app_sessions, email_clicks, spend, txn_count, engagement_score (0–100), decline label |
| `transactions` | 120,000 | txn_id, cardholder, date, merchant_category, amount, channel, rewards |
| `campaigns` | 10 | budget, impressions, clicks, conversions, attributed_revenue |
| `ab_test_daily` | 120 | 60 days × 2 arms (control / treatment) |

MySQL equivalents: `sql/01_schema_mysql.sql` (DDL + indexes),
`sql/02_analytics_views_mysql.sql` (7 views: monthly KPIs, RFM, campaign
performance, channel rollup, category mix, A/B summary, engagement momentum).

## 3. ML — engagement-decline prediction
- **Target**: `engagement_score(t+3) < 0.80 × engagement_score(t)` (documented in kb/KB-08).
- **Features (36)**: current-month activity, trailing 3-month averages, momentum
  (current − trailing), per-transaction ratios, firmographics, one-hot
  industry/region, calendar month. **No look-ahead leakage** (tested).
- **Split**: time-based — train on the first 18 months, test on the last 6
  (18,000 test rows, 26.6% decline rate).
- **Models**: Logistic Regression (scaled, class_weight=balanced) and
  Random Forest (300 trees, min_samples_leaf=20). Best: RF, ROC-AUC **0.768**.
- **Deployment artifact**: `outputs/engagement_decline_model.joblib` scores every
  cardholder's latest month → `at_risk_cardholders.csv` (top-100 intervention list).

## 4. RAG
- **Corpus**: 14 markdown docs in `kb/` with front matter (`id`, `title`).
- **Chunking**: blank-line paragraphs, min 60 chars → 54 chunks.
- **Embeddings**: TF-IDF = word(1–2)-grams ∪ char_wb(3–5)-grams, sublinear tf,
  L2-normalized (cosine = dot product). Char n-grams give morphology robustness.
- **Search**: cosine top-k (k=4), best chunk per document.
- **Answer**: extractive composition — best-scoring sentences from retrieved
  chunks, one bullet per source, `[KB-xx]` citations + relevance scores.
- **Evaluation**: 14 labeled queries → hit-rate@3 **0.929**, P@1 0.929, MRR 0.929.

## 5. Agentic AI (LangGraph)
State: `question, analysis, retrieved, recommendation, validation, attempts,
trace, status`.

| Node | Tool | Does |
|---|---|---|
| `analyze` | `analytics_tool` | routes the question to a whitelisted analytics query (kpis / campaign_performance / channel_rollup / segment_summary / decline_risk_summary / ab_test) over the pipeline outputs + SQLite |
| `retrieve` | `retrieval_tool` | top-3 KB passages with relevance scores |
| `recommend` | `recommendation_tool` | composes a grounded recommendation (analytics findings + KB guidance) |
| `validate` | `validator_tool` | gates: non-empty, ≥1 `[KB-xx]` citation, ≥1 figure matches the analytics context, no PAN-like sequences |

Conditional edge after `validate`: pass → END · fail & attempts < 2 → retry
(re-retrieve) · fail & attempts = 2 → END with `status = needs_review`.

## 6. Evaluation & responsible AI
| Suite | File | Gates / outputs |
|---|---|---|
| Model | `src/evaluation/model_eval.py` | 5-fold stratified CV AUC (0.736 ± 0.004), Brier 0.269, fairness by segment (gap ≤ 1.5 pp), AUC floor 0.65 |
| RAG | `src/evaluation/rag_eval.py` | hit-rate@3, P@1, MRR on 14 labeled queries; floors 0.75 / 0.60; 100% cited |
| Agent | `src/evaluation/agent_eval.py` | 5 questions: workflow reliability 1.00, validation pass 1.00, deterministic repeat runs |
| Privacy | `src/evaluation/privacy.py` | PAN/email scan of all CSVs (context-aware regex), masking utility self-check |
| Unit/integration | `tests/` (34 tests) | data integrity, leakage, model quality, campaign math, A/B detection, RAG grounding, agent validation, privacy controls |

## 7. Dashboard
`streamlit run src/dashboard/app.py` — 8 pages, verified headlessly with
Streamlit `AppTest` (all pages: 0 exceptions, 0 errors; RAG chat and agent
button exercised interactively). Charts are matplotlib PNGs from `outputs/`;
RAG chat and agent run live against the KB and pipeline artifacts.
