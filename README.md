# 💳 Agentic AI-Powered Commercial Card Intelligence & Marketing Optimization Platform

An end-to-end analytics + ML + GenAI + **agentic AI** platform that turns commercial
card transaction and engagement data into marketing decisions — built to match the
**Analyst – Data Analytics (Agentic)** profile: data analytics, machine learning,
RAG, agentic workflows, marketing optimization, evaluation & responsible AI, and a
professional dashboard.

> **100% synthetic data.** Every number in this project comes from a seeded,
> reproducible generator (`src/data_generation.py`). No real cardholder data is used
> anywhere. The demo runs **fully offline — no paid API, no downloads, no keys.**

---

## ✅ What it does (feature → JD requirement map)

| # | Requirement | Where it lives |
|---|---|---|
| 1 | **Data Analytics** — Python (pandas/NumPy) + SQL/MySQL | `src/data_generation.py`, `src/db.py`, `src/features.py`, `sql/01_schema_mysql.sql`, `sql/02_analytics_views_mysql.sql` (7 analytics views), SQLite demo DB |
| 2 | **Machine Learning** — engagement-decline prediction + metrics | `src/models.py` — Logistic Regression + Random Forest, ROC-AUC/accuracy/precision/recall/F1, ROC curves, feature importance, at-risk scoring |
| 3 | **GenAI & RAG** — embeddings + vector search + RAG with citations | `src/rag/` (embeddings → vector store → LLM → pipeline), `kb/` — 14 marketing knowledge-base documents |
| 4 | **Agentic AI** — multi-step workflow with tools + validation | `src/agent/` — LangGraph `StateGraph`: analyze → retrieve → recommend → validate, with conditional retry; 4 LangChain tools |
| 5 | **Marketing Optimization** — campaigns, segments, A/B testing | `src/campaigns.py` (ROAS/CTR/CVR/CPA, channel rollups, two-proportion z-test), `src/segmentation.py` (K-Means RFM personas) |
| 6 | **Evaluation & Responsible AI** — model, retrieval, response-accuracy, workflow-reliability, privacy | `src/evaluation/` + `tests/` — CV, calibration, fairness-by-segment, RAG hit-rate/MRR, agent reliability/determinism, automated PII/PAN scan |
| 7 | **Dashboard** — Streamlit app | `src/dashboard/app.py` — 8 pages: Overview, Transactions, Segments, Campaigns & A/B, ML Predictions, RAG Assistant (chat), Agent, Evaluation |

## 📊 Verified results (this exact run, seed 42 — `outputs/run_summary.json`)

| Area | Metric | Result |
|---|---|---|
| Data | cardholders / engagement-months / transactions | 3,000 / 72,000 / 120,000 |
| Data | GMV (2 years) | ₹48.99 Cr |
| ML | Random Forest ROC-AUC (held-out 6-month window) | **0.768** |
| ML | Logistic Regression ROC-AUC | 0.729 |
| ML | 5-fold CV AUC (logistic) | 0.736 ± 0.004 |
| ML | Test-window decline rate | 26.6% (18,000 rows) |
| ML | High-risk cardholders (p ≥ 0.5, latest month) | 1,302 of 3,000 |
| Responsible AI | Fairness: predicted-positive-rate gap (Large / Mid / SMB) | **≤ 1.5 pp** (0.666 / 0.651 / 0.658) |
| Responsible AI | Brier score (calibration) | 0.269 |
| Segmentation | K-Means personas (k=5) | High-Value Engaged 706 · Steady Core 818 · Growing Mid-Market 835 · At-Risk High-Spend 559 · Low-Engagement SMB 81 |
| Campaigns | Blended ROAS (budget ₹7.45L → revenue ₹38.73L) | **5.20×** |
| Campaigns | Best / worst campaign by ROAS | Travel Rewards Boost / Win-Back Dormant Users |
| A/B test | Q4 Cashback Boost, treatment vs control | **+6.6% relative lift, p = 2.3e-07 → roll out** |
| RAG | 54 KB chunks indexed; labeled queries (14) | hit-rate@3 **0.929** · P@1 0.929 · MRR 0.929 · 100% answers cited |
| Agent | 5 business questions, LangGraph workflow | reliability **1.00** · validation pass 1.00 · deterministic repeat runs |
| Privacy | Automated scan of 14 CSVs (PAN + email patterns) | **clean** · masking utility verified |
| Tests | `pytest` suite | **34/34 passed** |

All numbers above are produced by `python run_pipeline.py` and written to
`outputs/` — nothing is hard-coded or estimated.

---

## 🚀 Quickstart

### Windows (cmd / PowerShell)
```bat
cd amex_agentic_card_intelligence
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run src\dashboard\app.py
```
Then open http://localhost:8501

### macOS / Linux
```bash
cd amex_agentic_card_intelligence
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run src/dashboard/app.py
```
Then open http://localhost:8501

### Exact commands (all of them)
| Task | Command |
|---|---|
| Install dependencies | `pip install -r requirements.txt` |
| Run the full pipeline (data → models → RAG → agent → evaluation) | `python run_pipeline.py` |
| Run the test suite | `python -m pytest tests/ -q` |
| Launch the dashboard | `streamlit run src/dashboard/app.py` |
| Generate only the synthetic data | `python -m src.data_generation` |
| RAG smoke test | `python -m src.rag.rag` |
| Agent smoke test | `python -m src.agent.graph` |
| Verify the dashboard headlessly (no browser) | `python -c "from streamlit.testing.v1 import AppTest; at=AppTest.from_file('src/dashboard/app.py'); at.run(); print('exceptions:', len(at.exception))"` |

**Requirements:** Python 3.10+ (developed on 3.13). No GPU, no API keys, no MySQL
server needed for the demo (SQLite is used; MySQL scripts are included for production).

---

## 🗂 Project structure

```
amex_agentic_card_intelligence/
├── run_pipeline.py            # ONE command: data → SQL → features → models →
│                              #   segmentation → campaigns → RAG → agent → evaluation
├── config.yaml                # all knobs (seed, thresholds, floors, backends)
├── requirements.txt           # core deps (+ commented optional upgrades)
├── .env.example               # optional production overrides (LLM, MySQL…)
├── sql/
│   ├── 01_schema_mysql.sql    # MySQL 8 production schema (5 tables)
│   └── 02_analytics_views_mysql.sql  # 7 analytics views (KPIs, RFM, ROAS, A/B…)
├── kb/                        # RAG corpus — 14 marketing knowledge-base docs
│   ├── KB-01-campaign-playbook.md
│   ├── KB-02-segmentation-guide.md
│   ├── ... KB-14-budget-allocation.md
├── src/
│   ├── config.py              # config loader + env overrides
│   ├── data_generation.py     # seeded synthetic data (cardholders, engagement,
│   │                          #   transactions, campaigns, A/B test)
│   ├── db.py                  # SQLite demo loader (+ optional MySQL loader)
│   ├── features.py            # leakage-safe feature engineering + time split
│   ├── models.py              # LR + RF training, metrics, at-risk scoring, plots
│   ├── segmentation.py        # RFM + K-Means personas + PCA plot
│   ├── campaigns.py           # campaign performance + A/B z-test + plots
│   ├── rag/
│   │   ├── embeddings.py      # TF-IDF (word + char n-grams); optional ST backend
│   │   ├── vectorstore.py     # chunking + cosine vector search
│   │   ├── llm.py             # LocalLLM (offline composer) + optional API LLM
│   │   └── rag.py             # retrieve → compose → cite pipeline
│   ├── agent/
│   │   ├── tools.py           # 4 LangChain tools + analytics context
│   │   └── graph.py           # LangGraph: analyze → retrieve → recommend → validate
│   ├── evaluation/
│   │   ├── model_eval.py      # CV, calibration, fairness by segment
│   │   ├── rag_eval.py        # labeled-query retrieval metrics
│   │   ├── agent_eval.py      # workflow reliability + determinism
│   │   └── privacy.py         # PII/PAN scanner + masking utility
│   └── dashboard/app.py       # Streamlit — 8 pages
├── tests/                     # pytest: 34 tests (data, ML, campaigns, RAG, agent, privacy)
├── data/                      # generated CSVs + demo.db (gitignored, regenerated)
├── outputs/                   # all artifacts: metrics, charts, evaluations, run summary
└── docs/architecture.md       # architecture deep-dive
```

---

## 🧠 How it works

### Pipeline (`python run_pipeline.py`, ~36 s)
1. **Generate** synthetic data: 3,000 cardholders × 24 months of engagement
   (with a realistic ~28% "decline cohort"), 120k transactions, 10 campaigns,
   a 60-day A/B test with a true +8% treatment lift.
2. **Load** everything into a SQLite database (`data/demo.db`).
3. **Features** — 36 leakage-safe features per cardholder-month (current activity,
   trailing 3-month averages, momentum, firmographics). Label: engagement score
   drops >20% within 3 months.
4. **Models** — Logistic Regression + Random Forest, evaluated on a held-out
   6-month time window. Best model scores every cardholder's latest month →
   at-risk intervention list.
5. **Segmentation** — K-Means (k=5) on log-RFM + engagement → named personas.
6. **Campaigns** — CTR/CVR/ROAS/CPA per campaign and channel; A/B test read with
   a two-proportion z-test, relative lift + 95% CI.
7. **RAG** — the 14 `kb/` documents are chunked (54 chunks) and indexed with
   TF-IDF (word 1–2 grams ∪ char 3–5 grams — the char n-grams make retrieval
   robust to morphology, e.g. "allocate" ↔ "allocation").
8. **Agent** — a LangGraph workflow answers business questions using tools.
9. **Evaluation** — model CV/calibration/fairness, RAG labeled-query metrics,
   agent reliability, and a privacy scan all run and write JSON reports.
10. **Summary** — `outputs/run_summary.json`.

### Agentic workflow (LangGraph)
```
                 ┌─────────┐   ┌──────────┐   ┌─────────────┐   ┌──────────┐
 question ─────▶│ analyze │──▶│ retrieve │──▶│ recommend   │──▶│ validate │──▶ answer
 (analytics     │ (tool:  │   │ (tool:   │   │ (tool: LLM  │   │ (tool:   │
  tool:         │ whitelist│   │ KB vector│   │  composer)  │   │ checks:  │
  SQL/pandas)   │ queries) │   │ search)  │   │             │   │ citations│
                 └─────────┘   └──────────┘   └─────────────┘   │ numbers  │
                                                                │ vs data  │
                        ▲──────────── retry ◀───────────────────│ no PANs  │
                        │        (max 2 attempts)               └──────────┘
                        └──────────────────────────────────────────────▶ END
```
Every agent answer is **validated before delivery** (cites KB sources, quotes
figures that match the analytics context, contains no PAN-like sequences). If
validation fails, the graph re-retrieves and retries (max 2 attempts), then flags
`needs_review` instead of guessing.

### RAG (offline, honest)
- **Retrieval**: TF-IDF embeddings (word + char n-grams) over KB chunks, cosine
  similarity, top-k with per-document dedup.
- **Generation (demo)**: `LocalLLM` is a **deterministic, extractive composer** —
  every sentence in its answers is copied from a retrieved KB passage and cited
  `[KB-xx]`. It cannot invent facts. For production, swap in any LLM
  (see "Demo mode vs production" below).

---

## 🧪 Demo mode vs production (honesty section)

**What is real and verified in this repo:**
- The full data pipeline, SQL schema/views, feature engineering, both ML models
  and all metrics; segmentation; campaign analytics; the A/B significance test.
- A working RAG system: chunking, embedding, vector search, cited answers —
  evaluated on a labeled query set (hit-rate@3 = 0.929).
- A working **LangGraph agent** with 4 tools, conditional retry, and a validator
  gate — evaluated for reliability (1.00) and determinism (True).
- The Streamlit dashboard (all 8 pages verified headlessly with Streamlit AppTest,
  including the RAG chat and the agent run button).
- 34 passing pytest tests; privacy scanner with positive/negative controls.

**What is deliberately demo-grade (and how to upgrade):**
| Component | Demo (this repo) | Production upgrade |
|---|---|---|
| Embeddings | TF-IDF (offline) | `pip install sentence-transformers` + set `RAG_EMBEDDING_BACKEND=sentence-transformers` |
| Vector store | in-memory numpy | `pip install faiss-cpu` (or Pinecone/Weaviate) |
| LLM | `LocalLLM` extractive composer (no API) | set `LLM_BACKEND` + `OLLAMA_HOST`/`OPENAI_BASE_URL` + model env vars — `src/rag/llm.py:APILLM` speaks the OpenAI chat format (works with Ollama, vLLM, Azure…) |
| Database | SQLite (`data/demo.db`) | MySQL: apply `sql/01_schema_mysql.sql` + `sql/02_analytics_views_mysql.sql`, set `MYSQL_*` env vars, loader in `src/db.py:load_mysql` |

---
