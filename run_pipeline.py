#!/usr/bin/env python3
"""End-to-end pipeline — one command builds everything.

  python run_pipeline.py

Steps:
  1. Generate synthetic data            (data/*.csv)
  2. Load into SQLite demo DB           (data/demo.db)
  3. Feature engineering                (engagement-decline features)
  4. Train + evaluate ML models         (outputs/model_*)
  5. Customer segmentation (K-Means)    (outputs/*segment*)
  6. Campaign performance + A/B test    (outputs/campaign_*, ab_test_*)
  7. Build RAG index over kb/           (in-memory, TF-IDF, offline)
  8. Smoke-test RAG + LangGraph agent
  9. Run evaluation suite               (outputs/evaluation/*)
 10. Write run summary

Demo mode is 100% offline — no paid API, no downloads.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.config import load_config, paths
from src.data_generation import generate_all
from src.db import load_sqlite
from src.features import build_feature_table
from src.models import run_modeling
from src.segmentation import run_segmentation
from src.campaigns import run_campaigns
from src.rag.rag import build_rag
from src.agent.graph import build_agent, run_agent
from src.evaluation.model_eval import run_model_evaluation
from src.evaluation.rag_eval import run_rag_evaluation
from src.evaluation.agent_eval import run_agent_evaluation
from src.evaluation.privacy import run_privacy_scan


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    p = paths(cfg)
    (p["outputs"]).mkdir(parents=True, exist_ok=True)
    (p["outputs"] / "evaluation").mkdir(parents=True, exist_ok=True)
    seed = cfg["project"]["seed"]

    step = lambda n, msg: print(f"\n=== [{n}/10] {msg} " + "=" * max(0, 50 - len(msg)))

    step(1, "Generating synthetic data")
    dfs = generate_all(p["data"], seed=seed,
                       n_cardholders=cfg["data"]["n_cardholders"],
                       months=cfg["data"]["months"],
                       start_month=cfg["data"]["start_month"],
                       n_transactions=cfg["data"]["n_transactions"])
    for name, df in dfs.items():
        print(f"  {name:15s} {df.shape}")

    step(2, "Loading SQLite demo database")
    counts = load_sqlite(p["data"], p["data"] / "demo.db")
    print("  rows:", counts)

    step(3, "Feature engineering")
    feature_table, feature_cols = build_feature_table(dfs["engagement"], dfs["cardholders"])
    feature_table.to_csv(p["outputs"] / "feature_table.csv", index=False)
    print(f"  features: {len(feature_cols)}  rows: {len(feature_table)}")

    step(4, "Training engagement-decline models")
    model_summary = run_modeling(feature_table, feature_cols, p["outputs"],
                                 random_state=cfg["model"]["random_state"],
                                 test_months=cfg["model"]["test_months"])
    for m in model_summary["metrics"]:
        print(f"  {m['model']:22s} ROC-AUC {m['roc_auc']}  F1 {m['f1']}")
    print(f"  best: {model_summary['best_model']}")

    step(5, "Customer segmentation (K-Means)")
    seg_summary = run_segmentation(dfs["transactions"], dfs["engagement"],
                                   dfs["cardholders"], p["outputs"],
                                   k=cfg["segmentation"]["k"],
                                   random_state=cfg["segmentation"]["random_state"])
    for prof in seg_summary["profiles"]:
        print(f"  {prof['persona']:22s} n={prof['n']:5d}  engagement={prof['avg_engagement']}")

    step(6, "Campaign performance + A/B test")
    camp_summary = run_campaigns(dfs["campaigns"], dfs["ab_test"], p["outputs"])
    print(f"  blended ROAS: {camp_summary['blended_roas']}")
    ab = camp_summary["ab_test"]
    print(f"  A/B: lift {ab['lift_relative']:+.1%}  p={ab['p_value']:.2e}  "
          f"-> {ab['verdict']}")

    step(7, "Building RAG index over knowledge base")
    rag = build_rag(p["kb"], top_k=cfg["rag"]["top_k"],
                    embedding_backend=cfg["rag"]["embedding_backend"])
    print(f"  chunks indexed: {len(rag.store.chunks)}  "
          f"embedding: {rag.store.backend}")
    smoke = rag.answer("What is the engagement score formula?")
    print(f"  smoke answer cites: {[s['doc_id'] for s in smoke.sources]}")

    step(8, "Smoke-testing LangGraph agent")
    app = build_agent(p["outputs"], p["data"] / "demo.db", rag,
                      max_attempts=cfg["agent"]["max_attempts"])
    smoke_agent = run_agent(app, "Which campaigns have the best ROAS and what should we do?")
    print(f"  agent status: {smoke_agent['status']}  "
          f"nodes: {[t['node'] for t in smoke_agent['trace']]}")

    step(9, "Running evaluation suite")
    model_eval = run_model_evaluation(feature_table, feature_cols, p["outputs"],
                                      min_auc=cfg["evaluation"]["min_auc"],
                                      test_months=cfg["model"]["test_months"])
    print(f"  model  : CV AUC {model_eval['cv_auc_mean']} +/- {model_eval['cv_auc_std']}"
          f"  floor passed: {model_eval['auc_floor_passed']}")
    rag_eval = run_rag_evaluation(rag, p["outputs"], k=3,
                                  min_hit_rate=cfg["evaluation"]["min_hit_rate"],
                                  min_mrr=cfg["evaluation"]["min_mrr"])
    print(f"  rag    : hit-rate@3 {rag_eval['hit_rate_at_k']}  "
          f"P@1 {rag_eval['precision_at_1']}  MRR {rag_eval['mrr']}  "
          f"floors passed: {rag_eval['hit_rate_floor_passed'] and rag_eval['mrr_floor_passed']}")
    agent_eval = run_agent_evaluation(app, p["outputs"])
    print(f"  agent  : reliability {agent_eval['workflow_reliability']}  "
          f"deterministic: {agent_eval['deterministic_repeat_run']}")
    privacy = run_privacy_scan(p["data"], p["outputs"], p["outputs"])
    print(f"  privacy: scanned {privacy['files_scanned']} files  "
          f"clean: {privacy['scan_clean']}")

    step(10, "Writing run summary")
    summary = {
        "config": {"seed": seed, "demo_mode": cfg["project"]["demo_mode"]},
        "data": {name: list(df.shape) for name, df in dfs.items()},
        "db_rows": counts,
        "model": model_summary,
        "segmentation": {"k": seg_summary["k"],
                         "personas": seg_summary["persona_names"]},
        "campaigns": {k: v for k, v in camp_summary.items() if k != "ab_test"},
        "ab_test": camp_summary["ab_test"],
        "rag": {"chunks": len(rag.store.chunks), "backend": rag.store.backend},
        "agent_smoke_status": smoke_agent["status"],
        "evaluation": {
            "model": {k: model_eval[k] for k in
                      ("cv_auc_mean", "cv_auc_std", "test_roc_auc", "brier_score",
                       "auc_floor_passed")},
            "rag": {k: rag_eval[k] for k in
                    ("hit_rate_at_k", "precision_at_1", "mrr", "all_answers_cited",
                     "hit_rate_floor_passed", "mrr_floor_passed")},
            "agent": {k: agent_eval[k] for k in
                      ("workflow_reliability", "validation_pass_rate",
                       "deterministic_repeat_run")},
            "privacy": {k: privacy[k] for k in
                        ("files_scanned", "scan_clean", "masking_utility_works")},
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(p["outputs"] / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"  elapsed: {summary['elapsed_seconds']}s")
    print(f"  outputs: {p['outputs']}")
    print(f"  dashboard: streamlit run src/dashboard/app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
