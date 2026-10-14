"""Shared test fixtures: small synthetic datasets + a miniature outputs dir."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation import generate_all  # noqa: E402
from src.db import SQLITE_SCHEMA  # noqa: E402


@pytest.fixture(scope="session")
def small_data(tmp_path_factory):
    """Small but real synthetic dataset (fast to generate)."""
    d = tmp_path_factory.mktemp("data")
    return generate_all(d, seed=7, n_cardholders=400, months=12,
                        start_month="2024-01", n_transactions=4000)


@pytest.fixture(scope="session")
def feature_table(small_data):
    from src.features import build_feature_table
    ft, cols = build_feature_table(small_data["engagement"], small_data["cardholders"])
    return ft, cols


@pytest.fixture(scope="session")
def mini_outputs(tmp_path_factory, small_data):
    """A miniature outputs/ + demo.db so the agent can run without the full
    pipeline (keeps the test-suite fast while exercising the real code path)."""
    tmp_path = tmp_path_factory.mktemp("mini")
    out = tmp_path / "outputs"
    (out / "evaluation").mkdir(parents=True)
    db = tmp_path / "demo.db"

    conn = sqlite3.connect(db)
    conn.executescript(SQLITE_SCHEMA)
    small_data["cardholders"].head(50).to_sql("cardholders", conn, if_exists="append", index=False)
    small_data["engagement"].head(500).to_sql("engagement_monthly", conn, if_exists="append", index=False)
    small_data["transactions"].head(500).to_sql("transactions", conn, if_exists="append", index=False)
    small_data["campaigns"].to_sql("campaigns", conn, if_exists="append", index=False)
    small_data["ab_test"].to_sql("ab_test_daily", conn, if_exists="append", index=False)
    conn.commit(); conn.close()

    from src.campaigns import campaign_performance
    perf = campaign_performance(small_data["campaigns"])
    perf.round(4).to_csv(out / "campaign_performance.csv", index=False)
    chan = perf.groupby("channel").agg(budget=("budget", "sum"),
                                      revenue=("attributed_revenue", "sum"),
                                      conversions=("conversions", "sum"))
    chan["roas"] = (chan["revenue"] / chan["budget"]).round(2)
    chan.to_csv(out / "channel_rollup.csv")

    risk = pd.DataFrame({"cardholder_id": range(1, 51),
                         "month": ["2024-12"] * 50,
                         "decline_probability": [0.1 * (i % 10) for i in range(50)]})
    risk.to_csv(out / "latest_risk_scores.csv", index=False)
    risk.assign(risk_band="High").to_csv(out / "at_risk_cardholders.csv", index=False)

    with open(out / "model_summary.json", "w") as f:
        json.dump({"best_model": "logistic_regression",
                   "metrics": [{"model": "logistic_regression", "roc_auc": 0.81,
                                "accuracy": 0.75, "precision": 0.6, "recall": 0.7,
                                "f1": 0.65}]}, f)
    with open(out / "segmentation_summary.json", "w") as f:
        json.dump({"k": 5, "profiles": [{"persona": "Steady Core", "n": 100,
                                         "monetary": 50000, "frequency": 30,
                                         "recency_days": 5, "avg_engagement": 55}]}, f)
    with open(out / "ab_test_result.json", "w") as f:
        json.dump({"test_name": "t", "lift_relative": 0.08, "p_value": 1e-9,
                   "significant_at_5pct": True,
                   "verdict": "Treatment wins — roll out to 100% of the audience."}, f)
    return {"outputs": out, "db": db, "data": small_data}
