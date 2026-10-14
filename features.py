"""Feature engineering for the engagement-decline model.

Features use ONLY information available at month t (no look-ahead leakage):
current-month activity, trailing 3-month averages, momentum, and static
cardholder attributes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "logins", "app_sessions", "email_clicks", "spend", "txn_count", "engagement_score",
    "logins_3m_avg", "spend_3m_avg", "score_3m_avg",
    "logins_mom", "spend_mom", "score_mom",
    "logins_per_txn", "spend_per_txn",
    "tenure_months", "employees", "annual_spend_baseline",
    "month_index", "calendar_month",
    "is_smb", "is_midmarket", "is_large",
]
CATEGORICAL = ["industry", "region"]


def build_feature_table(engagement: pd.DataFrame, cardholders: pd.DataFrame) -> pd.DataFrame:
    eng = engagement.sort_values(["cardholder_id", "month_index"]).copy()
    g = eng.groupby("cardholder_id")

    # trailing 3-month averages (current month included, past only)
    for col, name in [("logins", "logins_3m_avg"), ("spend", "spend_3m_avg"),
                      ("engagement_score", "score_3m_avg")]:
        eng[name] = g[col].transform(lambda s: s.rolling(3, min_periods=1).mean())

    # momentum vs trailing average
    eng["logins_mom"] = eng["logins"] - eng["logins_3m_avg"]
    eng["spend_mom"] = eng["spend"] - eng["spend_3m_avg"]
    eng["score_mom"] = eng["engagement_score"] - eng["score_3m_avg"]

    eng["logins_per_txn"] = eng["logins"] / eng["txn_count"].clip(lower=1)
    eng["spend_per_txn"] = eng["spend"] / eng["txn_count"].clip(lower=1)

    eng = eng.merge(
        cardholders[["cardholder_id", "industry", "region", "segment",
                     "employees", "tenure_months", "annual_spend_baseline"]],
        on="cardholder_id", how="left",
    )
    eng["calendar_month"] = pd.PeriodIndex(eng["month"], freq="M").month
    eng["is_smb"] = (eng["segment"] == "SMB").astype(int)
    eng["is_midmarket"] = (eng["segment"] == "Mid-Market").astype(int)
    eng["is_large"] = (eng["segment"] == "Large").astype(int)

    eng = pd.get_dummies(eng, columns=CATEGORICAL, prefix=CATEGORICAL)
    bool_cols = [c for c in eng.columns if c.startswith(("industry_", "region_"))]
    eng[bool_cols] = eng[bool_cols].astype(int)

    feature_cols = FEATURE_COLS + bool_cols
    # NB: month_index is already in FEATURE_COLS — dedupe to avoid duplicated
    # column labels (breaks boolean indexing downstream). "segment" is kept
    # for the fairness evaluation (NOT used as a model feature).
    keep = list(dict.fromkeys(
        ["cardholder_id", "month", "month_index", "engagement_decline", "segment"]
        + feature_cols))
    out = eng[keep].copy()
    out = out[out["engagement_decline"] >= 0]  # drop non-evaluable rows (no t+3)
    return out.reset_index(drop=True), feature_cols


def split_time(feature_table: pd.DataFrame, feature_cols: list[str],
               test_months: int = 6):
    """Time-based split: train on the earliest months, test on the last
    `test_months` — realistic for a deployment scenario."""
    max_idx = feature_table["month_index"].max()
    test = feature_table[feature_table["month_index"] > max_idx - test_months]
    train = feature_table[feature_table["month_index"] <= max_idx - test_months]
    X_train, y_train = train[feature_cols], train["engagement_decline"]
    X_test, y_test = test[feature_cols], test["engagement_decline"]
    meta_test = test[["cardholder_id", "month", "month_index"]]
    return X_train, X_test, y_train, y_test, meta_test
