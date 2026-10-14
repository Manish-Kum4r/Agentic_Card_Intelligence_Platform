"""Tests for feature engineering + the engagement-decline ML model."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import split_time


def test_feature_table_no_leakage_columns(feature_table):
    ft, cols = feature_table
    # the label and its look-ahead must not be features
    for bad in ("engagement_decline", "score_fwd3"):
        assert bad not in cols
    assert len(cols) >= 20


def test_features_complete(feature_table):
    ft, cols = feature_table
    assert not ft[cols].isna().any().any(), "NaNs in features"
    assert np.isfinite(ft[cols].to_numpy()).all()


def test_time_split_is_ordered(feature_table):
    ft, cols = feature_table
    X_train, X_test, y_train, y_test, meta = split_time(ft, cols, test_months=3)
    assert meta["month_index"].min() > ft.loc[X_train.index, "month_index"].max()
    assert len(X_train) > 0 and len(X_test) > 0


def test_model_beats_random(feature_table):
    """The model must genuinely predict decline (honest floor, real data)."""
    ft, cols = feature_table
    X_train, X_test, y_train, y_test, _ = split_time(ft, cols, test_months=3)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000,
                                                class_weight="balanced"))])
    pipe.fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    assert auc > 0.65, f"AUC too low: {auc:.3f} — target signal broken?"


def test_feature_importance_direction(feature_table):
    """Sanity: declining momentum should be among the strongest signals."""
    ft, cols = feature_table
    X_train, X_test, y_train, y_test, _ = split_time(ft, cols, test_months=3)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000,
                                                class_weight="balanced"))])
    pipe.fit(X_train, y_train)
    coefs = pipe.named_steps["clf"].coef_[0]
    top = sorted(zip(cols, coefs), key=lambda x: -abs(x[1]))[:5]
    top_names = {c for c, _ in top}
    assert top_names & {"score_mom", "logins_mom", "engagement_score",
                        "logins_3m_avg", "score_3m_avg"}, \
        f"momentum features should dominate, got {top_names}"
