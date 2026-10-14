"""Engagement-decline prediction models (scikit-learn).

Trains Logistic Regression + Random Forest, evaluates on a held-out time
window, persists metrics + model artifacts, and scores every cardholder's
latest month for the at-risk list used by the dashboard and the agent.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import split_time


def train_models(X_train, y_train, random_state: int = 42) -> dict:
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=random_state)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, class_weight="balanced",
            random_state=random_state, n_jobs=-1),
    }
    for m in models.values():
        m.fit(X_train, y_train)
    return models


def evaluate_models(models: dict, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    rows, curves = [], {}
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, proba)
        curves[name] = (fpr, tpr)
        rows.append({
            "model": name,
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, pred, zero_division=0), 4),
            "roc_auc_ci_note": "single split; see evaluation/model_eval.py for CV",
        })
    return pd.DataFrame(rows), curves


def plot_roc(curves: dict, out_path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, lw=2, label=name.replace("_", " ").title())
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Engagement-decline prediction — ROC (held-out time window)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_feature_importance(model, feature_cols, out_path, top_n=15) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:  # logistic regression — absolute standardized coefficients
        imp = np.abs(model.named_steps["clf"].coef_[0])
    fi = (pd.DataFrame({"feature": feature_cols, "importance": imp})
          .sort_values("importance", ascending=False).head(top_n))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="#1f77b4")
    ax.set_title("Top feature importance — engagement-decline model")
    ax.set_xlabel("importance"); fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)
    return fi


def plot_risk_distribution(scores: pd.Series, out_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=40, color="#d62728", alpha=0.75, edgecolor="white")
    ax.axvline(0.5, color="k", ls="--", lw=1, label="decision threshold 0.5")
    ax.set_xlabel("predicted decline probability"); ax.set_ylabel("cardholders")
    ax.set_title("At-risk distribution — latest month scores")
    ax.legend(); fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def run_modeling(feature_table: pd.DataFrame, feature_cols: list[str], outputs_dir,
                 random_state: int = 42, test_months: int = 6) -> dict:
    outputs_dir = Path(outputs_dir); outputs_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, meta_test = split_time(
        feature_table, feature_cols, test_months=test_months)

    models = train_models(X_train, y_train, random_state=random_state)
    metrics_df, curves = evaluate_models(models, X_test, y_test)

    best_name = metrics_df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    best = models[best_name]

    plot_roc(curves, outputs_dir / "roc_curves.png")
    fi = plot_feature_importance(best, feature_cols, outputs_dir / "feature_importance.png")

    # ---- score the LATEST month of every cardholder -> at-risk list ----------
    latest_idx = feature_table.groupby("cardholder_id")["month_index"].idxmax()
    latest = feature_table.loc[latest_idx]
    latest = latest.copy()
    latest["decline_probability"] = best.predict_proba(latest[feature_cols])[:, 1]
    latest["risk_band"] = pd.cut(latest["decline_probability"],
                                 [-0.01, 0.25, 0.5, 1.0],
                                 labels=["Low", "Medium", "High"])
    at_risk = (latest.sort_values("decline_probability", ascending=False)
               .head(100)[["cardholder_id", "month", "decline_probability", "risk_band",
                          "engagement_score", "spend", "txn_count"]]
               .round(4))
    at_risk.to_csv(outputs_dir / "at_risk_cardholders.csv", index=False)
    latest[["cardholder_id", "month", "decline_probability"]].round(4).to_csv(
        outputs_dir / "latest_risk_scores.csv", index=False)
    plot_risk_distribution(latest["decline_probability"],
                           outputs_dir / "risk_distribution.png")

    metrics_df.to_csv(outputs_dir / "model_metrics.csv", index=False)
    fi.to_csv(outputs_dir / "feature_importance.csv", index=False)

    # persist the full pipeline (scaler + model) for reuse by agent/dashboard
    joblib.dump({"model": best, "feature_cols": feature_cols, "model_name": best_name},
                outputs_dir / "engagement_decline_model.joblib")

    cm = confusion_matrix(y_test, (best.predict_proba(X_test)[:, 1] >= 0.5).astype(int))
    summary = {
        "best_model": best_name,
        "test_rows": int(len(y_test)),
        "test_decline_rate": round(float(y_test.mean()), 4),
        "confusion_matrix": cm.tolist(),
        "metrics": metrics_df.to_dict(orient="records"),
        "n_features": len(feature_cols),
        "train_rows": int(len(y_train)),
    }
    with open(outputs_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
