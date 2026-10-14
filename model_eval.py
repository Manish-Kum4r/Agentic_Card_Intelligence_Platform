"""Model evaluation & responsible-AI checks.

- 5-fold stratified cross-validation on the training window (honest
  generalization estimate beyond the single time split)
- Calibration (Brier score)
- Fairness: predicted-positive-rate gap across company segments
- Threshold assertion against config (min_auc floor)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_model_evaluation(feature_table: pd.DataFrame, feature_cols: list[str],
                         outputs_dir, min_auc: float = 0.65,
                         test_months: int = 6) -> dict:
    out = Path(outputs_dir) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    from src.features import split_time
    X_train, X_test, y_train, y_test, meta_test = split_time(
        feature_table, feature_cols, test_months=test_months)

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    pipe.fit(X_train, y_train)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, proba_test)
    brier = brier_score_loss(y_test, proba_test)

    # fairness: predicted positive rate by company segment
    seg = feature_table.loc[X_test.index, "segment"] if "segment" in feature_table.columns else None
    fair = {}
    if seg is not None:
        pred = (proba_test >= 0.5).astype(int)
        tmp = pd.DataFrame({"segment": seg.to_numpy(), "pred": pred,
                            "proba": proba_test})
        fair = tmp.groupby("segment").agg(
            n=("pred", "size"),
            predicted_positive_rate=("pred", "mean"),
            mean_probability=("proba", "mean")).round(4).to_dict("index")
        rates = [v["predicted_positive_rate"] for v in fair.values()]
        fair["_max_pp_rate_gap"] = round(float(max(rates) - min(rates)), 4)

    with open(Path(outputs_dir) / "model_summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    report = {
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_auc_std": round(float(cv_auc.std()), 4),
        "cv_auc_folds": [round(float(a), 4) for a in cv_auc],
        "test_roc_auc": round(float(test_auc), 4),
        "brier_score": round(float(brier), 4),
        "min_auc_floor": min_auc,
        "auc_floor_passed": bool(test_auc >= min_auc and cv_auc.mean() >= min_auc),
        "fairness_by_segment": fair,
        "best_model_from_pipeline": summary["best_model"],
    }
    with open(out / "model_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
