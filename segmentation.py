"""Customer segmentation — RFM + engagement features, K-Means personas."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PERSONA_NAMES = {
    0: "High-Value Engaged",
    1: "Steady Core",
    2: "Growing Mid-Market",
    3: "At-Risk High-Spend",
    4: "Low-Engagement SMB",
}


def build_rfm(transactions: pd.DataFrame, engagement: pd.DataFrame) -> pd.DataFrame:
    last_date = pd.Timestamp(transactions["txn_date"].max())
    rfm = transactions.groupby("cardholder_id").agg(
        recency_days=("txn_date", lambda s: (last_date - pd.Timestamp(s.max())).days),
        frequency=("txn_id", "count"),
        monetary=("amount", "sum"),
    ).reset_index()
    eng = engagement.groupby("cardholder_id").agg(
        avg_engagement=("engagement_score", "mean"),
        avg_monthly_spend=("spend", "mean"),
        months_since_login_proxy=("month_index", "count"),
    ).reset_index()
    rfm = rfm.merge(eng, on="cardholder_id", how="left")
    rfm["recency_days"] = rfm["recency_days"].clip(lower=0)
    return rfm


def run_segmentation(transactions: pd.DataFrame, engagement: pd.DataFrame,
                     cardholders: pd.DataFrame, outputs_dir, k: int = 5,
                     random_state: int = 42) -> dict:
    outputs_dir = Path(outputs_dir); outputs_dir.mkdir(parents=True, exist_ok=True)
    rfm = build_rfm(transactions, engagement)

    feat = rfm[["recency_days", "frequency", "monetary", "avg_engagement"]].copy()
    feat["monetary"] = np.log1p(feat["monetary"])
    feat["frequency"] = np.log1p(feat["frequency"])
    X = StandardScaler().fit_transform(feat)

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    rfm["cluster"] = km.fit_predict(X)

    # name clusters by size-ordered value: rank on monetary+engagement
    prof = rfm.groupby("cluster").agg(
        n=("cardholder_id", "count"),
        monetary=("monetary", "mean"), frequency=("frequency", "mean"),
        recency_days=("recency_days", "mean"), avg_engagement=("avg_engagement", "mean"),
    ).round(2)
    # deterministic persona naming: highest (monetary+engagement) = High-Value Engaged
    order = (prof["monetary"] * 0.5 + prof["avg_engagement"] * 0.5)
    at_risk_mask = (prof["recency_days"] > prof["recency_days"].median()) & \
                   (prof["avg_engagement"] < prof["avg_engagement"].median())
    names = {}
    remaining = list(prof.index)
    # assign "At-Risk High-Spend" = high spend, high recency, low engagement
    if at_risk_mask.any():
        ar = prof[at_risk_mask].sort_values("monetary", ascending=False).index[0]
        names[ar] = "At-Risk High-Spend"; remaining.remove(ar)
    remaining.sort(key=lambda c: -order[c])
    pool = ["High-Value Engaged", "Growing Mid-Market", "Steady Core", "Low-Engagement SMB"]
    for c, nm in zip(remaining, pool):
        names[c] = nm
    rfm["persona"] = rfm["cluster"].map(names)

    rfm = rfm.merge(cardholders[["cardholder_id", "segment", "industry", "region"]],
                    on="cardholder_id", how="left")
    rfm.round(3).to_csv(outputs_dir / "cardholder_segments.csv", index=False)
    prof["persona"] = [names[c] for c in prof.index]
    prof.to_csv(outputs_dir / "segment_profiles.csv")

    # PCA scatter
    pca = PCA(n_components=2, random_state=random_state)
    pts = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in sorted(rfm["cluster"].unique()):
        m = rfm["cluster"].to_numpy() == c
        ax.scatter(pts[m, 0], pts[m, 1], s=8, alpha=0.5, label=names[c])
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title("Cardholder personas — K-Means on RFM + engagement (PCA view)")
    ax.legend(markerscale=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outputs_dir / "segment_pca.png", dpi=130); plt.close(fig)

    summary = {
        "k": k,
        "n_cardholders": int(len(rfm)),
        "inertia": round(float(km.inertia_), 2),
        "profiles": prof.reset_index().to_dict(orient="records"),
        "persona_names": {int(c): n for c, n in names.items()},
    }
    with open(outputs_dir / "segmentation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
