"""Agent tools (LangChain @tool) + analytics context.

Four tools, all offline and deterministic:
  1. analytics_tool      — whitelisted analytics queries over the pipeline outputs
  2. retrieval_tool      — semantic retrieval over the marketing knowledge base
  3. recommendation_tool — composes a grounded recommendation (analytics + KB)
  4. validator_tool      — checks the recommendation before it is returned
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from src.evaluation.privacy import PAN_REGEX
from src.rag.llm import LocalLLM
from src.rag.rag import RAGPipeline

ALLOWED_QUERIES = ("kpis", "campaign_performance", "channel_rollup",
                   "segment_summary", "decline_risk_summary", "ab_test")


class AnalyticsContext:
    """Read-only view over the pipeline outputs + SQLite demo DB."""

    def __init__(self, outputs_dir, db_path):
        self.outputs = Path(outputs_dir)
        self.db_path = Path(db_path)

    def _load_json(self, name):
        with open(self.outputs / name, encoding="utf-8") as f:
            return json.load(f)

    def run(self, query: str) -> dict:
        if query not in ALLOWED_QUERIES:
            raise ValueError(f"query '{query}' not in whitelist {ALLOWED_QUERIES}")
        if query == "kpis":
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            k = {
                "cardholders": conn.execute("SELECT COUNT(*) FROM cardholders").fetchone()[0],
                "transactions": conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0],
                "gmv": round(conn.execute("SELECT SUM(amount) FROM transactions").fetchone()[0], 2),
                "avg_engagement_score": round(conn.execute(
                    "SELECT AVG(engagement_score) FROM engagement_monthly").fetchone()[0], 2),
                "campaigns": conn.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0],
            }
            conn.close()
            risk = pd.read_csv(self.outputs / "latest_risk_scores.csv")
            k["high_risk_cardholders"] = int((risk["decline_probability"] >= 0.5).sum())
            k["avg_decline_probability"] = round(float(risk["decline_probability"].mean()), 4)
            return k
        if query == "campaign_performance":
            perf = pd.read_csv(self.outputs / "campaign_performance.csv")
            return {
                "blended_roas": round(float(perf["attributed_revenue"].sum()
                                            / perf["budget"].sum()), 2),
                "total_budget": float(perf["budget"].sum()),
                "total_attributed_revenue": float(perf["attributed_revenue"].sum()),
                "top3_by_roas": perf.head(3)[
                    ["campaign_name", "channel", "roas", "ctr", "cvr"]].to_dict("records"),
                "bottom3_by_roas": perf.tail(3)[
                    ["campaign_name", "channel", "roas"]].to_dict("records"),
            }
        if query == "channel_rollup":
            return pd.read_csv(self.outputs / "channel_rollup.csv").to_dict("records")
        if query == "segment_summary":
            return self._load_json("segmentation_summary.json")["profiles"]
        if query == "decline_risk_summary":
            model = self._load_json("model_summary.json")
            at_risk = pd.read_csv(self.outputs / "at_risk_cardholders.csv")
            return {
                "best_model": model["best_model"],
                "roc_auc": max(m["roc_auc"] for m in model["metrics"]),
                "high_risk_count": int((at_risk["risk_band"] == "High").sum()),
                "top5_at_risk": at_risk.head(5).to_dict("records"),
            }
        if query == "ab_test":
            ab = self._load_json("ab_test_result.json")
            return {"test_name": ab["test_name"],
                    "lift_relative": ab["lift_relative"],
                    "p_value": f"{ab['p_value']:.2e}",
                    "significant_at_5pct": ab["significant_at_5pct"],
                    "verdict": ab["verdict"]}
        raise ValueError(query)


# ---------------------------------------------------------------------------
# module-level wiring (set once by build_tools / the graph builder)
# ---------------------------------------------------------------------------
_ANALYTICS: AnalyticsContext | None = None
_RAG: RAGPipeline | None = None
_LLM = LocalLLM()


def configure_tools(outputs_dir, db_path, rag: RAGPipeline) -> None:
    global _ANALYTICS, _RAG
    _ANALYTICS = AnalyticsContext(outputs_dir, db_path)
    _RAG = rag


@tool
def analytics_tool(query: str) -> str:
    """Run a whitelisted analytics query over pipeline outputs.

    Allowed queries: kpis, campaign_performance, channel_rollup,
    segment_summary, decline_risk_summary, ab_test. Returns JSON.
    """
    return json.dumps(_ANALYTICS.run(query), default=str)


@tool
def retrieval_tool(question: str) -> str:
    """Retrieve the most relevant marketing knowledge-base passages for a
    question. Returns JSON: list of {doc_id, title, score, snippet}."""
    results = _RAG.retrieve(question, k=3)
    return json.dumps([{"doc_id": r.chunk.doc_id, "title": r.chunk.doc_title,
                        "score": round(r.score, 4), "snippet": r.chunk.text[:300]}
                       for r in results], default=str)


@tool
def recommendation_tool(question: str, analysis_json: str, retrieved_json: str) -> str:
    """Compose a recommendation grounded in analytics findings + KB passages.

    analysis_json: JSON string from analytics_tool.
    retrieved_json: JSON string from retrieval_tool.
    """
    analysis = json.loads(analysis_json)
    retrieved = json.loads(retrieved_json)
    # adapt retrieval_tool's compact dicts to what the composer expects
    class _R:  # minimal stand-in with .chunk/.score
        def __init__(self, d):
            self.score = d["score"]
            self.chunk = type("C", (), {"doc_id": d["doc_id"], "doc_title": d["title"],
                                        "text": d["snippet"]})()
    results = [_R(d) for d in retrieved]
    return _LLM.compose_recommendation(question, analysis, results)


@tool
def validator_tool(recommendation: str, analysis_json: str) -> str:
    """Validate a recommendation before returning it to the user.

    Checks: non-empty, cites at least one KB source [KB-xx], quotes at least
    one figure that matches the analytics context, contains no PAN-like
    number. Returns JSON {passed, issues[]}.
    """
    issues = []
    if not recommendation or len(recommendation.strip()) < 80:
        issues.append("recommendation too short or empty")
    if not re.search(r"\[KB-\d+\]", recommendation):
        issues.append("no knowledge-base citation [KB-xx] found")
    analysis = json.loads(analysis_json)
    flat = json.dumps(analysis, default=str)
    nums_in_analysis = set(re.findall(r"\d+\.?\d*", flat))
    nums_in_rec = set(re.findall(r"\d+\.?\d*", recommendation))
    if not (nums_in_analysis & nums_in_rec):
        issues.append("no figure from the analytics context appears in the recommendation")
    if PAN_REGEX.search(recommendation):
        issues.append("PAN-like digit sequence detected (privacy violation)")
    return json.dumps({"passed": not issues, "issues": issues})
