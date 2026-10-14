"""LangGraph agent: analyze → retrieve → recommend → validate (with retry).

State flows through four nodes. Each node uses one of the agent tools.
After validation, a conditional edge either ends the run or retries
(re-retrieval) up to `max_attempts` times.
"""
from __future__ import annotations

import json
from typing import Annotated, TypedDict

import operator
from langgraph.graph import END, StateGraph

from src.agent import tools as T


class AgentState(TypedDict, total=False):
    question: str
    analysis: dict
    retrieved: list
    recommendation: str
    validation: dict
    attempts: int
    trace: Annotated[list, operator.add]
    status: str


def _analyze_node(state: AgentState) -> dict:
    """Pick the right analytics query from the question (deterministic router)."""
    q = state["question"].lower()
    if any(w in q for w in ("campaign", "roas", "channel", "budget", "spend roi")):
        query = "campaign_performance"
    elif any(w in q for w in ("segment", "persona", "cluster", "smb", "mid-market", "large")):
        query = "segment_summary"
    elif any(w in q for w in ("decline", "risk", "churn", "retention", "win-back", "engagement drop")):
        query = "decline_risk_summary"
    elif any(w in q for w in ("a/b", "ab test", "test result", "experiment")):
        query = "ab_test"
    elif any(w in q for w in ("channel", "email", "in-app", "statement", "mail")):
        query = "channel_rollup"
    else:
        query = "kpis"
    raw = T.analytics_tool.invoke({"query": query})
    return {
        "analysis": json.loads(raw),
        "trace": [{"node": "analyze", "detail": f"analytics query: {query}"}],
    }


def _retrieve_node(state: AgentState) -> dict:
    raw = T.retrieval_tool.invoke({"question": state["question"]})
    return {
        "retrieved": json.loads(raw),
        "trace": [{"node": "retrieve",
                   "detail": "docs: " + ", ".join(d["doc_id"] for d in json.loads(raw))}],
    }


def _recommend_node(state: AgentState) -> dict:
    rec = T.recommendation_tool.invoke({
        "question": state["question"],
        "analysis_json": json.dumps(state["analysis"], default=str),
        "retrieved_json": json.dumps(state["retrieved"], default=str),
    })
    return {"recommendation": rec,
            "trace": [{"node": "recommend", "detail": f"{len(rec)} chars composed"}]}


def _validate_node(state: AgentState) -> dict:
    raw = T.validator_tool.invoke({
        "recommendation": state["recommendation"],
        "analysis_json": json.dumps(state["analysis"], default=str),
    })
    validation = json.loads(raw)
    attempts = state.get("attempts", 0) + 1
    return {
        "validation": validation,
        "attempts": attempts,
        "trace": [{"node": "validate",
                   "detail": "passed" if validation["passed"]
                             else "issues: " + "; ".join(validation["issues"])}],
    }


def _route_after_validation(state: AgentState) -> str:
    if state["validation"]["passed"]:
        return "pass"
    if state.get("attempts", 0) < _MAX_ATTEMPTS:
        return "retry"
    return "give_up"


_MAX_ATTEMPTS = 2


def build_agent(outputs_dir, db_path, rag, max_attempts: int = 2) -> StateGraph:
    """Wire tools + nodes into a compiled LangGraph application."""
    global _MAX_ATTEMPTS
    _MAX_ATTEMPTS = max_attempts
    T.configure_tools(outputs_dir, db_path, rag)

    g = StateGraph(AgentState)
    g.add_node("analyze", _analyze_node)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("recommend", _recommend_node)
    g.add_node("validate", _validate_node)
    g.set_entry_point("analyze")
    g.add_edge("analyze", "retrieve")
    g.add_edge("retrieve", "recommend")
    g.add_edge("recommend", "validate")
    g.add_conditional_edges("validate", _route_after_validation,
                            {"pass": END, "retry": "retrieve", "give_up": END})
    return g.compile()


def run_agent(app, question: str) -> dict:
    """Run the compiled graph on a business question; returns the final state."""
    final = app.invoke({"question": question, "attempts": 0,
                        "trace": [], "status": "running"})
    passed = final.get("validation", {}).get("passed", False)
    final["status"] = "validated" if passed else "needs_review"
    return final


if __name__ == "__main__":
    from src.config import load_config, paths
    from src.rag.rag import build_rag
    cfg = load_config(); p = paths(cfg)
    rag = build_rag(p["kb"], top_k=cfg["rag"]["top_k"])
    app = build_agent(p["outputs"], p["data"] / "demo.db", rag,
                      max_attempts=cfg["agent"]["max_attempts"])
    out = run_agent(app, "Which campaigns should we scale and which should we pause?")
    print("status:", out["status"])
    print(out["recommendation"])
