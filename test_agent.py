"""Tests for the LangGraph agent: tools, workflow, validation, privacy."""
from __future__ import annotations

import json
import re

import pytest


@pytest.fixture(scope="module")
def agent_setup(mini_outputs):
    from pathlib import Path
    from src.rag.rag import build_rag
    from src.agent.graph import build_agent
    root = Path(__file__).resolve().parents[1]
    rag = build_rag(root / "kb", top_k=4)
    app = build_agent(mini_outputs["outputs"], mini_outputs["db"], rag, max_attempts=2)
    return {"app": app, "rag": rag, "ctx": mini_outputs}


def test_analytics_tool_whitelist(agent_setup):
    from src.agent import tools as T
    T.configure_tools(agent_setup["ctx"]["outputs"], agent_setup["ctx"]["db"],
                      agent_setup["rag"])
    out = json.loads(T.analytics_tool.invoke({"query": "kpis"}))
    assert out["cardholders"] > 0 and out["transactions"] > 0
    with pytest.raises(Exception):
        T.analytics_tool.invoke({"query": "DROP TABLE cardholders"})


def test_retrieval_tool(agent_setup):
    from src.agent import tools as T
    T.configure_tools(agent_setup["ctx"]["outputs"], agent_setup["ctx"]["db"],
                      agent_setup["rag"])
    out = json.loads(T.retrieval_tool.invoke(
        {"question": "ROAS benchmarks by channel"}))
    assert len(out) == 3
    assert any(d["doc_id"] == "KB-10" for d in out)


def test_validator_catches_bad_recommendation(agent_setup):
    from src.agent import tools as T
    T.configure_tools(agent_setup["ctx"]["outputs"], agent_setup["ctx"]["db"],
                      agent_setup["rag"])
    bad = json.loads(T.validator_tool.invoke(
        {"recommendation": "short, no citations, no numbers",
         "analysis_json": json.dumps({"roas": 4.5})}))
    assert bad["passed"] is False and len(bad["issues"]) >= 2
    pan = json.loads(T.validator_tool.invoke(
        {"recommendation": "Call cardholder 4111111111111111 now [KB-01] roas 4.5",
         "analysis_json": json.dumps({"roas": 4.5})}))
    assert pan["passed"] is False
    assert any("PAN" in i for i in pan["issues"])


def test_agent_end_to_end_validated(agent_setup):
    from src.agent.graph import run_agent
    result = run_agent(agent_setup["app"],
                       "Which campaigns have the best ROAS and what should we do?")
    nodes = [t["node"] for t in result["trace"]]
    assert all(n in nodes for n in ("analyze", "retrieve", "recommend", "validate"))
    assert result["status"] == "validated"
    assert result["validation"]["passed"] is True
    assert re.search(r"\[KB-\d+\]", result["recommendation"])


def test_agent_deterministic(agent_setup):
    from src.agent.graph import run_agent
    q = "How many cardholders are at high risk of engagement decline?"
    r1 = run_agent(agent_setup["app"], q)["recommendation"]
    r2 = run_agent(agent_setup["app"], q)["recommendation"]
    assert r1 == r2
