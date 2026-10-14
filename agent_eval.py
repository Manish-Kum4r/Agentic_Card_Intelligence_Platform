"""Agent evaluation: workflow reliability, validation pass rate, determinism."""
from __future__ import annotations

import json
from pathlib import Path

AGENT_QUESTIONS = [
    "Which campaigns should we scale up and which should we pause based on ROAS?",
    "How many cardholders are at high risk of engagement decline and who are the top 5?",
    "What should we do about the at-risk high-spend segment?",
    "Did the Q4 Cashback Boost A/B test win, and what should we do next?",
    "How should we allocate next quarter's marketing budget?",
]


def run_agent_evaluation(app, outputs_dir) -> dict:
    from src.agent.graph import run_agent
    out = Path(outputs_dir) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for q in AGENT_QUESTIONS:
        result = run_agent(app, q)
        nodes = [t["node"] for t in result.get("trace", [])]
        rows.append({
            "question": q,
            "status": result["status"],
            "validation_passed": result.get("validation", {}).get("passed", False),
            "attempts": result.get("attempts", 0),
            "nodes_executed": nodes,
            "all_four_nodes_ran": all(n in nodes for n in
                                      ("analyze", "retrieve", "recommend", "validate")),
            "recommendation_chars": len(result.get("recommendation", "")),
            "cited_kb": "[KB-" in result.get("recommendation", ""),
        })
    # determinism: same question twice must give the same recommendation
    q0 = AGENT_QUESTIONS[0]
    r1 = run_agent(app, q0)["recommendation"]
    r2 = run_agent(app, q0)["recommendation"]
    deterministic = (r1 == r2)

    n = len(rows)
    report = {
        "n_questions": n,
        "validation_pass_rate": round(sum(r["validation_passed"] for r in rows) / n, 4),
        "workflow_complete_rate": round(sum(r["all_four_nodes_ran"] for r in rows) / n, 4),
        "all_cited_kb": all(r["cited_kb"] for r in rows),
        "deterministic_repeat_run": deterministic,
        "workflow_reliability": round(
            (sum(r["validation_passed"] and r["all_four_nodes_ran"] for r in rows) / n), 4),
        "per_question": rows,
    }
    with open(out / "agent_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
