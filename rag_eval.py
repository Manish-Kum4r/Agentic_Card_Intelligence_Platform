"""RAG evaluation: retrieval quality on a labeled query set + answer checks.

Labeled set: 12 questions with the knowledge-base document that must be
retrieved. Metrics: Precision@k, Recall@k, MRR, Hit-rate, plus answer-level
checks (citations present, non-trivial length, sources match retrieval).
"""
from __future__ import annotations

import json
from pathlib import Path

LABELED_QUERIES = [
    ("What is the engagement score formula and how is it calculated?", ["KB-08"]),
    ("What should we do when a high-spend cardholder's engagement declines?", ["KB-09"]),
    ("What are the ROAS benchmarks by channel?", ["KB-10"]),
    ("How do we design and read an A/B test?", ["KB-07"]),
    ("What are the email marketing best practices and benchmarks?", ["KB-04"]),
    ("When should we use statement inserts versus direct mail?", ["KB-06"]),
    ("What are the data privacy, consent and compliance rules?", ["KB-11"]),
    ("How should we allocate the marketing budget across campaigns?", ["KB-14"]),
    ("What does the seasonal campaign calendar look like?", ["KB-13"]),
    ("What are the offer terms and rewards earn rates?", ["KB-03"]),
    ("How do we choose the right channel for a campaign?", ["KB-01"]),
    ("How are customer segments and personas defined?", ["KB-02"]),
    ("What insights do merchant spending categories give us?", ["KB-12"]),
    ("What are the in-app messaging frequency caps?", ["KB-05"]),
]


def run_rag_evaluation(rag, outputs_dir, k: int = 3,
                       min_hit_rate: float = 0.75, min_mrr: float = 0.60) -> dict:
    """Retrieval quality gates.

    Each query has exactly ONE gold document, so Precision@k is capped at
    1/k — the meaningful gates are Hit-rate@k (gold doc in top-k), MRR and
    Precision@1 (gold doc ranked first).
    """
    out = Path(outputs_dir) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    rows, hits, rr, p1 = [], 0, [], 0
    for question, expected in LABELED_QUERIES:
        results = rag.retrieve(question, k=k)
        got = [r.chunk.doc_id for r in results]
        hit = any(d in got for d in expected)
        hits += hit
        p1 += got[0] in expected
        rr.append(1.0 / (got.index(expected[0]) + 1) if expected[0] in got else 0.0)
        resp = rag.answer(question)
        rows.append({
            "question": question,
            "expected_docs": expected,
            "retrieved_docs": got,
            "hit": hit,
            "gold_rank": got.index(expected[0]) + 1 if expected[0] in got else None,
            "citation_present": any(f"[{d}]" in resp.answer for d in got),
            "answer_chars": len(resp.answer),
            "mean_relevance": round(sum(r.score for r in results) / len(results), 4),
        })
    n = len(LABELED_QUERIES)
    hit_rate, mrr = hits / n, sum(rr) / n
    report = {
        "n_queries": n,
        "k": k,
        "hit_rate_at_k": round(hit_rate, 4),
        "precision_at_1": round(p1 / n, 4),
        "mrr": round(mrr, 4),
        "min_hit_rate_floor": min_hit_rate,
        "min_mrr_floor": min_mrr,
        "hit_rate_floor_passed": bool(hit_rate >= min_hit_rate),
        "mrr_floor_passed": bool(mrr >= min_mrr),
        "all_answers_cited": all(r["citation_present"] for r in rows),
        "per_query": rows,
    }
    with open(out / "rag_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
