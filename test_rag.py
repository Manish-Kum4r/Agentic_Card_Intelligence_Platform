"""Tests for the RAG pipeline (retrieval quality + citation behaviour)."""
from __future__ import annotations

import re

import pytest

from src.rag.rag import build_rag


@pytest.fixture(scope="module")
def rag():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    return build_rag(root / "kb", top_k=4, embedding_backend="tfidf")


def test_kb_indexed(rag):
    assert len(rag.store.chunks) >= 50, "knowledge base not chunked properly"
    doc_ids = {c.doc_id for c in rag.store.chunks}
    assert len(doc_ids) == 14, f"expected 14 KB docs, got {len(doc_ids)}"


@pytest.mark.parametrize("question,expected_doc", [
    ("What is the engagement score formula?", "KB-08"),
    ("What do we do when a high-spend cardholder's engagement declines?", "KB-09"),
    ("What are the ROAS benchmarks by channel?", "KB-10"),
    ("How do we design an A/B test?", "KB-07"),
    ("What are the privacy and consent rules?", "KB-11"),
    ("How should we allocate the marketing budget?", "KB-14"),
])
def test_retrieval_finds_right_doc(rag, question, expected_doc):
    results = rag.retrieve(question, k=3)
    got = [r.chunk.doc_id for r in results]
    assert expected_doc in got, f"{question!r}: expected {expected_doc} in {got}"


def test_answer_has_citations(rag):
    resp = rag.answer("What are the email marketing best practices?")
    assert re.search(r"\[KB-\d+\]", resp.answer), "answer must cite [KB-xx] sources"
    assert resp.sources, "sources list must be non-empty"
    assert resp.confidence > 0


def test_answer_is_grounded(rag):
    """Every bullet in the answer must come from a retrieved chunk (no
    invented facts — the demo composer is extractive by construction)."""
    resp = rag.answer("What are the in-app messaging frequency caps?")
    bullets = [l for l in resp.answer.splitlines() if l.startswith("• ")]
    assert bullets, "expected extractive bullets"
    # normalize whitespace: sentences are newline-collapsed in the answer
    full_text = " ".join(" ".join(c.text.split()) for c in rag.store.chunks)
    for b in bullets:
        sentence = " ".join(b[2:].split(" [KB-")[0].split())
        assert sentence[:60] in full_text, f"ungrounded sentence: {sentence[:60]}"
