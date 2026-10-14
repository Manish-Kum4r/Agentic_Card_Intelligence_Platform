"""Retrieval-Augmented Generation pipeline.

Flow: question → embed → vector search (top-k) → answer composition with
citations (LocalLLM in demo mode) → structured response with sources.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.rag.llm import LocalLLM, build_llm
from src.rag.vectorstore import SearchResult, VectorStore


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer,
                "sources": self.sources, "retrieved": self.retrieved,
                "confidence": round(self.confidence, 4)}


class RAGPipeline:
    def __init__(self, kb_dir, top_k: int = 4, embedding_backend: str = "tfidf",
                 min_chars: int = 60, llm=None):
        self.top_k = top_k
        self.store = VectorStore(backend=embedding_backend, min_chars=min_chars)
        self.store.build(kb_dir)
        self.llm = llm or build_llm()

    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        return self.store.search(question, k or self.top_k)

    def answer(self, question: str, k: int | None = None) -> RAGResponse:
        results = self.retrieve(question, k)
        answer = (self.llm.compose_answer(question, results)
                  if isinstance(self.llm, LocalLLM)
                  else self.llm.generate(self._prompt(question, results)))
        # keep the best-scoring chunk per document
        best: dict[str, object] = {}
        for r in results:
            if (r.chunk.doc_id not in best
                    or r.score > best[r.chunk.doc_id].score):
                best[r.chunk.doc_id] = r
        top = sorted(best.values(), key=lambda r: -r.score)
        sources = [{"doc_id": r.chunk.doc_id, "title": r.chunk.doc_title,
                    "relevance": round(r.score, 4)} for r in top]
        retrieved = [{"chunk_id": r.chunk.chunk_id, "doc_id": r.chunk.doc_id,
                      "score": round(r.score, 4), "text": r.chunk.text[:400]}
                     for r in top]
        confidence = float(sum(r.score for r in top) / len(top)) if top else 0.0
        return RAGResponse(question=question, answer=answer, sources=sources,
                           retrieved=retrieved, confidence=confidence)

    @staticmethod
    def _prompt(question: str, results: list[SearchResult]) -> str:
        ctx = "\n\n".join(f"[{r.chunk.doc_id}] {r.chunk.text}" for r in results)
        return (f"Answer the question using ONLY the context below. Cite sources "
                f"as [KB-xx].\n\nContext:\n{ctx}\n\nQuestion: {question}")


def build_rag(kb_dir, top_k: int = 4, embedding_backend: str = "tfidf") -> RAGPipeline:
    return RAGPipeline(kb_dir, top_k=top_k, embedding_backend=embedding_backend)


if __name__ == "__main__":
    from src.config import load_config, paths
    cfg = load_config(); p = paths(cfg)
    rag = build_rag(p["kb"], top_k=cfg["rag"]["top_k"],
                    embedding_backend=cfg["rag"]["embedding_backend"])
    print(f"KB chunks indexed: {len(rag.store.chunks)}")
    resp = rag.answer("What should I do when a high-spend cardholder's engagement declines?")
    print(resp.answer)
    print("\nsources:", [s["doc_id"] for s in resp.sources])
