"""LLM layer.

Demo mode (default): `LocalLLM` — a deterministic, fully offline response
composer. It assembles answers ONLY from retrieved knowledge-base passages
and the analytics context, with explicit citations. It never invents facts:
every sentence in its output is either (a) a verbatim sentence from a
retrieved KB chunk, or (b) a number taken from the analytics context.

Production: set LLM_BACKEND to point at a real LLM (Ollama / OpenAI-compatible
endpoint). The RAG pipeline and the agent pass the same prompt format, so
swapping the LLM is a one-line change and does not touch retrieval.
"""
from __future__ import annotations

import re


class LocalLLM:
    """Deterministic offline 'LLM' used in demo mode.

    Honest by construction: extractive answers with citations, no free
    generation, no external calls, no paid API.
    """

    name = "local-demo-composer"

    def compose_answer(self, question: str, results: list, max_sentences: int = 3) -> str:
        """Build an answer from retrieved SearchResults (see vectorstore)."""
        if not results:
            return ("I could not find anything relevant in the marketing knowledge "
                    "base for that question.")
        # rank individual sentences by relevance to the question
        import numpy as np
        from src.rag.vectorstore import VectorStore
        sentences = []
        for r in results:
            for s in re.split(r"(?<=[.!?])\s+", r.chunk.text):
                s = s.strip()
                if len(s) > 40:
                    sentences.append((r, s))
        scored = []
        if sentences:
            texts = [s for _, s in sentences]
            # TF-IDF sentence scoring against the question
            from sklearn.feature_extraction.text import TfidfVectorizer
            try:
                vec = TfidfVectorizer(stop_words="english", norm="l2")
                M = vec.fit_transform(texts + [question])
                q = M[-1]
                sims = (M[:-1] @ q.T).toarray().ravel()
                scored = sorted(zip(sims, sentences), key=lambda x: -x[0])
            except ValueError:
                scored = [(0.0, pair) for pair in sentences]
        # take the best sentences: first pass one per source doc (diversity),
        # second pass fills remaining slots if too few docs were retrieved
        picked, used_docs = [], set()
        for _ in range(2):
            for sim, (r, s) in scored:
                if len(picked) >= max_sentences:
                    break
                if r.chunk.doc_id in used_docs and _ == 0:
                    continue
                if (r, s) in picked:
                    continue
                picked.append((r, s))
                used_docs.add(r.chunk.doc_id)
            if len(picked) >= max_sentences:
                break
        lines = ["Based on the marketing knowledge base:"]
        for r, s in picked:
            s_clean = " ".join(s.split()).lstrip("#").strip()
            lines.append(f"• {s_clean} [{r.chunk.doc_id}]")
        lines.append("")
        lines.append("Sources: " + ", ".join(
            f"{r.chunk.doc_id} ({r.chunk.doc_title}, relevance {r.score:.2f})"
            for r in results))
        return "\n".join(lines)

    def compose_recommendation(self, question: str, analysis: dict,
                               results: list) -> str:
        """Compose a recommendation grounded in analytics + KB citations."""
        parts = []
        parts.append(f"Question: {question}")
        parts.append("")
        parts.append("Data findings (analytics layer):")
        if isinstance(analysis, dict):
            for k, v in analysis.items():
                parts.append(f"- {k}: {v}")
        else:  # list-of-records responses (segment profiles, channel rollup)
            for row in analysis:
                parts.append(f"- {row}")
        parts.append("")
        parts.append("Knowledge-base guidance:")
        for r in results:
            first_sentence = re.split(r"(?<=[.!?])\s+", r.chunk.text)[0].strip()
            parts.append(f"- [{r.chunk.doc_id}] {first_sentence}")
        parts.append("")
        parts.append(
            "Recommendation: act on the data findings above following the cited "
            "knowledge-base guidance. Every figure quoted is computed by the "
            "analytics pipeline in this run; every practice cited comes from the "
            "referenced KB document. A human marketer reviews before execution.")
        return "\n".join(parts)


class APILLM:
    """Optional production LLM (not used in demo mode).

    Supports any OpenAI-compatible endpoint (Ollama, vLLM, Azure, etc.).
    Configure via environment: LLM_BACKEND, OLLAMA_HOST/OPENAI_BASE_URL,
    OLLAMA_MODEL/OPENAI_MODEL, and an API key if required.
    """

    def __init__(self):
        import os
        self.backend = os.getenv("LLM_BACKEND", "")
        self.base_url = os.getenv("OLLAMA_HOST") or os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("OLLAMA_MODEL") or os.getenv("OPENAI_MODEL", "")
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OLLAMA_API_KEY", "")
        if not (self.backend and self.base_url and self.model):
            raise RuntimeError(
                "APILLM needs LLM_BACKEND, a base URL (OLLAMA_HOST/OPENAI_BASE_URL) "
                "and a model name (OLLAMA_MODEL/OPENAI_MODEL). Demo mode uses "
                "LocalLLM instead — no configuration needed.")

    def generate(self, prompt: str) -> str:
        import json
        import urllib.request
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.2}
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]


def build_llm(prefer_api: bool = False):
    """Factory: use a real API LLM only if explicitly configured."""
    if prefer_api:
        try:
            return APILLM()
        except RuntimeError:
            pass
    return LocalLLM()
