"""In-memory vector store over the marketing knowledge base.

Documents are markdown files in kb/ with YAML-ish front matter:
    ---\n    id: KB-01\n    title: Campaign Playbook\n    ---
Each document is chunked on blank lines; chunks are embedded and searched
with cosine similarity (TF-IDF space in demo mode).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.rag.embeddings import build_embedding

FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


@dataclass
class VectorStore:
    backend: str = "tfidf"
    min_chars: int = 60
    chunks: list[Chunk] = field(default_factory=list)
    _emb: object = None
    _matrix: np.ndarray | None = None

    # ------------------------------------------------------------------ build
    def add_documents(self, kb_dir) -> int:
        kb_dir = Path(kb_dir)
        n = 0
        for path in sorted(kb_dir.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            m = FRONT_MATTER.match(raw)
            meta, body = {}, raw
            if m:
                for line in m.group(1).splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip()] = v.strip()
                body = raw[m.end():]
            doc_id = meta.get("id", path.stem)
            title = meta.get("title", path.stem)
            for i, para in enumerate([p.strip() for p in re.split(r"\n\s*\n", body)]):
                if len(para) < self.min_chars:
                    continue
                self.chunks.append(Chunk(f"{doc_id}#c{i}", doc_id, title, para))
                n += 1
        return n

    def fit(self) -> "VectorStore":
        self._emb = build_embedding(self.backend)
        # embed title + text so title terms ("ROAS Benchmarks by Channel")
        # are searchable even when the body chunk is short
        self._emb.fit([f"{c.doc_title} — {c.text}" for c in self.chunks])
        M = self._emb.matrix
        # TF-IDF is sparse — densify (the KB is small; dense is fine and fast)
        self._matrix = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
        return self

    def build(self, kb_dir) -> "VectorStore":
        self.add_documents(kb_dir)
        return self.fit()

    # ----------------------------------------------------------------- search
    @staticmethod
    def _densify(x):
        return x.toarray() if hasattr(x, "toarray") else np.asarray(x)

    @classmethod
    def _cosine(cls, a, b) -> np.ndarray:
        a = cls._densify(a)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        b = cls._densify(b)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        return (a @ b.T) / (np.linalg.norm(a, axis=1, keepdims=True)
                            * np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)

    def search(self, query: str, k: int = 4) -> list[SearchResult]:
        if self._matrix is None or not len(self.chunks):
            raise RuntimeError("vector store is empty — call build()/fit() first")
        q = self._densify(self._emb.embed([query]))
        sims = self._cosine(q, self._matrix)[0]
        top = np.argsort(-sims)[:k]
        return [SearchResult(self.chunks[i], float(sims[i])) for i in top]

    def doc_index(self) -> dict[str, str]:
        return {c.doc_id: c.doc_title for c in self.chunks}
