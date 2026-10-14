"""Embedding backends.

Demo mode (default, fully offline): TF-IDF embeddings via scikit-learn.
Optional upgrade: sentence-transformers (semantic, dense) — set
RAG_EMBEDDING_BACKEND=sentence-transformers and install the package.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedding:
    """TF-IDF embeddings — deterministic, offline, no downloads.

    Feature space = word 1-2 grams UNION character 3-5 grams (char_wb).
    The char n-grams make matching robust to morphology (e.g. a query with
    "allocate" still matches documents containing "allocation").
    Rows are L2-normalized so dot product = cosine similarity.
    """

    name = "tfidf"

    def __init__(self, max_features: int = 8000, char_max_features: int = 30000):
        from scipy.sparse import hstack
        from sklearn.preprocessing import normalize
        self._hstack, self._normalize = hstack, normalize
        self.max_features = max_features
        self._word = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True, norm=None,
        )
        self._char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), lowercase=True,
            max_features=char_max_features, sublinear_tf=True, norm=None,
        )
        self._X = None

    def _combine(self, Xw, Xc):
        return self._normalize(self._hstack([Xw, Xc]).tocsr(), norm="l2")

    def fit(self, texts: list[str]) -> "TfidfEmbedding":
        self._X = self._combine(self._word.fit_transform(texts),
                                self._char.fit_transform(texts))
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._combine(self._word.transform(texts),
                             self._char.transform(texts))

    @property
    def matrix(self) -> np.ndarray:
        return self._X

    @property
    def dim(self) -> int:
        return self._X.shape[1] if self._X is not None else self.max_features


class SentenceTransformerEmbedding:
    """Optional dense semantic embeddings (requires sentence-transformers)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dep
        self._model = SentenceTransformer(model_name)
        self._X = None

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedding":
        self._X = self._model.encode(texts, normalize_embeddings=True,
                                      show_progress_bar=False)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False)

    @property
    def matrix(self) -> np.ndarray:
        return self._X

    @property
    def dim(self) -> int:
        return self._X.shape[1] if self._X is not None else 384


def build_embedding(backend: str = "tfidf"):
    if backend == "tfidf":
        return TfidfEmbedding()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedding()
    raise ValueError(f"unknown embedding backend: {backend}")
