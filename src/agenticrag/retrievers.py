from __future__ import annotations

import math
from collections import Counter
from typing import Any, List, Optional, Tuple


class TokenOverlapRetriever:
    """Baseline Jaccard-overlap retriever (already in core.py, mirrored here for unified API)."""

    def retrieve(self, query: str, corpus: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        results = []
        q_toks = set(query.lower().split())
        for doc in corpus:
            d_toks = set(doc.lower().split())
            union = q_toks | d_toks
            score = len(q_toks & d_toks) / len(union) if union else 0.0
            results.append((doc, score))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


class BM25Retriever:
    """BM25 retriever.

    Falls back to pure-Python BM25 implementation when rank-bm25 is not
    installed so the package stays importable without the optional dependency.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: List[str] = []
        self._tokenized: List[List[str]] = []
        self._df: Counter = Counter()
        self._avgdl: float = 0.0
        self._use_rank_bm25 = False
        self._bm25_obj = None

    def index(self, corpus: List[str]) -> None:
        self._corpus = corpus
        self._tokenized = [doc.lower().split() for doc in corpus]

        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            self._bm25_obj = BM25Okapi(self._tokenized, k1=self.k1, b=self.b)
            self._use_rank_bm25 = True
        except ImportError:
            self._use_rank_bm25 = False
            self._build_index()

    def _build_index(self) -> None:
        self._df = Counter()
        for toks in self._tokenized:
            for t in set(toks):
                self._df[t] += 1
        self._avgdl = (
            sum(len(t) for t in self._tokenized) / len(self._tokenized)
            if self._tokenized
            else 0.0
        )

    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        N = len(self._tokenized)
        doc_toks = self._tokenized[doc_idx]
        dl = len(doc_toks)
        tf = Counter(doc_toks)
        score = 0.0
        for qt in query_tokens:
            if qt not in tf:
                continue
            df = self._df.get(qt, 0)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            numer = tf[qt] * (self.k1 + 1)
            denom = tf[qt] + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
            score += idf * numer / denom
        return score

    def retrieve(self, query: str, corpus: Optional[List[str]] = None, top_k: int = 5) -> List[Tuple[str, float]]:
        if corpus is not None:
            self.index(corpus)

        if not self._corpus:
            return []

        q_toks = query.lower().split()

        if self._use_rank_bm25 and self._bm25_obj is not None:
            scores = self._bm25_obj.get_scores(q_toks)
            ranked = sorted(
                zip(self._corpus, scores), key=lambda x: -x[1]
            )
        else:
            scores = [self._score(q_toks, i) for i in range(len(self._tokenized))]
            ranked = sorted(zip(self._corpus, scores), key=lambda x: -x[1])

        return ranked[:top_k]


class DenseRetriever:
    """Dense bi-encoder retriever using sentence-transformers (cosine similarity).

    Provides the same ``retrieve(query, corpus, top_k) -> [(doc, score), ...]``
    interface as ``BM25Retriever`` so it drops into ``LLMAgent`` /
    ``AgenticRAGPipeline`` unchanged. Embeddings for a corpus are cached by
    object identity so re-querying the same corpus across hops does not re-encode.

    Falls back to token-overlap scoring when ``sentence-transformers`` is not
    installed, keeping the package importable and tests runnable offline. Scores
    are non-negative so the ``score > 0`` relevance gate in the pipelines behaves
    consistently with BM25.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._available = False
        self._cache_key: Optional[int] = None
        self._cache_emb: Any = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_name)
            self._available = True
        except Exception:  # ImportError or model-download failure
            self._available = False

    def _encode(self, texts: List[str]) -> Any:
        import numpy as np  # local import; numpy is a hard dependency

        emb = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(emb, dtype="float32")

    def retrieve(
        self, query: str, corpus: Optional[List[str]] = None, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        if not corpus:
            return []

        if not self._available:
            # Deterministic offline fallback: Jaccard token overlap.
            q_toks = set(query.lower().split())
            ranked = []
            for doc in corpus:
                d_toks = set(doc.lower().split())
                union = q_toks | d_toks
                ranked.append((doc, len(q_toks & d_toks) / len(union) if union else 0.0))
            ranked.sort(key=lambda x: -x[1])
            return ranked[:top_k]

        import numpy as np

        key = id(corpus)
        if self._cache_key != key:
            self._cache_emb = self._encode(corpus)
            self._cache_key = key
        q_emb = self._encode([query])[0]
        sims = self._cache_emb @ q_emb  # cosine sim (vectors are normalized)
        order = np.argsort(-sims)[:top_k]
        # Clamp to [0, 1] so the score>0 relevance gate matches BM25 semantics.
        return [(corpus[i], float(max(0.0, sims[i]))) for i in order]
