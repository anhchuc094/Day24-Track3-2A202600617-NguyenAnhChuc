from __future__ import annotations

"""Module 3: Cross-encoder reranking with a deterministic fallback."""

import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _fallback_score(query: str, document: str) -> float:
    q = Counter(_tokens(query))
    d = Counter(_tokens(document))
    overlap = sum(min(q[t], d[t]) for t in q)
    return overlap + 0.01 * len(set(q) & set(d))


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._load_failed = False

    def _load_model(self):
        """Load sentence-transformers CrossEncoder when explicitly enabled."""
        if self._model is None and not self._load_failed and os.getenv("RAG_USE_REAL_MODELS") == "1":
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._load_failed = True
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents and return the top-k results."""
        if not documents:
            return []

        model = self._load_model()
        if model is not None:
            scores = model.predict([(query, doc["text"]) for doc in documents])
            if isinstance(scores, (int, float)):
                scores = [scores]
            scores = [float(score) for score in scores]
        else:
            scores = [_fallback_score(query, doc["text"]) for doc in documents]

        scored = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Optional lightweight alternative with the same return contract."""

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        return CrossEncoderReranker().rerank(query, documents, top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhan vien duoc nghi phep bao nhieu ngay?"
    docs = [
        {"text": "Nhan vien duoc nghi 12 ngay/nam.", "score": 0.8, "metadata": {}},
        {"text": "Mat khau thay doi moi 90 ngay.", "score": 0.7, "metadata": {}},
    ]
    for result in CrossEncoderReranker().rerank(query, docs):
        print(f"[{result.rank}] {result.rerank_score:.4f} | {result.text}")
