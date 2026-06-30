from __future__ import annotations

"""Module 2: Hybrid Search with BM25, dense search, and RRF."""

import os
import re
import sys
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BM25_TOP_K, COLLECTION_NAME, DENSE_TOP_K, EMBEDDING_DIM, EMBEDDING_MODEL, HYBRID_TOP_K, QDRANT_HOST, QDRANT_PORT


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words, falling back to raw text when underthesea is absent."""
    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        return text


def _tokens(text: str) -> list[str]:
    segmented = segment_vietnamese(text.lower())
    return re.findall(r"\w+", segmented, flags=re.UNICODE)


def _lexical_score(query_tokens: list[str], text: str) -> float:
    counts = Counter(_tokens(text))
    return float(sum(counts.get(t, 0) for t in query_tokens))


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build a BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [_tokens(chunk["text"]) for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25, or a token-overlap fallback."""
        if not self.documents:
            return []

        query_tokens = _tokens(query)
        if self.bm25 is not None:
            scores = [float(score) for score in self.bm25.get_scores(query_tokens)]
        else:
            scores = [_lexical_score(query_tokens, doc["text"]) for doc in self.documents]

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=scores[i],
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices
            if scores[i] > 0
        ]


class DenseSearch:
    def __init__(self):
        self.client = None
        self._encoder = None
        self._memory_chunks: list[dict] = []
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception:
            self.client = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant, with an in-memory fallback."""
        self._memory_chunks = chunks
        if self.client is None:
            return
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            texts = [c["text"] for c in chunks]
            vectors = self._get_encoder().encode(texts, show_progress_bar=False)
            points = [
                PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload={**chunk.get("metadata", {}), "text": chunk["text"]},
                )
                for i, (vector, chunk) in enumerate(zip(vectors, chunks))
            ]
            self.client.upsert(collection, points)
        except Exception:
            self.client = None

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors, falling back to lexical scoring."""
        if self.client is not None:
            try:
                query_vector = self._get_encoder().encode(query).tolist()
                response = self.client.query_points(collection, query=query_vector, limit=top_k)
                return [
                    SearchResult(
                        text=pt.payload.get("text", ""),
                        score=float(pt.score),
                        metadata={k: v for k, v in pt.payload.items() if k != "text"},
                        method="dense",
                    )
                    for pt in response.points
                ]
            except Exception:
                self.client = None

        query_tokens = _tokens(query)
        scored = [
            (_lexical_score(query_tokens, chunk["text"]), chunk)
            for chunk in self._memory_chunks
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(chunk["text"], score, chunk.get("metadata", {}), "dense")
            for score, chunk in scored[:top_k]
            if score > 0
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """Merge ranked lists using reciprocal rank fusion."""
    fused: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results):
            if result.text not in fused:
                fused[result.text] = {"score": 0.0, "result": result}
            fused[result.text]["score"] += 1.0 / (k + rank + 1)

    ordered = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in ordered
    ]


class HybridSearch:
    """Combine BM25 and dense search."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(segment_vietnamese("Nhan vien duoc nghi phep nam"))
