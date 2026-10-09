"""
app/phase2/hybrid_retriever.py

Combines two very different search strategies:
  - Sparse (BM25): keyword/exact-term matching.
  - Dense (embeddings): semantic/meaning matching.

Reciprocal Rank Fusion (RRF) combines their RANKS (not raw scores,
which live on incompatible scales) into one ordering.
"""

from abc import ABC, abstractmethod
import hashlib
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from app.phase2.financial_chunker import Chunk


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        ...


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Real provider — used once GEMINI_API_KEY is set on your machine."""

    def __init__(self, model: str = "gemini-embedding-001"):
        import os
        from google import genai
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        from google.genai import types

        vectors = []
        for start in range(0, len(texts), 100):
            batch = texts[start:start + 100]
            contents = [
                types.Content(parts=[types.Part.from_text(text=text)])
                for text in batch
            ]
            result = self._client.models.embed_content(model=self._model, contents=contents)
            vectors.extend(embedding.values for embedding in result.embeddings)
        return np.array(vectors)


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based pseudo-embeddings for testing the fusion
    logic without a network call — not semantically meaningful."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            vectors.append(rng.normal(size=self.dim))
        return np.array(vectors)


@dataclass
class SearchResult:
    chunk: Chunk
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None


class HybridIndex:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._dense_vectors: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._dense_vectors = self.embedding_provider.embed([c.text for c in chunks])

    def search(self, query: str, k: int = 5, rrf_k: int = 60) -> list[SearchResult]:
        bm25_scores = self._bm25.get_scores(query.lower().split())
        sparse_order = np.argsort(-bm25_scores)
        sparse_rank_of = {idx: rank for rank, idx in enumerate(sparse_order)}

        query_vec = self.embedding_provider.embed([query])[0]
        sims = self._dense_vectors @ query_vec / (
            np.linalg.norm(self._dense_vectors, axis=1) * np.linalg.norm(query_vec) + 1e-9
        )
        dense_order = np.argsort(-sims)
        dense_rank_of = {idx: rank for rank, idx in enumerate(dense_order)}

        fused_scores = {}
        for idx in range(len(self.chunks)):
            s_rank = sparse_rank_of.get(idx)
            d_rank = dense_rank_of.get(idx)
            score = 0.0
            if s_rank is not None:
                score += 1.0 / (rrf_k + s_rank)
            if d_rank is not None:
                score += 1.0 / (rrf_k + d_rank)
            fused_scores[idx] = (score, s_rank, d_rank)

        ranked = sorted(fused_scores.items(), key=lambda item: -item[1][0])[:k]
        return [
            SearchResult(chunk=self.chunks[idx], rrf_score=score, sparse_rank=s_rank, dense_rank=d_rank)
            for idx, (score, s_rank, d_rank) in ranked
        ]


if __name__ == "__main__":
    from app.phase2.financial_chunker import chunk_filing

    sample_10k = """
Item 1A. Risk Factors
The Company depends on a small number of key suppliers for critical
sensor components, and any disruption to these suppliers could
materially affect production.

Item 7. Management's Discussion and Analysis
Revenue grew year over year driven primarily by the industrial IoT segment.
Gross margin expanded due to favorable component pricing.

| Metric | FY2024 | FY2023 |
|---|---|---|
| Revenue | $412M | $365M |
| Gross Margin | 58.2% | 55.1% |

The Board approved a $50M share repurchase program during the fourth quarter.
"""
    chunks = chunk_filing(sample_10k, doc_id="ACME-10K-2024")

    index = HybridIndex(embedding_provider=FakeEmbeddingProvider())
    index.build(chunks)

    results = index.search("Does the company rely on few vendors for parts?", k=3)
    for r in results:
        print(f"RRF={r.rrf_score:.4f}  sparse_rank={r.sparse_rank}  dense_rank={r.dense_rank}  -> {r.chunk.text[:60]!r}")