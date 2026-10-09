"""
app/phase2/ingestion_api.py

Exposes Phase 2's pipeline (fetch -> parse -> chunk -> index) as two
HTTP endpoints so it runs under the same `uvicorn app.main:app`
process as Phase 1, instead of being separate standalone scripts.

In-memory index cache: fine for learning/single-process use. Phase 7
replaces _indexes with a persistent store (pgvector) so the index
survives a server restart and can be shared across multiple workers.
"""

import os

import requests
from fastapi import APIRouter, HTTPException
from google.genai.errors import APIError
from pydantic import BaseModel, Field

from app.phase2.edgar_fetcher import fetch_latest_10k
from app.phase2.html_parser import html_to_text_preserving_tables
from app.phase2.financial_chunker import chunk_filing
from app.phase2.hybrid_retriever import HybridIndex, GeminiEmbeddingProvider

router = APIRouter(prefix="/v2", tags=["ingestion"])

_indexes: dict[str, HybridIndex] = {}


class IngestRequest(BaseModel):
    ticker: str
    max_chunks: int = Field(default=10, ge=1, le=100)


class IngestResponse(BaseModel):
    ticker: str
    filing_date: str
    chunk_count: int
    total_chunk_count: int
    table_chunk_count: int


@router.post("/ingest", response_model=IngestResponse)
async def ingest_filing(req: IngestRequest):
    try:
        filing = fetch_latest_10k(req.ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except requests.exceptions.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"SEC EDGAR rejected the request ({exc}). Check SEC_USER_AGENT in edgar_fetcher.py.",
        )
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach SEC EDGAR: {exc}")

    text = html_to_text_preserving_tables(filing["html"])
    doc_id = f"{req.ticker.upper()}-10K-{filing['filing_date']}"
    all_chunks = chunk_filing(text, doc_id=doc_id)
    chunks = all_chunks[:req.max_chunks]

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured; set it in .env before ingesting filings.",
        )

    try:
        index = HybridIndex(embedding_provider=GeminiEmbeddingProvider())
        index.build(chunks)
    except APIError as exc:
        if exc.code == 429:
            raise HTTPException(
                status_code=503,
                detail="Gemini embedding quota or rate limit exceeded. Check your Gemini API project quota and billing, then retry.",
            ) from exc

        status = exc.status or "API_ERROR"
        message = exc.message or "No additional details provided by Gemini."
        raise HTTPException(
            status_code=502,
            detail=f"Gemini embedding request failed ({exc.code} {status}): {message}",
        ) from exc

    _indexes[req.ticker.upper()] = index

    return IngestResponse(
        ticker=req.ticker.upper(),
        filing_date=filing["filing_date"],
        chunk_count=len(chunks),
        total_chunk_count=len(all_chunks),
        table_chunk_count=sum(1 for c in chunks if c.chunk_type == "table"),
    )


class SearchRequest(BaseModel):
    ticker: str
    query: str
    k: int = 5


@router.post("/search")
async def search_filing(req: SearchRequest):
    index = _indexes.get(req.ticker.upper())
    if index is None:
        raise HTTPException(
            status_code=404,
            detail=f"No filing ingested for '{req.ticker}'. Call /v2/ingest first.",
        )

    results = index.search(req.query, k=req.k)
    return {
        "results": [
            {
                "section": r.chunk.section,
                "chunk_type": r.chunk.chunk_type,
                "rrf_score": round(r.rrf_score, 5),
                "text": r.chunk.text,
            }
            for r in results
        ]
    }