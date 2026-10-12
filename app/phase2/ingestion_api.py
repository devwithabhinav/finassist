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
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
import requests
from fastapi import APIRouter, HTTPException, File, Form, UploadFile, Depends
from google.genai.errors import APIError
from pydantic import BaseModel, Field

from app.phase2.edgar_fetcher import fetch_latest_10k
from app.phase2.html_parser import html_to_text_preserving_tables
from app.phase2.financial_chunker import chunk_filing
from app.phase2.hybrid_retriever import HybridIndex, GeminiEmbeddingProvider
from app.phase2.pdf_parser import pdf_to_text_preserving_tables
from app.phase2.entity_graph import EntityGraph, extract_entities, save_graph, load_graphs

GRAPH_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "graphs"
_graphs: dict[str, EntityGraph] = {}


@asynccontextmanager
async def graph_lifespan(app):
    _graphs.update(load_graphs(GRAPH_DATA_DIR))
    yield


router = APIRouter(prefix="/v2", tags=["ingestion"], lifespan=graph_lifespan)

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


@router.post("/ingest-pdf")
def ingest_pdf(
    file: UploadFile = File(...),
    ticker: str = Form(..., min_length=1),
    max_chunks: int = Form(10, ge=1, le=100),
):
    """Upload a PDF under a ticker or arbitrary label; search using that label."""
    label = ticker.strip().upper()
    if not label:
        raise HTTPException(status_code=422, detail="Ticker/label must not be blank.")
    data = file.file.read()
    try:
        text = pdf_to_text_preserving_tables(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    doc_id = f"{label}-PDF-{hashlib.sha256(data).hexdigest()[:16]}"
    all_chunks = chunk_filing(text, doc_id)
    if not all_chunks:
        raise HTTPException(status_code=400, detail="The PDF contains no indexable content.")
    chunks = all_chunks[:max_chunks]
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured.")
    try:
        index = HybridIndex(embedding_provider=GeminiEmbeddingProvider())
        index.build(chunks)
    except APIError as exc:
        raise HTTPException(
            status_code=503 if exc.code == 429 else 502,
            detail="Gemini embedding request failed; retry later or check provider configuration.",
        ) from exc
    except httpx.TransportError as exc:
        raise HTTPException(status_code=502, detail="Could not reach the embedding provider.") from exc
    _indexes[label] = index
    return {
        "ticker": label,
        "chunk_count": len(chunks),
        "total_chunk_count": len(all_chunks),
        "table_chunk_count": sum(c.chunk_type == "table" for c in chunks),
    }


class GraphBuildRequest(BaseModel):
    ticker: str = Field(min_length=1)


def get_entity_extractor():
    return extract_entities


@router.post("/graph/build")
def build_entity_graph(req: GraphBuildRequest, extractor=Depends(get_entity_extractor)):
    ticker = req.ticker.strip().upper()
    index = _indexes.get(ticker)
    if index is None or not index.chunks:
        raise HTTPException(status_code=404, detail="No ingested chunks for this ticker; ingest a filing first.")
    graph = EntityGraph(extractor=extractor)
    try:
        graph.ingest_chunks(index.chunks)
    except APIError as exc:
        raise HTTPException(status_code=503 if exc.code == 429 else 502, detail="Entity extraction provider failed.") from exc
    except (httpx.TransportError, requests.exceptions.RequestException) as exc:
        raise HTTPException(status_code=502, detail="Could not reach the entity extraction provider.") from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Entity extraction returned invalid data.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Entity extraction provider is not configured.") from exc
    try:
        save_graph(GRAPH_DATA_DIR, ticker, graph)
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Could not persist the entity graph.") from exc
    _graphs[ticker] = graph
    return {"ticker": ticker, "node_count": graph.graph.number_of_nodes(), "relationship_count": graph.graph.number_of_edges()}


@router.get("/graph/{ticker}/entity/{name}")
def get_connected_entities(ticker: str, name: str):
    graph = _graphs.get(ticker.strip().upper())
    if graph is None:
        raise HTTPException(status_code=404, detail="No graph available for this ticker; build it first.")
    return {"ticker": ticker.strip().upper(), "entity": name, "relationships": graph.find_connected(name)}


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
                "section_title": r.chunk.section_title,
                "metadata": r.chunk.metadata,
                "chunk_type": r.chunk.chunk_type,
                "rrf_score": round(r.rrf_score, 5),
                "text": r.chunk.text,
            }
            for r in results
        ]
    }
