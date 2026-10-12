# AGENTS.md — FinAssist (Enterprise Financial Intelligence & Portfolio Assistant)

## What this project is
A learning-grade but production-patterned AI system: answers questions over SEC 10-K filings,
computes portfolio metrics, and (later) proposes portfolio rebalancing with human approval.
Backend: Python 3.12 + FastAPI. Frontend (later): TypeScript/Next.js. LLM: Gemini API
(`google-genai` SDK, model `gemini-2.5-flash`). No self-hosted models.

## Core architectural rules (do not violate)
1. **The LLM never computes numbers.** Sharpe ratio, weights, P&L, etc. come from deterministic,
   unit-tested Python. The LLM only retrieves, orchestrates, and narrates.
2. **Citation before generation.** SSE event order is: `citation` -> `metric` -> `token`* -> `done`
   (`error` on failure). Never emit a citation after the text it supports.
3. **Tables are atomic chunks.** The chunker must never split a table across chunks.
4. **Every provider call fails gracefully.** Network/provider errors become clean HTTP errors or
   SSE `error` events, never raw 500s or hung connections.
5. Provider access is behind interfaces (e.g. `EmbeddingProvider`) so implementations can be swapped.

## Project layout
app/
  main.py                 # single FastAPI app; calls load_dotenv() BEFORE importing routers
  phase1/
    token_economics.py    # cost/latency estimator + tier routing
    streaming_api.py      # POST /v1/query/stream (SSE, Gemini streaming)
  phase2/
    edgar_fetcher.py      # ticker -> CIK -> latest 10-K HTML (needs real SEC_USER_AGENT)
    html_parser.py        # HTML -> text, <table> converted to markdown pipes
    financial_chunker.py  # section-aware + table-atomic chunking
    hybrid_retriever.py   # BM25 + dense embeddings fused with RRF
    entity_graph.py       # networkx GraphRAG-lite (Gemini JSON extraction), not wired to API yet
    ingestion_api.py      # POST /v2/ingest, POST /v2/search (in-memory index cache)

## Run
- Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Secrets: copy `.env.example` to `.env`, set `GEMINI_API_KEY`
- Server: `uvicorn app.main:app --reload`  -> docs at http://localhost:8000/docs
- Run a module standalone from repo root with `-m`, never by file path:
  `python -m app.phase2.financial_chunker`

## Status
- Phase 1 (streaming endpoint, cost estimator): done.
- Phase 2: built but NOT yet verified against real SEC data / real Gemini embeddings.

## Pending Phase 2 work (in priority order)
1. Chunker: skip table-of-contents "Item N." matches (real 10-Ks list items in the TOC first).
2. `HybridIndex.build()`: batch embedding requests with retry/backoff (hundreds of chunks).
3. Table chunks: attach caption, units ("in millions"), and section title as context.
4. PDF table extraction path (pdfplumber or similar).
5. Wire `entity_graph.py` to an endpoint; persist it.
6. Add pytest tests for chunker, html_parser, and RRF fusion.

## Upcoming phases
3 Model Gateway (auth, rate limits, Redis semantic cache, model routing, provider fallback) ·
4 LangGraph agents + tools + human approval gates · 5 Guardrails/PII/audit log ·
6 Evals (Ragas, numeric faithfulness) · 7 Docker/K8s/OpenTelemetry.

## Working with the owner
- Experienced engineer (12+ yrs, C#/.NET) but new to Python: explain Python-specific idioms and
  give exact commands; map to .NET equivalents when helpful.
- Prefer small, runnable, tested increments. Say what you verified and what you could not.