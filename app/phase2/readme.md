# Phase 2: Financial document retrieval

Phase 2 turns SEC 10-K HTML or an uploaded PDF into searchable evidence. It also extracts entities and relationships into a persistent graph. It returns passages and relationships; it does not yet pass retrieved evidence to Phase 1 to generate grounded answers.

```text
SEC ticker -> latest 10-K HTML --+
                               +-> text with markdown tables -> section-aware chunks
PDF upload -> PDF extraction --+                                  |
                                                                 +-> BM25 + embeddings -> RRF search
                                                                 +-> entity extraction -> JSON graph -> relationship lookup
```

## Current status

The six Phase 2 implementation items now have code and automated coverage:

- Ignore short repeated table-of-contents Item headings while preserving short real sections.
- Batch embedding calls with bounded retry and exponential backoff.
- Attach table caption/heading, units, and section title to both chunk text and metadata.
- Extract PDF text and tables using pdfplumber.
- Build and query entity graphs through the API, persist them, and reload them at startup.
- Test parsing, chunking, RRF fusion, PDF ingestion, graph persistence, and failure handling. GitHub Actions runs pytest on every push.

The latest local run passed 46 tests using the project `.venv` (Python 3.13.9); CI is configured for Python 3.12. Two deprecation warnings originate from Starlette dependencies. These tests use fake providers and do not establish live SEC/Gemini compatibility or real retrieval quality. CI execution on GitHub still needs to be checked after pushing.

## Run on Windows

Run commands from the repository root. Python 3.12 is the project target. If `.venv` does not exist, create it with `py -3.12 -m venv .venv`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Set `GEMINI_API_KEY` in `.env`. `app/main.py` loads it before importing routers. SEC ingestion also uses the identifying `SEC_USER_AGENT` configured in `edgar_fetcher.py`. PDF ingestion does not contact SEC, but indexing still requires Gemini embeddings.

Use the same Python executable for installing dependencies, running the server, and running tests. A bare `python`, `pip`, or `uvicorn` can resolve to a different system/Conda environment. If startup reports `No module named 'pdfplumber'`, install requirements with the `.venv` command above, stop the failed reloader with Ctrl+C, and restart.

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- Stop the server: Ctrl+C

## API reference

| Method and path | Input | Result |
|---|---|---|
| `POST /v2/ingest` | JSON: `ticker`, optional `max_chunks` | Fetch and index the latest SEC 10-K; return filing date and chunk counts. |
| `POST /v2/ingest-pdf` | Multipart: `file`, `ticker`, optional `max_chunks` | Parse and index a PDF under a ticker or arbitrary label; return chunk counts. |
| `POST /v2/search` | JSON: `ticker`, `query`, optional `k` | Return ranked passages with section, section title, metadata, chunk type, text, and RRF score. |
| `POST /v2/graph/build` | JSON: `ticker` | Extract a graph from that ticker's already-indexed chunks, save it, and return node/relationship counts. |
| `GET /v2/graph/{ticker}/entity/{name}` | Ticker/label and exact entity name in URL | Return direct incoming and outgoing relationships. |

Both ingestion routes default to `max_chunks=10`, with an allowed range of 1 to 100. They index the first chunks in document order, not the most relevant chunks. `table_chunk_count` counts indexed tables; `total_chunk_count` counts all parsed chunks. The PDF response has no filing date because uploaded PDFs do not supply verified filing metadata.

Tickers/labels are stored in uppercase. Search and graph building use the same indexed chunks. Ingesting again under the same key replaces its search index; graph rebuilding is a separate operation.

### 1. Ingest SEC HTML

```powershell
$ingest = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v2/ingest" `
  -ContentType "application/json" `
  -Body '{"ticker":"AAPL","max_chunks":10}'
$ingest
```

Check `filing_date`, `chunk_count`, `total_chunk_count`, and `table_chunk_count`. Ten chunks are useful for a small initial test but may not reach risk factors or financial statements. Increasing the limit uses more embedding quota.

### 2. Alternatively, upload a PDF

In Swagger UI, select `POST /v2/ingest-pdf`, click **Try it out**, choose a PDF for `file`, enter `SAMPLE` for `ticker`, and set `max_chunks` to 10 or 100. Execute the request.

From PowerShell, use `curl.exe` (rather than the PowerShell `curl` alias):

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v2/ingest-pdf" `
  -F "file=@C:/path/to/filing.pdf;type=application/pdf" `
  -F "ticker=SAMPLE" `
  -F "max_chunks=100"
```

The PDF parser extracts selectable text and detected ruled tables. Empty, unreadable, encrypted-with-password, and image-only PDFs produce a clean 400 response. OCR is not implemented.

### 3. Search the indexed document

```powershell
$search = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v2/search" `
  -ContentType "application/json" `
  -Body '{"ticker":"AAPL","query":"What are the main products and services?","k":3}'
$search.results | Format-List section, section_title, chunk_type, metadata, rrf_score, text
```

Use `SAMPLE` instead of `AAPL` to search the uploaded PDF. `k` defaults to 5. Read the returned text and compare it with the original document. RRF scores express ranking, not confidence or probability. Missing evidence may be outside the indexed prefix.

### 4. Build the graph

```powershell
$graph = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v2/graph/build" `
  -ContentType "application/json" `
  -Body '{"ticker":"AAPL"}'
$graph
```

This makes additional Gemini generation calls over the indexed chunks. It does not fetch a filing or ingest missing chunks. It returns `ticker`, `node_count`, and `relationship_count`. An empty extracted graph can be a valid result.

### 5. Query relationships

Entity names are exact and case-sensitive. Use a name present in the extracted graph; the name below is illustrative and is not guaranteed to be extracted from a particular filing.

```powershell
$entity = [uri]::EscapeDataString("Apple Inc.")
$connections = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/v2/graph/AAPL/entity/$entity"
$connections.relationships | Format-List source, relation, target, doc
```

A relationship contains `source`, `relation`, `target`, and `doc` (source document ID). An unknown entity returns an empty relationship list. A missing ticker graph returns 404. There is currently no endpoint listing all entity names, and graph results do not automatically augment `/v2/search`.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `edgar_fetcher.py` | Resolve ticker to CIK, locate the latest 10-K, and download HTML with identifying headers and request timeouts. |
| `html_parser.py` | Convert tables to markdown pipes, preserve captions, normalize headings with inline elements, and flatten other HTML. |
| `pdf_parser.py` | Extract text and tables in reading order, omit table characters from prose to prevent duplication, and normalize bad/empty PDF errors. |
| `financial_chunker.py` | Detect Item sections, filter likely TOC entries, chunk prose, keep tables atomic, and attach table context. |
| `hybrid_retriever.py` | Define the embedding interface and real/fake providers; build BM25/dense indexes and combine search ranks with RRF. |
| `entity_graph.py` | Validate extracted JSON, build a NetworkX `MultiDiGraph`, find connected edges, and save/load node-link JSON. |
| `ingestion_api.py` | Expose ingestion, search, graph building, and graph lookup; load persisted graphs through the router lifespan. |

### Chunking and table context

`Chunk` carries `chunk_id`, `text`, `section`, `section_title`, `chunk_type`, and `metadata`. Prose is chunked at paragraph/word boundaries with a target maximum of 1,800 characters. Tables are never split to satisfy that limit.

Table chunks contain a header like this before the complete pipe table:

```text
Section: Item 7. Management's Discussion and Analysis
Caption: Consolidated Statements of Operations
Units: in millions

| Metric | 2025 | 2024 |
|---|---|---|
| Revenue | 412 | 365 |
```

Metadata includes `doc_id`, `caption`, `units`, and `section_title`. Caption and units detection is heuristic: it checks nearby preceding lines and unit labels in the table's first rows. Missing caption/units are `null` in metadata and omitted from the text header. Units are not inferred from numeric values. HTML captions are retained before conversion; PDFs use extracted surrounding text.

TOC filtering skips empty sections and short repeated Item entries or page-number-only bodies. Short genuine sections remain supported, but unusual filing layouts still need manual validation.

### Embedding batches and RRF

`HybridIndex.build()` sends batches of 32 texts. For API status codes 408, 429, 500, 502, 503, and 504, or HTTPX transport errors, it retries only the failed batch up to three times, waiting 1, 2, then 4 seconds. Other API errors fail immediately. Invalid embedding shapes are rejected. A failed rebuild preserves the previous index.

`GeminiEmbeddingProvider` uses `gemini-embedding-001` and also has an internal 100-text batch limit. `FakeEmbeddingProvider` generates repeatable hash-based vectors for tests; it does not model semantic similarity.

Search independently ranks BM25 keyword scores and cosine embedding similarity. For each chunk it adds:

```text
RRF score = 1 / (rrf_k + sparse_rank) + 1 / (rrf_k + dense_rank)
```

Ranks are zero-based and `rrf_k` defaults to 60. The API exposes the fused score but not the individual ranks. RRF uses ranks rather than mixing raw BM25 and cosine scores.

### Graph batches and persistence

`EntityGraph` accepts an injectable extractor. FastAPI exposes it through the `get_entity_extractor` dependency so tests can replace Gemini without network calls. The production extractor currently uses `gemini-3.1-flash-lite`, requests JSON, validates the returned entities/relationships, and sets a 60-second provider timeout.

Graph batches contain at most eight chunks and target roughly 12,000 characters. Batches do not mix document IDs; an oversized atomic table is sent intact. Graph extraction does not have the embedding retry/backoff policy.

Graphs are saved to `data/graphs/<sha256-of-ticker>.json` in NetworkX node-link format (`nodes` and `links`). The ticker is stored as graph metadata. Saving writes a temporary file followed by an atomic replacement; an extraction or disk failure leaves the previously published graph intact. The application reloads saved graphs on startup and logs/skips unreadable files. `/data/` is gitignored.

| Data | Survives server restart? |
|---|---|
| Parsed chunks and BM25/dense indexes in `_indexes` | No; ingest again before search or graph rebuilding. |
| Graph JSON in `data/graphs/` | Yes; relationship lookup works after startup reload without re-ingestion. |
| Uploaded original PDF | No; the application does not save the uploaded source file. |

Re-ingesting a ticker does not invalidate its old graph. Rebuild explicitly to refresh it. Storage and caches are intended for a single process; concurrent builds/multiple workers do not have coordination or cache synchronization.

## Errors and troubleshooting

| Situation | Current behavior |
|---|---|
| Unknown SEC ticker or no 10-K | SEC ingest returns 404. |
| SEC rejection/network failure | SEC ingest returns 502; check its detail and the configured user-agent. |
| Bad, empty, or image-only PDF | PDF ingest returns 400 with an explanation. |
| Missing multipart fields, blank PDF label, or invalid chunk limit | PDF ingest returns 422. |
| Search/build before ingestion | 404; ingest first. |
| Graph lookup before building/loading | 404; build first. |
| Missing Gemini key during ingestion/build | 503. |
| Gemini rate limit during ingestion/build | 503; check quota and retry later. |
| Other Gemini API failures during ingestion/build | 502. |
| Malformed graph extraction output | Graph build returns 502. |
| Graph disk persistence failure | Graph build returns 503 without publishing a partial graph. |

PDF ingestion and graph building also translate transport failures into 502 responses. Error handling is not yet uniform: `/v2/search` does not currently translate Gemini failures, and the SEC ingest route does not catch all embedding transport failures. The synchronous SEC ingestion/search work still runs inside async routes and can block the event loop. These remain implementation gaps.

## Tests and CI

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest tests/test_html_parser.py tests/test_financial_chunker.py tests/test_hybrid_retriever.py -v
```

| Test file | Coverage |
|---|---|
| `tests/test_html_parser.py` | Empty HTML, table values/captions/order, and heading/whitespace normalization. |
| `tests/test_financial_chunker.py` | TOC filtering, short/unknown sections, long prose, paragraph packing, atomic tables, and table context. |
| `tests/test_hybrid_retriever.py` | Batch ordering, retries/backoff, failed rebuilds, shape validation, exact RRF scores/ranks, result limits, and repeatable fake-provider search. |
| `tests/test_pdf_parser.py` | A PDF generated in the test, extraction order, blank cells, upload/search, invalid PDFs, and provider transport errors. |
| `tests/test_entity_graph_api.py` | Fake extraction, batching, both relationship directions, persistence/startup reload, malformed output, provider errors, and disk failure. |

`.github/workflows/ci.yml` installs `requirements.txt` and runs `python -m pytest` on every push using Ubuntu and Python 3.12. No Gemini credentials or SEC requests are required for this suite.

## Remaining work and next milestone

- Validate live SEC filings, real embeddings, and entity extraction; offline tests verify mechanics rather than answer quality.
- Index beyond the current 100-chunk API cap or choose an explicit document coverage policy.
- Persist search indexes and source documents; coordinate updates across workers.
- Improve complex HTML/PDF layouts, nested tables, merged cells, borderless tables, and multipage tables. PDF OCR is absent; a table spanning pages may be detected as separate tables.
- Improve caption/units and section detection, and evaluate prose overlap where context crosses boundaries.
- Return `chunk_id`, filing date, and source URL in search results for complete citations. Section title and document ID are already available.
- Validate search `k`, standardize provider error responses, and move blocking work out of async routes.
- Add entity enumeration, aliases/case normalization, deduplication, chunk-level graph provenance, and graph freshness tracking. Current lookups are one hop and repeated extracted edges may remain parallel edges.
- Move the hardcoded SEC user-agent into configuration.
- Connect top retrieved chunks to Phase 1 prompts and citations, instruct the model to acknowledge insufficient evidence, and measure retrieval precision/recall and answer faithfulness on a small evaluation set.

The next milestone is retrieval-grounded generation: ingest, retrieve evidence, pass it into Phase 1, and stream an answer whose citations identify the source chunks.
