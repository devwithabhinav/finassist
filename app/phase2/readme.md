**Phase 2 Intent**

Phase 2 builds the retrieval half of a financial RAG system:

```text
SEC ticker
  -> CIK lookup
  -> latest 10-K HTML
  -> cleaned text with tables
  -> section-aware chunks
  -> BM25 + embeddings
  -> hybrid search results
```

The main API is wired through `main.py:1-37`:

- `POST /v2/ingest`: fetches and indexes a filing.
- `POST /v2/search`: searches the indexed filing.

## Module and Symbol Responsibilities

### `edgar_fetcher.py:1-77`

This module obtains real SEC filings.

- `get_cik_for_ticker(ticker)`: converts a ticker such as `AAPL` into its SEC CIK identifier.
- `get_latest_10k_url(cik)`: finds the latest `10-K` submission and constructs its document URL.
- `fetch_filing_html(document_url)`: downloads the filing HTML.
- `fetch_latest_10k(ticker)`: coordinates the complete fetch process and returns filing metadata plus HTML.
- `SEC_USER_AGENT` and `HEADERS`: identify the client to SEC EDGAR, as required by SEC policy.

You learned that external data sources often require proper identification, rate limiting, timeouts, and error handling.

### `html_parser.py:1-62`

This module transforms HTML into searchable text.

- `_table_to_markdown(table_tag)`: converts HTML tables into pipe-delimited Markdown-like text.
- `html_to_text_preserving_tables(html)`: removes HTML formatting while preserving table content.

The important design decision is that tables are financial evidence, not decoration. Converting them to text makes values searchable while retaining rows and columns reasonably well.

### `financial_chunker.py:1-145`

This module creates retrieval-ready units.

- `Chunk`: stores the text and metadata for one searchable unit.
- `split_into_sections(text)`: identifies sections such as Item 1, Item 1A, and Item 7.
- `split_section_preserving_tables(section_text)`: keeps tables intact and chunks prose separately.
- `_chunk_prose(text)`: divides long prose into chunks up to `MAX_CHUNK_CHARS`.
- `chunk_filing(text, doc_id)`: builds the final list of `Chunk` objects.

You learned that chunking is not merely splitting text every N characters. Financial documents have meaningful boundaries:

- SEC section boundaries
- paragraph boundaries
- table boundaries
- document and filing metadata

The `Chunk` metadata later supports filtering, citations, and debugging.

### `hybrid_retriever.py:1-146`

This module searches the chunks using two retrieval strategies.

- `EmbeddingProvider`: abstract interface for any embedding implementation.
- `GeminiEmbeddingProvider`: produces real semantic embeddings through Gemini.
- `FakeEmbeddingProvider`: produces deterministic test vectors without network access.
- `SearchResult`: combines a chunk with its sparse rank, dense rank, and RRF score.
- `HybridIndex.build(chunks)`: builds the BM25 and embedding indexes.
- `HybridIndex.search(query, k, rrf_k)`: performs both searches and combines their rankings.

The two search types have different strengths:

- BM25 finds exact terms, names, numbers, and phrases.
- Embeddings find semantically similar wording.
- Reciprocal Rank Fusion combines their rankings without comparing incompatible raw scores.

One important lesson: the fake embedding provider is useful for testing mechanics, but it does not understand meaning. It should never be used to judge real retrieval quality.

### `ingestion_api.py:1-122`

This module exposes the pipeline through FastAPI.

- `IngestRequest`: validates the ticker and chunk limit.
- `IngestResponse`: reports filing and chunk statistics.
- `ingest_filing(req)`: performs fetch, parse, chunk, embedding, and in-memory indexing.
- `SearchRequest`: validates search input.
- `search_filing(req)`: retrieves results from the ticker’s in-memory index.

The `_indexes` dictionary is a simple cache:

```python
_indexes[ticker] = HybridIndex(...)
```

It is appropriate for learning, but the index disappears when the server restarts and cannot be safely shared across multiple workers.

### `entity_graph.py:1-76`

This is an additional GraphRAG experiment.

- `extract_entities(chunk_text)`: asks Gemini to extract entities and relationships as JSON.
- `EntityGraph`: stores those relationships in a NetworkX `MultiDiGraph`.
- `EntityGraph.ingest_chunk(...)`: extracts and stores entities and edges.
- `EntityGraph.find_connected(entity_name)`: returns directly connected incoming and outgoing relationships.

This supports questions such as:

> Which suppliers are connected to this company?

It is different from vector search. Vector search finds relevant passages; the graph represents explicit relationships.

However, this module is currently not connected to `/v2/ingest` or `/v2/search`. It is a standalone experiment.

## What You Have Learned

Phase 2 has taught you several important AI engineering concepts:

1. Real-world documents require ingestion and normalization before an LLM can use them.
2. Financial tables must be preserved because they contain critical evidence.
3. Chunk size and boundaries directly affect retrieval quality.
4. Metadata is essential for traceability and future citations.
5. BM25 and embeddings solve different retrieval problems.
6. Hybrid retrieval is often more reliable than relying on only one search method.
7. RRF combines rankings rather than incompatible raw scores.
8. Embedding providers should be abstracted behind an interface.
9. External APIs require timeouts, headers, rate limits, and useful error responses.
10. Retrieval is not generation. Phase 2 finds evidence, but it does not yet produce a grounded answer.

Together, Phase 1 and Phase 2 are close to a complete RAG foundation:

```text
Phase 2: retrieve financial evidence
Phase 1: generate and stream an answer
Next: connect the retrieved evidence to the model prompt
```

## Open Questions and Missing Pieces

The biggest open question is:

> How will retrieved chunks be passed into Phase 1 so Gemini answers only from filing evidence and cites the source?

Several implementation gaps remain:

- Only the first `max_chunks` are indexed, so the default request searches only the first 10 chunks of a filing.
- `_indexes` is in memory and disappears after restart.
- Search results do not return `chunk_id`, filing date, section title, or document URL, so citations are incomplete.
- The parser does not fully handle `rowspan`, `colspan`, nested tables, or complex SEC formatting.
- The section regex may miss unusual headings or accidentally match text that resembles an SEC heading.
- The chunker has no overlap between prose chunks, which can lose context at boundaries.
- The Gemini embedding calls are synchronous inside `async` FastAPI endpoints.
- `SearchRequest.k` has no validation range.
- There are no automated tests for parser correctness, chunk boundaries, ranking behavior, API errors, or citation metadata.
- `entity_graph.py` has no error handling for malformed model JSON and currently supports only direct one-hop lookups.
- The entity graph is not persisted and is not integrated with the filing index.
- The hardcoded SEC user-agent contains a personal email address and should eventually come from configuration.
- The current retrieval system returns passages but does not evaluate precision, recall, ranking quality, or answer faithfulness.

## Recommended Next Milestone

The next meaningful milestone is to connect the systems:

1. Ingest the full filing or use a deliberate indexing limit.
2. Search for the user’s question.
3. Pass the top chunks into the Phase 1 Gemini prompt.
4. Return citations containing document ID, filing date, section, and chunk ID.
5. Instruct the model to say when the evidence is insufficient.
6. Add a small evaluation set with known questions and expected filing passages.

That would turn Phase 2 from a search demonstration into a genuinely grounded financial question-answering system.

---------------------------------------------------------------------------------------------

**Recommended Workflow**

Work through the pipeline in small steps. Keep the server running in one PowerShell window so the in-memory index remains available while you search.

```text
+-----------------------------------------------+
| 1. Start the FastAPI app                      |
+----------------------+------------------------+
                       |
                       v
+-----------------------------------------------+
| 2. POST /v2/ingest with a company ticker       |
+----------------------+------------------------+
                       |
                       v
+-----------------------------------------------+
| 3. Find the company's latest 10-K on SEC EDGAR |
+----------------------+------------------------+
                       |
                       v
+-----------------------------------------------+
| 4. Download HTML and convert it to clean text  |
|    while preserving tables                    |
+----------------------+------------------------+
                       |
                       v
+-----------------------------------------------+
| 5. Split text into section-aware chunks        |
|    and keep each table together                |
+----------------------+------------------------+
                       |
                       v
+-----------------------------------------------+
| 6. Build keyword (BM25) and embedding indexes  |
+----------------------+------------------------+
                       |
                       v
+-----------------------------------------------+
| 7. POST /v2/search with a question             |
+----------------------+------------------------+
                       |
                       v
+-----------------------------------------------+
| 8. Rank results and combine rankings with RRF  |
+----------------------+------------------------+
                       |
                       v
+-----------------------------------------------+
| 9. Inspect the retrieved filing passages       |
+----------------------+------------------------+
                       |
                       v
+-----------------------------------------------+
| 10. Next: pass evidence to Phase 1 and answer  |
|     with citations to the filing               |
+-----------------------------------------------+
```

1. **Prepare and start the app.** Confirm `.env` contains a valid `GEMINI_API_KEY`, and that `SEC_USER_AGENT` in `edgar_fetcher.py` identifies you with a contact email. From the project root, run:

	```powershell
	.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
	```

2. **Ingest one filing.** In a second PowerShell window, fetch and index a filing. Start with a familiar public company, such as Apple:

	```powershell
	$ingest = Invoke-RestMethod -Method Post `
	  -Uri "http://127.0.0.1:8000/v2/ingest" `
	  -ContentType "application/json" `
	  -Body '{"ticker":"AAPL","max_chunks":10}'
	$ingest
	```

	Check the returned filing date and chunk counts. `chunk_count` is how many chunks were indexed; `total_chunk_count` is how many the full filing produced. The default selects the first 10 chunks in filing order, not the 10 most relevant chunks.

3. **Search for evidence.** Ask a question that should be answered by a specific filing section:

	```powershell
	$search = Invoke-RestMethod -Method Post `
	  -Uri "http://127.0.0.1:8000/v2/search" `
	  -ContentType "application/json" `
	  -Body '{"ticker":"AAPL","query":"What risks does the company describe?","k":5}'
	$search.results | Format-List section, chunk_type, rrf_score, text
	```

	Read the returned text and compare it with the filing. The endpoint returns retrieved passages and scores; it does not ask Gemini to write an answer.

4. **Evaluate retrieval.** Try several questions: one using exact terms from the filing, one phrased differently from the filing, and one about a table value. For each, note whether a useful passage appears near the top and whether its section and table/prose type make sense. Treat incorrect or missing results as a retrieval issue to investigate, not as proof that the filing lacks the information.

5. **Improve one thing at a time.** If useful evidence is missing, inspect the parsed text first, then the chunks, then the search rankings. Adjust parsing, chunk boundaries, or retrieval only after identifying which stage lost or failed to find the evidence. Retest the same questions after each change so results are comparable.

6. **Connect retrieval to Phase 1.** Once search returns useful evidence consistently, pass the top chunks into the Phase 1 model prompt as filing context. Return citations from the retrieved chunks' section and document metadata, and check that the generated response is supported by those passages. This is the next milestone: retrieval-grounded generation rather than search alone.

Remember: the index is in memory. If the server restarts, ingest the ticker again before searching. Avoid sending many SEC requests in quick succession; the fetcher includes a small delay to respect SEC request limits.