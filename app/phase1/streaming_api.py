"""
streaming_api.py

Minimal FastAPI SSE endpoint demonstrating the streaming contract the
rest of the system builds on. Phase 3 wraps this with auth, rate
limiting, and model routing; Phase 5 wraps it again with guardrails.

For a .NET dev: this is roughly the shape of an IAsyncEnumerable<T>
action result in ASP.NET Core streamed over `text/event-stream` —
FastAPI's StreamingResponse is the equivalent primitive.
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# APIRouter is FastAPI's equivalent of an ASP.NET Core Controller:
# a self-contained group of endpoints that main.py mounts onto the
# single app instance, instead of every file creating its own app.
router = APIRouter(prefix="/v1", tags=["query"])


class QueryRequest(BaseModel):
    query: str
    document_id: str | None = None


@dataclass
class SSEEvent:
    event: str          # "token" | "citation" | "metric" | "done" | "error"
    data: dict

    def encode(self) -> str:
        # SSE wire format: "event: <name>\ndata: <json>\n\n"
        return f"event: {self.event}\ndata: {json.dumps(self.data)}\n\n"


async def fake_llm_token_stream(query: str):
    """
    Stand-in for the real model call (Phase 3 replaces this with the
    Model Gateway). Yields tokens one at a time to simulate generation.
    """
    response_text = (
        f"Based on the retrieved filing sections, here is the analysis "
        f"relevant to: '{query}'. Revenue grew year-over-year, driven "
        f"primarily by the cloud services segment."
    )
    for word in response_text.split(" "):
        await asyncio.sleep(0.03)  # simulated per-token generation latency
        yield word + " "


async def event_generator(req: QueryRequest):
    start = time.perf_counter()
    first_token_sent = False

    # In the real pipeline: retrieval happens here BEFORE generation
    # starts, and citation events are emitted as sources are pulled,
    # not invented after the fact.
    yield SSEEvent("citation", {
        "source": req.document_id or "10-K-2025-Q4",
        "section": "Item 7 - MD&A",
    }).encode()

    buffer = ""
    async for token in fake_llm_token_stream(req.query):
        if not first_token_sent:
            ttft_ms = int((time.perf_counter() - start) * 1000)
            yield SSEEvent("metric", {"ttft_ms": ttft_ms}).encode()
            first_token_sent = True

        buffer += token
        yield SSEEvent("token", {"text": token}).encode()

    total_ms = int((time.perf_counter() - start) * 1000)
    yield SSEEvent("done", {
        "total_ms": total_ms,
        "full_text": buffer.strip(),
        "disclaimer": "Generated analysis — not financial advice. Verify against source filing.",
    }).encode()


@router.post("/query/stream")
async def stream_query(req: QueryRequest):
    return StreamingResponse(
        event_generator(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering, critical for real TTFT
        },
    )
