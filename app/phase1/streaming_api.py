"""
streaming_api.py

Minimal FastAPI SSE endpoint demonstrating the streaming contract the
rest of the system builds on. Phase 3 wraps this with auth, rate
limiting, and model routing; Phase 5 wraps it again with guardrails.

For a .NET dev: this is roughly the shape of an IAsyncEnumerable<T>
action result in ASP.NET Core streamed over `text/event-stream` —
FastAPI's StreamingResponse is the equivalent primitive.
"""

import json
import os
import time
from dataclasses import dataclass

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from google import genai
from pydantic import BaseModel

router = APIRouter(prefix="/v1", tags=["query"])

_gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
# GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MODEL = "gemini-3.1-flash-lite"


class QueryRequest(BaseModel):
    query: str
    document_id: str | None = None


@dataclass
class SSEEvent:
    event: str          # "token" | "citation" | "metric" | "done" | "error"
    data: dict

    def encode(self) -> str:
        return f"event: {self.event}\ndata: {json.dumps(self.data)}\n\n"


async def real_llm_token_stream(query: str):
    """
    Real Gemini call, streamed. Phase 3's Model Gateway will wrap this
    with retries, provider fallback, and semantic caching — this is
    the bare, single-provider version for Phase 1.
    """
    system_instruction = (
        "You are a financial analysis assistant. Only discuss information "
        "that would come from the provided filing context. Never state a "
        "specific number as fact unless it was given to you in the context."
    )

    stream = await _gemini_client.aio.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=query,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )
    async for chunk in stream:
        if chunk.text:
            yield chunk.text


async def event_generator(req: QueryRequest):
    start = time.perf_counter()
    first_token_sent = False

    yield SSEEvent("citation", {
        "source": req.document_id or "10-K-2025-Q4",
        "section": "Item 7 - MD&A",
    }).encode()

    buffer = ""
    try:
        async for token in real_llm_token_stream(req.query):
            if not first_token_sent:
                ttft_ms = int((time.perf_counter() - start) * 1000)
                yield SSEEvent("metric", {"ttft_ms": ttft_ms}).encode()
                first_token_sent = True

            buffer += token
            yield SSEEvent("token", {"text": token}).encode()
    except Exception as exc:
        yield SSEEvent("error", {"message": "Model provider error. Please retry."}).encode()
        return

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
            "X-Accel-Buffering": "no",
        },
    )