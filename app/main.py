"""
app/main.py

The single FastAPI application instance for the whole project.
Every phase adds its router here instead of creating its own app.
This is the equivalent of Program.cs wiring up controllers in
ASP.NET Core.
"""

from fastapi import FastAPI

from app.phase1.streaming_api import router as phase1_router

app = FastAPI(
    title="FinAssist - Enterprise Financial Intelligence & Portfolio Assistant",
    version="0.1.0",
)

app.include_router(phase1_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
