from io import BytesIO

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

from app.phase2 import ingestion_api
from app.phase2.financial_chunker import chunk_filing
from app.phase2.hybrid_retriever import FakeEmbeddingProvider
from app.phase2.pdf_parser import pdf_to_text_preserving_tables


def make_pdf(blank=False):
    stream = BytesIO()
    pdf = canvas.Canvas(stream)
    if not blank:
        pdf.drawString(50, 800, "Item 7. Management's Discussion and Analysis")
        pdf.drawString(50, 775, "Consolidated Statements of Operations")
        pdf.drawString(50, 755, "(in millions)")
        table = Table([
            ["Metric", "2025", "2024"],
            ["Revenue", "412", "365"],
            ["Cash", "", "100"],
        ], colWidths=[150, 80, 80], rowHeights=25)
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
        table.wrapOn(pdf, 400, 100)
        table.drawOn(pdf, 50, 650)
        pdf.drawString(50, 620, "Operating expenses increased modestly.")
        pdf.showPage()
        pdf.drawString(50, 800, "Item 8. Financial Statements")
        pdf.drawString(50, 775, "See accompanying notes.")
    pdf.showPage()
    pdf.save()
    return stream.getvalue()


def test_pdf_tables_and_context_in_reading_order():
    text = pdf_to_text_preserving_tables(make_pdf())
    assert "| Metric | 2025 | 2024 |\n|---|---|---|" in text
    assert text.count("Revenue") == 1
    assert "| Cash |  | 100 |" in text
    assert text.index("(in millions)") < text.index("| Metric") < text.index("Operating expenses")
    chunks = chunk_filing(text, "test")
    tables = [chunk for chunk in chunks if chunk.chunk_type == "table"]
    assert len(tables) == 1
    assert tables[0].metadata["caption"] == "Consolidated Statements of Operations"
    assert tables[0].metadata["units"] == "in millions"
    assert tables[0].section == "Item 7"
    assert "Units: in millions" in tables[0].text
    assert chunks[-1].section == "Item 8"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setattr(ingestion_api, "GeminiEmbeddingProvider", FakeEmbeddingProvider)
    monkeypatch.setattr(ingestion_api, "_indexes", {})
    app = FastAPI()
    app.include_router(ingestion_api.router)
    with TestClient(app) as client:
        yield client


def test_upload_then_search(client):
    response = client.post(
        "/v2/ingest-pdf", data={"ticker": " sample ", "max_chunks": 100},
        files={"file": ("sample.pdf", make_pdf(), "application/pdf")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["ticker"] == "SAMPLE"
    assert response.json()["table_chunk_count"] == 1
    response = client.post("/v2/search", json={"ticker": "sample", "query": "Revenue", "k": 10})
    assert response.status_code == 200
    table = next(r for r in response.json()["results"] if r["chunk_type"] == "table")
    assert table["metadata"]["units"] == "in millions"


@pytest.mark.parametrize("data", [b"", b"not a PDF", b"%PDF-1.7\ntruncated", make_pdf(blank=True)])
def test_bad_or_empty_pdf_is_clean_400(client, data):
    response = client.post(
        "/v2/ingest-pdf", data={"ticker": "bad"},
        files={"file": ("bad.pdf", data, "application/pdf")},
    )
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str)
    assert "BAD" not in ingestion_api._indexes


def test_blank_label_is_rejected(client):
    response = client.post(
        "/v2/ingest-pdf", data={"ticker": " "},
        files={"file": ("sample.pdf", make_pdf(), "application/pdf")},
    )
    assert response.status_code == 422


def test_embedding_connection_failure_is_clean_502(client, monkeypatch):
    class UnreachableProvider:
        def embed(self, texts):
            raise httpx.ConnectError("offline")

    monkeypatch.setattr(ingestion_api, "GeminiEmbeddingProvider", UnreachableProvider)
    monkeypatch.setattr("app.phase2.hybrid_retriever.time.sleep", lambda _: None)
    response = client.post(
        "/v2/ingest-pdf", data={"ticker": "bad"},
        files={"file": ("sample.pdf", make_pdf(), "application/pdf")},
    )
    assert response.status_code == 502
    assert "BAD" not in ingestion_api._indexes
