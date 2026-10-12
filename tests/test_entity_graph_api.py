import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.genai.errors import APIError

from app.phase2 import ingestion_api as api
from app.phase2.entity_graph import GRAPH_BATCH_SIZE, EntityGraph, load_graphs
from app.phase2.financial_chunker import Chunk


def extraction(text):
    return {
        "entities": [{"name": "Acme", "type": "company"}, {"name": "Nexus", "type": "company"}],
        "relationships": [{"source": "Acme", "relation": "depends_on", "target": "Nexus"}],
    }


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "GRAPH_DATA_DIR", tmp_path / "graphs")
    monkeypatch.setattr(api, "_graphs", {})
    chunks = [Chunk(str(i), f"Acme excerpt {i}", "Item 1", "Business", "prose", {"doc_id": "ACME-10K"}) for i in range(GRAPH_BATCH_SIZE + 1)]
    monkeypatch.setattr(api, "_indexes", {"ACME": SimpleNamespace(chunks=chunks)})
    app = FastAPI()
    app.include_router(api.router)
    fake = Mock(side_effect=extraction)
    app.dependency_overrides[api.get_entity_extractor] = lambda: fake
    return app, fake


def test_build_batches_queries_and_startup_reload(setup):
    app, fake = setup
    with TestClient(app) as client:
        response = client.post("/v2/graph/build", json={"ticker": "acme"})
        assert response.status_code == 200, response.text
        assert response.json() == {"ticker": "ACME", "node_count": 2, "relationship_count": 2}
        assert fake.call_count == 2
        assert "Acme excerpt 0" in fake.call_args_list[0].args[0]
        assert fake.call_args_list[1].args[0] == f"Acme excerpt {GRAPH_BATCH_SIZE}"
        outgoing = client.get("/v2/graph/acme/entity/Acme").json()["relationships"]
        incoming = client.get("/v2/graph/ACME/entity/Nexus").json()["relationships"]
        assert outgoing == incoming
        assert outgoing[0] == {"source": "Acme", "relation": "depends_on", "target": "Nexus", "doc": "ACME-10K"}
        assert client.get("/v2/graph/ACME/entity/Unknown").json()["relationships"] == []
    files = list(api.GRAPH_DATA_DIR.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert "nodes" in payload and "links" in payload
    api._graphs.clear()
    api._indexes.clear()
    with TestClient(app) as client:
        assert client.get("/v2/graph/ACME/entity/Nexus").json()["relationships"] == incoming
    assert fake.call_count == 2


@pytest.mark.parametrize("error,status", [
    (APIError(429, httpx.Response(429, json={})), 503),
    (APIError(500, httpx.Response(500, json={})), 502),
    (httpx.ConnectError("offline"), 502),
    (ValueError("bad JSON"), 502),
    (RuntimeError("missing key"), 503),
])
def test_failed_rebuild_preserves_saved_graph(setup, error, status):
    app, fake = setup
    with TestClient(app) as client:
        assert client.post("/v2/graph/build", json={"ticker": "ACME"}).status_code == 200
        saved = next(api.GRAPH_DATA_DIR.glob("*.json")).read_bytes()
        previous = api._graphs["ACME"]
        fake.side_effect = [extraction(""), error]
        response = client.post("/v2/graph/build", json={"ticker": "ACME"})
        assert response.status_code == status
        assert api._graphs["ACME"] is previous
        assert next(api.GRAPH_DATA_DIR.glob("*.json")).read_bytes() == saved


def test_missing_ticker_and_malformed_response(setup):
    app, fake = setup
    with TestClient(app) as client:
        assert client.post("/v2/graph/build", json={"ticker": "missing"}).status_code == 404
        assert client.get("/v2/graph/missing/entity/Acme").status_code == 404
        fake.assert_not_called()
        fake.return_value = None
        fake.side_effect = lambda _: {"entities": [{}], "relationships": []}
        assert client.post("/v2/graph/build", json={"ticker": "ACME"}).status_code == 502
        assert "ACME" not in api._graphs


def test_disk_failure_does_not_publish_graph(setup, monkeypatch):
    app, _ = setup
    monkeypatch.setattr(api, "save_graph", Mock(side_effect=OSError("disk full")))
    with TestClient(app) as client:
        assert client.post("/v2/graph/build", json={"ticker": "ACME"}).status_code == 503
        assert "ACME" not in api._graphs


def test_corrupt_graph_is_skipped(tmp_path):
    (tmp_path / "bad.json").write_text("not JSON")
    assert load_graphs(tmp_path) == {}


def test_batches_do_not_mix_document_ids_or_split_tables():
    fake = Mock(side_effect=extraction)
    graph = EntityGraph(fake)
    graph.ingest_chunks([
        Chunk("a", "| row | value |\n" * 1000, "Item 7", "MD&A", "table", {"doc_id": "first"}),
        Chunk("b", "Other excerpt", "Item 8", "Statements", "prose", {"doc_id": "second"}),
    ])
    assert fake.call_count == 2
    assert fake.call_args_list[0].args[0] == "| row | value |\n" * 1000
    assert {edge["doc"] for edge in graph.find_connected("Acme")} == {"first", "second"}
