"""
app/phase2/entity_graph.py

"GraphRAG-lite": extracts structured entities and relationships from
filing chunks using Gemini's structured JSON output, and stores them
in an in-memory graph (networkx) so queries like "which filings
mention supply-chain dependency on Company X" become graph traversals
instead of hoping a vector search happens to surface the connection.

Production upgrade path: swap networkx (in-memory, resets on restart)
for Neo4j or Amazon Neptune once the entity count outgrows a single
process.
"""

import json
import os
import hashlib
import logging
from pathlib import Path
import tempfile
from typing import Callable

import networkx as nx
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

GRAPH_BATCH_SIZE = 8
GRAPH_BATCH_CHARS = 12000


class Entity(BaseModel):
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)


class Relationship(BaseModel):
    source: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    target: str = Field(min_length=1)


class Extraction(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]

ENTITY_EXTRACTION_PROMPT = """Extract entities and relationships from this SEC filing excerpt.
Return ONLY valid JSON, no markdown fences, matching this shape:
{{
  "entities": [{{"name": "...", "type": "company|subsidiary|executive|risk_factor|product"}}],
  "relationships": [{{"source": "...", "relation": "...", "target": "..."}}]
}}

Filing excerpt:
{chunk_text}
"""
# model = "gemini-2.5-flash"
model = "gemini-3.1-flash-lite"

def extract_entities(chunk_text: str) -> dict:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={"timeout": 60000})
    response = client.models.generate_content(
        # model = "gemini-2.5-flash",
        model = "gemini-3.1-flash-lite",
        contents=ENTITY_EXTRACTION_PROMPT.format(chunk_text=chunk_text),
        config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
    )
    return json.loads(response.text)


class EntityGraph:
    def __init__(self, extractor: Callable[[str], dict] | None = None):
        self.graph = nx.MultiDiGraph()
        self.extractor = extractor or extract_entities

    def ingest_chunk(self, chunk_text: str, doc_id: str) -> None:
        extracted = Extraction.model_validate(self.extractor(chunk_text))
        for entity in extracted.entities:
            self.graph.add_node(entity.name, type=entity.type)
        for rel in extracted.relationships:
            self.graph.add_edge(rel.source, rel.target, relation=rel.relation, source_doc=doc_id)

    def ingest_chunks(self, chunks) -> None:
        batch, size, doc_id = [], 0, None
        for chunk in chunks:
            source_doc = chunk.metadata.get("doc_id", chunk.chunk_id)
            if batch and (len(batch) >= GRAPH_BATCH_SIZE or size + len(chunk.text) > GRAPH_BATCH_CHARS or source_doc != doc_id):
                self.ingest_chunk("\n\n".join(batch), doc_id)
                batch, size = [], 0
            batch.append(chunk.text)
            size += len(chunk.text) + 2
            doc_id = source_doc
        if batch:
            self.ingest_chunk("\n\n".join(batch), doc_id)

    def find_connected(self, entity_name: str) -> list[dict]:
        if entity_name not in self.graph:
            return []
        edges = []
        for u, v, data in self.graph.edges(entity_name, data=True):
            edges.append({"source": u, "relation": data["relation"], "target": v, "doc": data["source_doc"]})
        for u, v, data in self.graph.in_edges(entity_name, data=True):
            if u == v:
                continue
            edges.append({"source": u, "relation": data["relation"], "target": v, "doc": data["source_doc"]})
        return edges


def save_graph(directory: Path, ticker: str, graph: EntityGraph) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    graph.graph.graph["ticker"] = ticker
    payload = nx.node_link_data(graph.graph, edges="links")
    filename = hashlib.sha256(ticker.encode()).hexdigest() + ".json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False)
        os.replace(temporary, directory / filename)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_graphs(directory: Path) -> dict[str, EntityGraph]:
    graphs = {}
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            graph = EntityGraph()
            graph.graph = nx.node_link_graph(data, edges="links")
            ticker = graph.graph.graph["ticker"]
            if not isinstance(ticker, str) or not graph.graph.is_directed() or not graph.graph.is_multigraph():
                raise ValueError("Invalid saved graph")
            for _, _, attrs in graph.graph.edges(data=True):
                if not all(isinstance(attrs.get(key), str) for key in ("relation", "source_doc")):
                    raise ValueError("Invalid saved relationship")
            graphs[ticker] = graph
        except (ValueError, KeyError, TypeError, OSError, nx.NetworkXException):
            logging.getLogger(__name__).warning("Skipping unreadable graph: %s", path)
    return graphs


if __name__ == "__main__":
    eg = EntityGraph()
    eg.ingest_chunk(
        "Acme Corp relies on Nexus Components as its sole supplier of "
        "pressure sensors used across its industrial product line.",
        doc_id="ACME-10K-2024",
    )
    print(eg.find_connected("Acme Corp"))
