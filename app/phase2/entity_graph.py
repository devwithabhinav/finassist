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

import networkx as nx
from google import genai
from google.genai import types

_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

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
    response = _client.models.generate_content(
        # model = "gemini-2.5-flash",
        model = "gemini-3.1-flash-lite",
        contents=ENTITY_EXTRACTION_PROMPT.format(chunk_text=chunk_text),
        config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
    )
    return json.loads(response.text)


class EntityGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def ingest_chunk(self, chunk_text: str, doc_id: str) -> None:
        extracted = extract_entities(chunk_text)
        for entity in extracted.get("entities", []):
            self.graph.add_node(entity["name"], type=entity["type"])
        for rel in extracted.get("relationships", []):
            self.graph.add_edge(rel["source"], rel["target"], relation=rel["relation"], source_doc=doc_id)

    def find_connected(self, entity_name: str) -> list[dict]:
        if entity_name not in self.graph:
            return []
        edges = []
        for u, v, data in self.graph.edges(entity_name, data=True):
            edges.append({"source": u, "relation": data["relation"], "target": v, "doc": data["source_doc"]})
        for u, v, data in self.graph.in_edges(entity_name, data=True):
            edges.append({"source": u, "relation": data["relation"], "target": v, "doc": data["source_doc"]})
        return edges


if __name__ == "__main__":
    eg = EntityGraph()
    eg.ingest_chunk(
        "Acme Corp relies on Nexus Components as its sole supplier of "
        "pressure sensors used across its industrial product line.",
        doc_id="ACME-10K-2024",
    )
    print(eg.find_connected("Acme Corp"))