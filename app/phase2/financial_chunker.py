"""
app/phase2/financial_chunker.py

Splits a 10-K's text into retrieval-ready chunks that respect two
boundaries naive fixed-size chunking ignores:
  1. Section structure (Item 1, Item 1A, Item 7, Item 8...)
  2. Table boundaries (a table is one atomic chunk, never split)
"""

import re
from dataclasses import dataclass, field

SECTION_HEADER_RE = re.compile(r"^\s*Item\s+(\d+[A-Z]?)\.\s*(.*)$", re.IGNORECASE | re.MULTILINE)
TABLE_BLOCK_RE = re.compile(r"(?:^\|.*\|\s*$\n?){2,}", re.MULTILINE)

MAX_CHUNK_CHARS = 1800
MIN_CHUNK_CHARS = 200

TABLE_UNITS_RE = re.compile(
    r"\b(?:in\s+(?:(?:U\.?S\.?\s+)?dollars?\s+in\s+)?"
    r"(?:thousands|millions|billions)|"
    r"(?:U\.?S\.?\s+)?dollars?\s+in\s+(?:thousands|millions|billions))\b"
    r"[^\n|]*",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    section: str
    section_title: str
    chunk_type: str
    metadata: dict = field(default_factory=dict)


def split_into_sections(text: str) -> list[tuple[str, str, str]]:
    matches = list(SECTION_HEADER_RE.finditer(text))
    if not matches:
        return [("Unknown", "Full document", text)]

    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        item_number = m.group(1)
        title = m.group(2).strip() or "(untitled)"
        section_text = text[start:end].strip()
        # TOCs repeat the real Item headings with little or no intervening
        # content (often just page numbers or a Part heading). Do not discard
        # short real sections merely because they are short.
        repeated_later = any(
            later.group(1).upper() == item_number.upper()
            for later in matches[i + 1:]
        )
        if not section_text or (
            len(section_text) < MIN_CHUNK_CHARS
            and (repeated_later or section_text.isdecimal())
        ):
            continue
        sections.append((item_number, title, section_text))
    return sections


def split_section_preserving_tables(section_text: str) -> list[tuple[str, str]]:
    pieces: list[tuple[str, str]] = []
    last_end = 0

    for table_match in TABLE_BLOCK_RE.finditer(section_text):
        prose_before = section_text[last_end:table_match.start()].strip()
        if prose_before:
            pieces.extend(("prose", p) for p in _chunk_prose(prose_before))
        pieces.append(("table", table_match.group().strip()))
        last_end = table_match.end()

    trailing_prose = section_text[last_end:].strip()
    if trailing_prose:
        pieces.extend(("prose", p) for p in _chunk_prose(trailing_prose))

    return pieces


def _chunk_prose(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks, current = [], ""

    for para in paragraphs:
        if len(para) > MAX_CHUNK_CHARS:
            if current:
                chunks.append(current)
                current = ""
            remaining = para
            while len(remaining) > MAX_CHUNK_CHARS:
                split_at = remaining.rfind(" ", 0, MAX_CHUNK_CHARS + 1)
                if split_at <= 0:
                    split_at = MAX_CHUNK_CHARS
                chunks.append(remaining[:split_at].strip())
                remaining = remaining[split_at:].strip()
            if remaining:
                current = remaining
            continue

        if len(current) + len(para) + 2 <= MAX_CHUNK_CHARS:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)

    if (
        len(chunks) > 1
        and len(chunks[-1]) < MIN_CHUNK_CHARS
        and len(chunks[-2]) + len(chunks[-1]) + 2 <= MAX_CHUNK_CHARS
    ):
        chunks[-2] = chunks[-2] + "\n\n" + chunks[-1]
        chunks.pop()
    return chunks


def _table_context(section_text: str) -> list[dict]:
    """Use only nearby explicit labels; never infer a unit from values."""
    contexts = []
    previous_end = 0
    for match in TABLE_BLOCK_RE.finditer(section_text):
        nearby = [
            line.strip() for line in section_text[previous_end:match.start()].splitlines()
            if line.strip()
        ][-4:]
        units = None
        caption = None
        # Prefer a unit label above the table, then check its header rows.
        for line in list(reversed(nearby)) + match.group().splitlines()[:3]:
            unit_match = TABLE_UNITS_RE.search(line)
            if unit_match and units is None:
                units = unit_match.group().strip().rstrip(") .")
        for line in reversed(nearby):
            if TABLE_UNITS_RE.search(line):
                continue
            if len(line) <= 160 and not line.endswith((".", "?", "!")):
                caption = line
            break
        contexts.append({"caption": caption, "units": units})
        previous_end = match.end()
    return contexts


def chunk_filing(text: str, doc_id: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for item_number, title, section_text in split_into_sections(text):
        table_contexts = iter(_table_context(section_text))
        for idx, (chunk_type, chunk_text) in enumerate(split_section_preserving_tables(section_text)):
            metadata = {"doc_id": doc_id}
            if chunk_type == "table":
                metadata.update(next(table_contexts))
                metadata["section_title"] = title
                context_header = [f"Section: Item {item_number}. {title}"]
                if metadata["caption"]:
                    context_header.append(f"Caption: {metadata['caption']}")
                if metadata["units"]:
                    context_header.append(f"Units: {metadata['units']}")
                chunk_text = "\n".join(context_header) + "\n\n" + chunk_text
            chunks.append(Chunk(
                chunk_id=f"{doc_id}:item{item_number}:{idx}",
                text=chunk_text,
                section=f"Item {item_number}",
                section_title=title,
                chunk_type=chunk_type,
                metadata=metadata,
            ))
    return chunks


if __name__ == "__main__":
    sample_10k = """
Item 1. Business
Acme Corp designs and sells industrial sensors globally, serving manufacturing
and logistics customers across 40 countries.

Item 1A. Risk Factors
The Company faces risks related to supply chain concentration, foreign currency
exposure, and dependence on a small number of key suppliers for critical
components used in its sensor product lines.

Item 7. Management's Discussion and Analysis
Revenue grew year over year driven primarily by the industrial IoT segment.
Gross margin expanded due to favorable component pricing.

| Metric | FY2024 | FY2023 |
|---|---|---|
| Revenue | $412M | $365M |
| Gross Margin | 58.2% | 55.1% |
| Operating Income | $88M | $71M |

Operating expenses increased modestly, primarily driven by R&D investment in
next-generation sensor firmware.
"""
    chunks = chunk_filing(sample_10k, doc_id="ACME-10K-2024")
    for c in chunks:
        preview = c.text.replace("\n", " ")[:70]
        print(f"[{c.chunk_type:5}] {c.section:10} | {c.chunk_id:28} | {preview}...")
