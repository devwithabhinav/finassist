from app.phase2.financial_chunker import (
    MAX_CHUNK_CHARS,
    chunk_filing,
    split_into_sections,
    split_section_preserving_tables,
)
import pytest


def test_toc_does_not_produce_junk_sections():
    document = """TABLE OF CONTENTS
Item 1. Business ................................ 3
Item 1A. Risk Factors
12
Item 7. Management's Discussion and Analysis
25
PART I
Item 1. Business
We design and sell industrial sensors.
Item 1A. Risk Factors
Supply chain concentration poses a risk.
Item 7. Management's Discussion and Analysis
Revenue increased during the year.
"""
    expected = [
        ("1", "Business", "We design and sell industrial sensors."),
        ("1A", "Risk Factors", "Supply chain concentration poses a risk."),
        ("7", "Management's Discussion and Analysis", "Revenue increased during the year."),
    ]

    assert split_into_sections(document) == expected
    chunks = chunk_filing(document, "example")
    assert [chunk.text for chunk in chunks] == [body for _, _, body in expected]
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_item_7_table_remains_one_chunk():
    table = "\n".join(
        ["| Metric | Value |", "|---|---|"]
        + [f"| Metric {index} | {index * 100} |" for index in range(150)]
    )
    assert len(table) > MAX_CHUNK_CHARS
    document = (
        "Item 7. Management's Discussion and Analysis\n"
        "Revenue increased during the year.\n\n"
        f"{table}\n\n"
        "Operating expenses increased modestly.\n"
        "Item 8. Financial Statements\n"
        "See the accompanying financial statements."
    )

    chunks = chunk_filing(document, "example")
    tables = [chunk for chunk in chunks if chunk.chunk_type == "table"]
    assert len(tables) == 1
    assert tables[0].text.split("\n\n", 1)[1] == table
    assert tables[0].text.startswith("Section: Item 7. Management's Discussion and Analysis")
    assert tables[0].section == "Item 7"
    assert [chunk.chunk_type for chunk in chunks if chunk.section == "Item 7"] == [
        "prose", "table", "prose"
    ]


def test_table_metadata_has_caption_units_and_section_title():
    document = """Item 7. Management's Discussion and Analysis
Revenue increased during the year.
Consolidated Results of Operations
(In millions, except per-share amounts)
| Metric | 2025 | 2024 |
|---|---|---|
| Revenue | 100 | 90 |
"""
    table = next(c for c in chunk_filing(document, "example") if c.chunk_type == "table")
    assert table.metadata == {
        "doc_id": "example",
        "caption": "Consolidated Results of Operations",
        "units": "In millions, except per-share amounts",
        "section_title": "Management's Discussion and Analysis",
    }
    assert table.text.startswith(
        "Section: Item 7. Management's Discussion and Analysis\n"
        "Caption: Consolidated Results of Operations\n"
        "Units: In millions, except per-share amounts\n\n| Metric"
    )


def test_table_context_does_not_leak_to_next_table_or_section():
    document = """Item 7. Management's Discussion and Analysis
Revenue Summary
(In millions)
| Metric | Value |
|---|---|
| Revenue | 100 |

The following table lists employees.
| Country | Employees |
|---|---|
| US | 500 |
Item 8. Financial Statements
| Metric | Value |
|---|---|
| Cash | 10 |
"""
    tables = [c for c in chunk_filing(document, "example") if c.chunk_type == "table"]
    assert len(tables) == 3
    assert tables[0].metadata["caption"] == "Revenue Summary"
    for table in tables[1:]:
        assert table.metadata["caption"] is None
        assert table.metadata["units"] is None
    assert tables[-1].metadata["section_title"] == "Financial Statements"
    assert tables[-1].text.startswith("Section: Item 8. Financial Statements\n\n|")
    assert "Units:" not in tables[-1].text
    assert "Caption:" not in tables[-1].text


def test_html_caption_and_header_units_survive_pipeline():
    from app.phase2.html_parser import html_to_text_preserving_tables

    html = """<p>Item 8. Financial Statements</p>
    <table><caption>Consolidated Balance Sheets</caption>
    <tr><th>USD in thousands</th><th>2025</th></tr>
    <tr><td>Cash</td><td>100</td></tr></table>"""
    tables = [
        c for c in chunk_filing(html_to_text_preserving_tables(html), "example")
        if c.chunk_type == "table"
    ]
    assert len(tables) == 1
    assert tables[0].metadata["caption"] == "Consolidated Balance Sheets"
    assert tables[0].metadata["units"] == "in thousands"
    assert tables[0].metadata["section_title"] == "Financial Statements"
    assert "Caption: Consolidated Balance Sheets\nUnits: in thousands\n\n" in tables[0].text


def test_nearest_html_heading_and_units_are_in_table_text_and_metadata():
    from app.phase2.html_parser import html_to_text_preserving_tables

    html = """<h1>Item 8. Financial Statements</h1>
    <h2>Financial Overview</h2><p>(in millions)</p>
    <h3>Consolidated <span>Statements of</span> Operations</h3>
    <p>(in thousands)</p>
    <table><tr><th>Metric</th><th>2025</th><th>2024</th></tr>
    <tr><td>Revenue</td><td>412</td><td>365</td></tr></table>"""
    tables = [
        c for c in chunk_filing(html_to_text_preserving_tables(html), "example")
        if c.chunk_type == "table"
    ]
    assert len(tables) == 1
    table = tables[0]
    assert table.metadata["caption"] == "Consolidated Statements of Operations"
    assert table.metadata["units"] == "in thousands"
    assert table.metadata["section_title"] == "Financial Statements"
    assert table.text == (
        "Section: Item 8. Financial Statements\n"
        "Caption: Consolidated Statements of Operations\n"
        "Units: in thousands\n\n"
        "| Metric | 2025 | 2024 |\n|---|---|---|\n| Revenue | 412 | 365 |"
    )


def test_document_without_item_headers_uses_full_document_section():
    text = "A standalone company overview."
    assert split_into_sections(text) == [("Unknown", "Full document", text)]
    chunks = chunk_filing(text, "overview")
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].section_title == "Full document"
    assert chunks[0].metadata == {"doc_id": "overview"}


def test_empty_document_produces_no_chunks():
    assert chunk_filing("", "empty") == []


def test_item_headers_are_case_insensitive_and_short_real_sections_survive():
    assert split_into_sections("item 1a. Risk Factors\nShort risk.\nITEM 7. Discussion\nShort discussion.") == [
        ("1a", "Risk Factors", "Short risk."),
        ("7", "Discussion", "Short discussion."),
    ]


@pytest.mark.parametrize("text", ["alpha beta " * 500, "x" * (MAX_CHUNK_CHARS * 2 + 23)])
def test_oversized_prose_is_bounded_without_losing_content(text):
    pieces = split_section_preserving_tables(text)
    assert len(pieces) > 1
    assert all(kind == "prose" and 0 < len(body) <= MAX_CHUNK_CHARS for kind, body in pieces)
    assert "".join(body.replace(" ", "") for _, body in pieces) == text.replace(" ", "")


def test_paragraphs_pack_together_until_chunk_limit():
    paragraphs = ["a" * 900, "b" * 800, "c" * 150]
    pieces = split_section_preserving_tables("\n\n".join(paragraphs))
    assert pieces == [("prose", paragraphs[0] + "\n\n" + paragraphs[1]), ("prose", paragraphs[2])]
