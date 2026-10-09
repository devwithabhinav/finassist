from app.phase2.financial_chunker import (
    MAX_CHUNK_CHARS,
    chunk_filing,
    split_into_sections,
)


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
    assert tables[0].text == table
    assert tables[0].section == "Item 7"
    assert [chunk.chunk_type for chunk in chunks if chunk.section == "Item 7"] == [
        "prose", "table", "prose"
    ]
