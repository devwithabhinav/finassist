import pytest

from app.phase2.html_parser import html_to_text_preserving_tables


@pytest.mark.parametrize("html", ["", "<html><body></body></html>", "<table></table>"])
def test_empty_html_has_no_spurious_table(html):
    assert html_to_text_preserving_tables(html) == ""


def test_tables_preserve_values_caption_and_document_order():
    html = """<p>Before &amp; after</p>
    <table><caption>Income <b>statement</b></caption>
    <tr><th>Metric</th><th>FY2025</th></tr>
    <tr><td>Revenue</td><td>412</td></tr></table>
    <p>Between tables</p>
    <table><tr><th>Asset</th><th>Value</th></tr>
    <tr><td>Cash</td><td>365</td></tr></table><p>End</p>"""
    text = html_to_text_preserving_tables(html)
    first = "| Metric | FY2025 |\n|---|---|\n| Revenue | 412 |"
    second = "| Asset | Value |\n|---|---|\n| Cash | 365 |"
    assert first in text and second in text
    assert text.index("Before & after") < text.index("Income statement") < text.index(first)
    assert text.index(first) < text.index("Between tables") < text.index(second) < text.index("End")
    assert text.count("412") == 1
    assert "<table>" not in text


def test_heading_inline_elements_and_whitespace_are_normalized():
    text = html_to_text_preserving_tables(
        "<h2>Consolidated <span>Balance</span> Sheets</h2>"
        "<p>Cash\t  increased.</p><p>Notes</p>"
    )
    assert text.splitlines()[0] == "Consolidated Balance Sheets"
    assert "Cash increased." in text
    assert "\n\n\n" not in text
