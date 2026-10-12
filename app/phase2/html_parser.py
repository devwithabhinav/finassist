"""
app/phase2/html_parser.py

Converts raw EDGAR filing HTML into plain text, WITHOUT losing table
structure — <table> tags are converted to markdown-pipe format first,
then everything else is flattened to text.

Known limitation: SEC HTML tables often use colspan/rowspan for visual
formatting, which this naive converter doesn't reconcile — expect
occasional ragged rows on real filings.
"""

import re

from bs4 import BeautifulSoup


def _table_to_markdown(table_tag) -> str:
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    header = rows[0]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def html_to_text_preserving_tables(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Keep headings with inline formatting on one line so the chunker can
    # associate the complete statement name with the following table.
    for heading in soup.find_all(re.compile(r"^h[1-6]$")):
        heading.replace_with("\n" + heading.get_text(" ", strip=True) + "\n")

    for table in soup.find_all("table"):
        markdown = _table_to_markdown(table)
        caption_tag = table.find("caption")
        if markdown and caption_tag:
            markdown = caption_tag.get_text(" ", strip=True) + "\n" + markdown
        table.replace_with("\n" + markdown + "\n" if markdown else "")

    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


if __name__ == "__main__":
    sample_html = """
    <html><body>
    <p>Item 7. Management's Discussion and Analysis</p>
    <p>Revenue grew year over year.</p>
    <table>
      <tr><th>Metric</th><th>FY2024</th><th>FY2023</th></tr>
      <tr><td>Revenue</td><td>$412M</td><td>$365M</td></tr>
      <tr><td>Gross Margin</td><td>58.2%</td><td>55.1%</td></tr>
    </table>
    <p>Operating expenses increased modestly.</p>
    </body></html>
    """
    print(html_to_text_preserving_tables(sample_html))
