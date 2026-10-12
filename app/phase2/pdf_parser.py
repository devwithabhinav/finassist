"""Extract PDF text and ruled tables in reading order for financial_chunker.

Image-only PDFs require OCR, which this parser does not perform.
"""

from io import BytesIO
import re

import pdfplumber
from pdfminer.pdfexceptions import PDFException
from pdfplumber.utils.exceptions import PdfminerException


def pdf_to_text_preserving_tables(data: bytes) -> str:
    if not data:
        raise ValueError("The uploaded PDF is empty.")
    pieces = []
    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            for page in pdf.pages:
                tables = sorted(page.find_tables(), key=lambda table: table.bbox[1])
                # Remove table characters from prose to avoid indexing rows twice.
                def outside_tables(obj):
                    x = (obj.get("x0", 0) + obj.get("x1", 0)) / 2
                    y = (obj.get("top", 0) + obj.get("bottom", 0)) / 2
                    return not any(
                        left <= x <= right and top <= y <= bottom
                        for left, top, right, bottom in (t.bbox for t in tables)
                    )

                prose = page.filter(outside_tables)
                start = page.bbox[1]
                for table in tables:
                    top = table.bbox[1]
                    text = prose.filter(
                        lambda obj: start <= obj.get("top", -1) < top
                    ).extract_text()
                    if text:
                        pieces.append(text)
                    rows = table.extract()
                    if rows:
                        width = max(len(row) for row in rows)
                        lines = []
                        for row in rows:
                            cells = [re.sub(r"\s+", " ", cell or "").strip().replace("|", "\\|") for cell in row]
                            cells.extend([""] * (width - len(cells)))
                            lines.append("| " + " | ".join(cells) + " |")
                        lines.insert(1, "|" + "---|" * width)
                        pieces.append("\n".join(lines))
                    start = top
                text = prose.filter(lambda obj: obj.get("top", -1) >= start).extract_text()
                if text:
                    pieces.append(text)
    except (PdfminerException, PDFException, ValueError, TypeError, OSError, EOFError) as exc:
        raise ValueError("The upload is not a readable, unencrypted PDF.") from exc
    text = "\n\n".join(pieces).strip()
    if not text:
        raise ValueError("The PDF contains no extractable text or tables; scanned PDFs require OCR.")
    return text
