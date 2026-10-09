"""
app/phase2/edgar_fetcher.py

Fetches 10-K filings from SEC EDGAR's free public API.

SEC REQUIRES a custom User-Agent header identifying a real person/org
and contact email, or it will reject requests (HTTP 403). This is not
optional and is enforced server-side — see:
https://www.sec.gov/os/webmaster-faq#developers

Rate limit: max ~10 requests/second.
"""

import time
import requests
# http://localhost:8000/docs
# CHANGE THIS to your real name/email — SEC will block generic or
# missing User-Agents.
SEC_USER_AGENT = "Abhinav Srivastava abhi89jandeveloper@gmail.com"

HEADERS = {"User-Agent": SEC_USER_AGENT}

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


def get_cik_for_ticker(ticker: str) -> int:
    resp = requests.get(TICKER_MAP_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    ticker = ticker.upper()
    for entry in data.values():
        if entry["ticker"] == ticker:
            return entry["cik_str"]
    raise ValueError(f"No CIK found for ticker '{ticker}'")


def get_latest_10k_url(cik: int) -> dict:
    resp = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=HEADERS, timeout=15)
    resp.raise_for_status()
    filings = resp.json()["filings"]["recent"]

    for i, form in enumerate(filings["form"]):
        if form == "10-K":
            accession = filings["accessionNumber"][i].replace("-", "")
            primary_doc = filings["primaryDocument"][i]
            filing_date = filings["filingDate"][i]
            doc_url = f"{ARCHIVES_BASE}/{cik}/{accession}/{primary_doc}"
            return {
                "filing_date": filing_date,
                "accession_number": filings["accessionNumber"][i],
                "document_url": doc_url,
            }
    raise ValueError(f"No 10-K filing found for CIK {cik}")


def fetch_filing_html(document_url: str) -> str:
    time.sleep(0.15)  # stay comfortably under 10 req/sec
    resp = requests.get(document_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_latest_10k(ticker: str) -> dict:
    cik = get_cik_for_ticker(ticker)
    filing_meta = get_latest_10k_url(cik)
    html = fetch_filing_html(filing_meta["document_url"])
    return {**filing_meta, "cik": cik, "html": html}


if __name__ == "__main__":
    result = fetch_latest_10k("AAPL")
    print(f"Filing date: {result['filing_date']}")
    print(f"Document URL: {result['document_url']}")
    print(f"HTML length: {len(result['html'])} chars")