"""Download the public-domain homebuying docs and extract plain text.

Downloads PDFs into rag/docs/, then extracts text into rag/docs/txt/.
Uses stdlib urllib for the download (no deps) and pypdf for extraction.
"""
import urllib.request
from pathlib import Path

# Public-domain IRS publications on homeownership taxes. Add more URLs here —
# the downloader skips anything already present and logs failures per file.
DOCS = {
    "irs_pub_530.pdf": "https://www.irs.gov/pub/irs-pdf/p530.pdf",  # tax info for homeowners
    "irs_pub_936.pdf": "https://www.irs.gov/pub/irs-pdf/p936.pdf",  # mortgage interest deduction
    "irs_pub_523.pdf": "https://www.irs.gov/pub/irs-pdf/p523.pdf",  # selling your home
}

DOCS_DIR = Path(__file__).resolve().parent / "docs"


def download():
    DOCS_DIR.mkdir(exist_ok=True, parents=True)
    for name, url in DOCS.items():
        dest = DOCS_DIR / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"skip   {name} (already present)")
            continue
        print(f"fetch  {name} <- {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                dest.write_bytes(r.read())
            print(f"  ok   {dest.stat().st_size} bytes")
        except Exception as e:
            print(f"  FAILED: {e}")
    return DOCS_DIR


def extract_text():
    from pypdf import PdfReader
    txt_dir = DOCS_DIR / "txt"
    txt_dir.mkdir(exist_ok=True)
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        out = txt_dir / (pdf.stem + ".txt")
        if out.exists() and out.stat().st_size > 1000:
            print(f"skip extract {pdf.name}")
            continue
        reader = PdfReader(str(pdf))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        out.write_text(text)
        print(f"extract {pdf.name} -> {out.name} ({len(text)} chars)")


if __name__ == "__main__":
    download()
    extract_text()
    print("done")
