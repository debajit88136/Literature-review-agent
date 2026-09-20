"""Get the best available text for a paper: full PDF text, else the abstract.

`get_paper_content` never raises. Any failure becomes an abstract-only fallback.
"""

from __future__ import annotations

import logging
import re

import requests

from .arxiv_client import USER_AGENT, wait_for_rate_limit
from .models import Paper, PaperContent

logger = logging.getLogger(__name__)

MAX_PDF_BYTES = 25 * 1024 * 1024
MIN_USEFUL_CHARS = 1_500
HEAD_CHARS = 20_000
TAIL_CHARS = 10_000
DOWNLOAD_TIMEOUT_SECONDS = 60


class PdfError(Exception):
    pass


def download_pdf(url: str) -> bytes:
    wait_for_rate_limit()
    try:
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS,
                                headers={"User-Agent": USER_AGENT})
    except requests.RequestException as e:
        raise PdfError(f"download failed: {e}") from e
    if response.status_code != 200:
        raise PdfError(f"HTTP {response.status_code}")
    if len(response.content) > MAX_PDF_BYTES:
        raise PdfError("PDF is too large")
    if not response.content.startswith(b"%PDF"):
        raise PdfError("file is not a PDF (possibly an HTML page)")
    return response.content


def extract_text(pdf_bytes: bytes) -> str:
    import pymupdf

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.needs_pass:
                raise PdfError("PDF is password protected")
            pages = [page.get_text() for page in doc]
    except PdfError:
        raise
    except Exception as e:
        raise PdfError(f"could not parse PDF: {e}") from e

    text = "\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_references(text: str) -> str:
    matches = list(re.finditer(r"\n\s*(references|bibliography)\s*\n", text, flags=re.IGNORECASE))
    if matches and matches[-1].start() > len(text) * 0.4:
        return text[:matches[-1].start()]
    return text


def shorten(text: str) -> str:
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text
    return text[:HEAD_CHARS] + "\n\n[... middle section omitted ...]\n\n" + text[-TAIL_CHARS:]


def get_paper_content(paper: Paper) -> PaperContent:
    fallback_text = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
    try:
        text = strip_references(extract_text(download_pdf(paper.pdf_url)))
        if len(text) < MIN_USEFUL_CHARS:
            raise PdfError(f"too little text ({len(text)} chars), possibly a scanned PDF")
        return PaperContent(paper, shorten(text), "full_text")
    except PdfError as e:
        logger.warning("%s: falling back to abstract (%s)", paper.arxiv_id, e)
        return PaperContent(paper, fallback_text, "abstract_only", note=str(e))
    except Exception as e:
        logger.exception("%s: unexpected error", paper.arxiv_id)
        return PaperContent(paper, fallback_text, "abstract_only", note=f"unexpected: {e}")
