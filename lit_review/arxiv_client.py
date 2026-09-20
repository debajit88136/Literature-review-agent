"""Search arXiv and turn the response into `Paper` objects.

The module is split into three layers on purpose:

1. build_query()   - pure function: topic string -> arXiv query string
2. _fetch_feed()   - the ONLY function that touches the network (rate limit,
                     timeout, retries)
3. parse_feed()    - pure function: XML text -> list[Paper]

Layers 1 and 3 are pure (no network, no clock), so they can be unit-tested
instantly and deterministically. Only layer 2 needs mocking.
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET

import requests

from .models import Paper

logger = logging.getLogger(__name__)

# --- Configuration ---------------------------------------------------------

# https, not http: same API, but the traffic is encrypted.
ARXIV_API_URL = "https://export.arxiv.org/api/query"

# arXiv's API terms of use ask for at most one request every 3 seconds.
MIN_SECONDS_BETWEEN_REQUESTS = 3.0

REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}  # "try again later" errors

# Identify ourselves politely; anonymous default user-agents get blocked more.
USER_AGENT = "autonomous-lit-review-agent/0.1 (learning project)"

VALID_SORT_ORDERS = {"relevance", "submittedDate", "lastUpdatedDate"}
MAX_RESULTS_LIMIT = 100  # per-request cap we allow ourselves

# Words that carry no topical signal. If we AND them into the query, we would
# demand that every paper contains e.g. "recent" and "advances" and get nothing.
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "with",
    "using", "via", "from", "by", "at", "is", "are", "recent", "advances",
    "survey", "review", "overview", "towards", "toward",
}

# XML namespaces used in arXiv's Atom feed. ElementTree needs these spelled out.
NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivError(Exception):
    """Anything that goes wrong while talking to arXiv (network, HTTP, bad XML).

    One dedicated exception type lets callers write `except ArxivError` without
    accidentally swallowing unrelated bugs (like a TypeError in our own code).
    """


# --- Layer 1: query building -------------------------------------------------


def build_query(topic: str) -> str:
    """Turn a free-text topic into an arXiv `search_query` string.

    "LLM hallucination detection" -> "all:LLM AND all:hallucination AND all:detection"

    `all:` searches title, abstract, authors, etc. We AND the words together so
    a result must be about *all* of them, but we do NOT require them to appear
    as an exact phrase, which would be far too strict.
    """
    # Keep only word characters and hyphens. This strips quotes, colons and
    # parentheses that have special meaning in arXiv's query syntax and could
    # otherwise break (or hijack) the query.
    words = re.findall(r"[\w\-]+", topic)
    terms = [w for w in words if w.lower() not in STOPWORDS]
    if not terms:
        raise ValueError(f"Topic {topic!r} contains no searchable words.")
    return " AND ".join(f"all:{term}" for term in terms)


# --- Layer 2: network --------------------------------------------------------

_last_request_time = float("-inf")  # monotonic timestamp of our last request


def wait_for_rate_limit() -> None:
    """Sleep just long enough to respect arXiv's 1-request-per-3-seconds rule."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
        time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
    _last_request_time = time.monotonic()


def _fetch_feed(params: dict) -> str:
    """GET the arXiv API and return the raw XML text.

    Retries transient failures (timeouts, connection drops, 429/5xx) with
    exponential backoff. Permanent failures (e.g. HTTP 400) fail immediately,
    since retrying a bad request can never help.
    """
    last_error = "unknown error"
    for attempt in range(1, MAX_RETRIES + 1):
        wait_for_rate_limit()
        try:
            response = requests.get(
                ARXIV_API_URL,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT_SECONDS,  # never wait forever
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = f"network error: {exc}"
        else:
            if response.status_code == 200:
                return response.text
            if response.status_code not in RETRYABLE_STATUS_CODES:
                raise ArxivError(f"arXiv returned HTTP {response.status_code}")
            last_error = f"HTTP {response.status_code}"

        if attempt < MAX_RETRIES:
            backoff = 2**attempt  # 2s, 4s, ...
            logger.warning(
                "arXiv request failed (%s); retry %d/%d in %ds",
                last_error, attempt, MAX_RETRIES - 1, backoff,
            )
            time.sleep(backoff)

    raise ArxivError(f"arXiv request failed after {MAX_RETRIES} attempts: {last_error}")


# --- Layer 3: parsing --------------------------------------------------------


def _text(element: ET.Element, path: str) -> str:
    """Text of a child element with whitespace collapsed ('' if missing).

    arXiv hard-wraps titles and abstracts with newlines and indentation;
    " ".join(text.split()) turns all runs of whitespace into single spaces.
    """
    child = element.find(path, NAMESPACES)
    if child is None or child.text is None:
        return ""
    return " ".join(child.text.split())


def _parse_entry(entry: ET.Element) -> Paper:
    """Convert one <entry> element into a Paper. Raises ValueError if unusable."""
    raw_id_url = _text(entry, "atom:id")  # e.g. http://arxiv.org/abs/2401.00001v2
    if "/abs/" not in raw_id_url:
        raise ValueError(f"unexpected id format: {raw_id_url!r}")
    # Split on "/abs/" rather than on the last "/": old-style ids contain a
    # slash themselves (hep-th/9901001v1).
    arxiv_id = raw_id_url.split("/abs/", 1)[1]
    abs_url = raw_id_url.replace("http://", "https://")

    title = _text(entry, "atom:title")
    if not title:
        raise ValueError(f"entry {arxiv_id} has no title")

    # "published" = date of the first version. ("updated" = latest revision.)
    published = _text(entry, "atom:published")  # "2024-01-01T09:30:00Z"
    year = int(published[:4])  # ValueError if malformed -> entry gets skipped

    authors = [
        name for a in entry.findall("atom:author", NAMESPACES)
        if (name := _text(a, "atom:name"))
    ]

    pdf_url = ""
    for link in entry.findall("atom:link", NAMESPACES):
        if link.get("title") == "pdf":
            pdf_url = (link.get("href") or "").replace("http://", "https://")
    if not pdf_url:
        # arXiv PDFs always live at /pdf/<id>, so we can build the URL ourselves.
        pdf_url = abs_url.replace("/abs/", "/pdf/")

    categories = [
        term for c in entry.findall("atom:category", NAMESPACES)
        if (term := c.get("term"))
    ]

    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        abstract=_text(entry, "atom:summary"),
        year=year,
        abs_url=abs_url,
        pdf_url=pdf_url,
        categories=categories,
    )


def parse_feed(xml_text: str) -> list[Paper]:
    """Parse an arXiv Atom feed into Papers, skipping individually broken entries."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ArxivError(f"Could not parse arXiv response as XML: {exc}") from exc

    papers: list[Paper] = []
    for entry in root.findall("atom:entry", NAMESPACES):
        # Quirk: when a query is malformed, arXiv still answers HTTP 200, with
        # a single fake <entry> whose id points at /api/errors.
        if "/api/errors" in _text(entry, "atom:id"):
            raise ArxivError(f"arXiv rejected the query: {_text(entry, 'atom:summary')}")
        try:
            papers.append(_parse_entry(entry))
        except ValueError as exc:
            # One bad entry must not lose the other 19 good ones.
            logger.warning("Skipping malformed arXiv entry: %s", exc)
    return papers


# --- Public API --------------------------------------------------------------


def search_arxiv(
    topic: str,
    max_results: int = 20,
    sort_by: str = "relevance",
) -> list[Paper]:
    """Search arXiv for `topic` and return up to `max_results` Papers.

    Raises:
        ValueError: bad arguments (empty topic, max_results out of range, ...)
        ArxivError: network / HTTP / response-format problems
    """
    if sort_by not in VALID_SORT_ORDERS:
        raise ValueError(f"sort_by must be one of {sorted(VALID_SORT_ORDERS)}")
    if not 1 <= max_results <= MAX_RESULTS_LIMIT:
        raise ValueError(f"max_results must be between 1 and {MAX_RESULTS_LIMIT}")

    params = {
        "search_query": build_query(topic),
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    logger.info("Searching arXiv: %s", params["search_query"])
    papers = parse_feed(_fetch_feed(params))
    logger.info("arXiv returned %d papers", len(papers))
    return papers
