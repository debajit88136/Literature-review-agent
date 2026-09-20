"""The whole agent, end to end: topic in, Markdown literature review out.

    search arXiv -> pick best papers -> read each paper -> extract notes
                 -> group into themes -> find recurring gaps -> write Markdown

The UI (or a notebook) only needs to call `run_pipeline`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from .arxiv_client import ArxivError, search_arxiv
from .extraction import extract_notes
from .llm import LLMClient
from .models import Gap, PaperNotes, Theme
from .pdf_reader import get_paper_content
from .report import build_markdown_review
from .selection import MAX_POOL, select_top_papers
from .synthesis import cluster_papers, identify_gaps

logger = logging.getLogger(__name__)

MIN_PAPERS = 3
MAX_PAPERS = 20
DEFAULT_PAPERS = 15

ProgressCallback = Callable[[float, str], None]


class PipelineError(Exception):
    """A problem the user should be told about (bad topic, arXiv down, LLM key/quota, ...)."""


@dataclass
class ReviewResult:
    topic: str
    markdown: str
    notes: list[PaperNotes]
    themes: list[Theme]
    gaps: list[Gap]
    num_candidates: int
    num_full_text: int
    num_failed: int


def run_pipeline(
    topic: str,
    llm: LLMClient,
    num_papers: int = DEFAULT_PAPERS,
    model_name: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
    max_consecutive_failures: int = 3,
) -> ReviewResult:
    topic = topic.strip()
    if not topic:
        raise ValueError("Please enter a research topic.")
    if not MIN_PAPERS <= num_papers <= MAX_PAPERS:
        raise ValueError(f"Number of papers must be between {MIN_PAPERS} and {MAX_PAPERS}.")

    def report(fraction: float, message: str) -> None:
        logger.info("[%3d%%] %s", round(fraction * 100), message)
        if progress:
            progress(min(max(fraction, 0.0), 1.0), message)

    report(0.02, "Searching arXiv...")
    try:
        candidates = search_arxiv(topic, max_results=min(MAX_POOL, num_papers * 2))
    except ValueError as e:
        raise PipelineError(str(e)) from e
    except ArxivError as e:
        raise PipelineError(f"Could not search arXiv: {e}") from e
    if not candidates:
        raise PipelineError("No papers found for this topic. Try different or broader keywords.")

    report(0.08, f"Found {len(candidates)} candidates. Choosing the best {num_papers}...")
    selected = select_top_papers(candidates, topic, llm, keep=num_papers)

    notes: list[PaperNotes] = []
    in_a_row = 0
    for i, paper in enumerate(selected, 1):
        report(0.12 + 0.68 * (i - 1) / len(selected), f"Reading paper {i}/{len(selected)}: {paper.title[:70]}")
        note = extract_notes(get_paper_content(paper), topic, llm)
        notes.append(note)
        if note.ok:
            in_a_row = 0
            continue
        in_a_row += 1
        if in_a_row >= max_consecutive_failures:
            raise PipelineError(
                f"Stopped after {in_a_row} papers in a row failed. Last error: {note.error}. "
                "Check your API key and quota."
            )

    ok_notes = [n for n in notes if n.ok]
    if not ok_notes:
        raise PipelineError("None of the papers could be analysed.")

    report(0.82, "Grouping papers into themes...")
    themes = cluster_papers(notes, topic, llm)

    report(0.90, "Looking for recurring open problems...")
    gaps = identify_gaps(notes, topic, llm)

    report(0.97, "Writing the review...")
    markdown = build_markdown_review(topic, notes, themes, gaps, model_name=model_name)
    report(1.0, "Done.")

    return ReviewResult(
        topic=topic,
        markdown=markdown,
        notes=notes,
        themes=themes,
        gaps=gaps,
        num_candidates=len(candidates),
        num_full_text=sum(1 for n in ok_notes if n.source == "full_text"),
        num_failed=len(notes) - len(ok_notes),
    )
