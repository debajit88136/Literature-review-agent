"""Pick the best K papers out of the arXiv candidates using one forced-choice LLM call."""

from __future__ import annotations

import logging

from .llm import LLMClient, LLMError, ask_json
from .models import Paper

logger = logging.getLogger(__name__)

MAX_POOL = 40

SELECT_SYSTEM = """You are choosing which papers to include in a literature review. Judge only from each paper's title and abstract.

Choose the papers that best serve a general review of the topic:
- Prefer papers whose main contribution is about the topic itself: core methods, evaluations, analyses and surveys.
- Cover different sub-areas of the topic. Do not pick several near-duplicate papers.
- Skip papers that only apply the topic to one narrow domain, unless nothing better is available.
- Skip papers that are unrelated or only mention the topic in passing.

Respond with a single JSON object and nothing else."""


def build_select_prompt(topic: str, pool: list[tuple[str, Paper]], keep: int) -> str:
    body = "\n\n".join(f"[{alias}] {p.title}\n{p.abstract[:900]}" for alias, p in pool)
    ids = ", ".join(alias for alias, _ in pool)
    return f"""Literature review topic: {topic}

<papers>
{body}
</papers>

Choose the {keep} papers most worth including, best first. Return a JSON object of the form {{"selected": ["P4", "P1", "P9"]}}.
Use only these ids: {ids}
Return fewer than {keep} ids only if fewer papers are genuinely relevant."""


def parse_selection(data: dict, allowed: dict, keep: int) -> list[str]:
    raw = data.get("selected")
    if not isinstance(raw, list):
        raise ValueError("missing 'selected' list")
    chosen = []
    for alias in raw:
        alias = str(alias).strip()
        if alias in allowed and alias not in chosen:
            chosen.append(alias)
    if not chosen:
        raise ValueError("no valid ids selected")
    return chosen[:keep]


def select_top_papers(papers: list[Paper], topic: str, llm: LLMClient, keep: int) -> list[Paper]:
    pool = [(f"P{i}", p) for i, p in enumerate(papers[:MAX_POOL], 1)]
    allowed = dict(pool)
    prompt = build_select_prompt(topic, pool, keep)
    try:
        chosen = ask_json(llm, SELECT_SYSTEM, prompt,
                          lambda d: parse_selection(d, allowed, keep), label="paper selection")
    except (LLMError, ValueError) as e:
        logger.warning("Paper selection failed (%s); keeping arXiv order", e)
        return papers[:keep]
    if len(chosen) < keep:
        logger.info("Model selected only %d of the %d requested papers", len(chosen), keep)
    return [allowed[a] for a in chosen]
