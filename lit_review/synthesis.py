"""Cross-paper analysis: group papers into themes and find recurring open problems.

Both steps ask the LLM for JSON that refers to papers by short aliases (P1, P2, ...),
then validate everything in code, so the model cannot invent, drop or duplicate papers.
"""

from __future__ import annotations

import logging

from .llm import LLMClient, LLMError, ask_json
from .models import Gap, PaperNotes, Theme

logger = logging.getLogger(__name__)

TYPE_LABELS = {
    "new_method": "New methods",
    "survey": "Surveys and overviews",
    "benchmark_or_dataset": "Benchmarks and datasets",
    "empirical_study": "Empirical studies",
    "theory": "Theoretical work",
    "system_or_tool": "Systems and tools",
    "other": "Other work",
}

CLUSTER_SYSTEM = """You are organising papers into themes for a literature review.

Rules:
- Use ONLY the notes provided.
- Group papers by shared approach or shared sub-topic, whichever gives the most meaningful groups.
- Every paper must appear in exactly one theme.
- Theme names are short (2-6 words) and specific. Avoid vague names like "Miscellaneous".
- Each description is 1-2 sentences saying what the papers in the theme have in common.
- Respond with a single JSON object and nothing else."""

GAP_SYSTEM = """You are identifying recurring open problems across papers for a literature review.

Rules:
- Use ONLY the limitations and open problems listed below. Do not add problems from outside knowledge.
- A recurring gap must be raised by at least 2 different papers. Do not include problems raised by only one paper.
- Merge problems that are essentially the same issue, even if worded differently.
- Respond with a single JSON object and nothing else."""


def alias_map(notes: list[PaperNotes]) -> dict[str, PaperNotes]:
    ok_notes = [n for n in notes if n.ok]
    return {f"P{i}": n for i, n in enumerate(ok_notes, 1)}


def theme_limits(n: int) -> tuple[int, int]:
    if n < 4:
        return 1, min(2, n)
    max_themes = max(2, min(6, round(n / 3)))
    return max(2, max_themes - 2), max_themes


def notes_line(alias: str, n: PaperNotes) -> str:
    return (f"[{alias}] {n.paper.title} ({n.paper.year}) | type: {n.paper_type} | "
            f"method: {n.core_method[:350]} | relation: {n.relation_to_topic[:250]}")


def build_cluster_prompt(topic: str, amap: dict, low: int, high: int) -> str:
    lines = "\n".join(notes_line(a, n) for a, n in amap.items())
    ids = ", ".join(amap)
    return f"""Literature review topic: {topic}

Paper notes:
{lines}

Create between {low} and {high} themes. Return a JSON object of the form:
{{"themes": [{{"name": "...", "description": "...", "papers": ["P1", "P3"]}}]}}
Every one of these ids must appear in exactly one theme: {ids}"""


def parse_themes(data: dict, amap: dict) -> list[Theme]:
    raw_themes = data.get("themes")
    if not isinstance(raw_themes, list):
        raise ValueError("missing 'themes' list")
    assigned: set[str] = set()
    themes = []
    for item in raw_themes:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        aliases = item.get("papers")
        if not isinstance(aliases, list):
            continue
        candidates = []
        for alias in aliases:
            alias = str(alias).strip()
            if alias in amap and alias not in assigned and alias not in candidates:
                candidates.append(alias)
        if name and candidates:
            assigned.update(candidates)
            themes.append(Theme(name, str(item.get("description", "")).strip(),
                                [amap[a] for a in candidates]))
    if not themes:
        raise ValueError("no valid themes")
    leftover = [n for a, n in amap.items() if a not in assigned]
    if leftover:
        themes.append(Theme("Other approaches", "Papers that did not fit the themes above.", leftover))
    return themes


def group_by_type(notes: list[PaperNotes]) -> list[Theme]:
    groups: dict[str, list[PaperNotes]] = {}
    for n in notes:
        groups.setdefault(n.paper_type, []).append(n)
    return [Theme(TYPE_LABELS.get(t, "Other work"),
                  "Papers grouped by type of contribution (automatic fallback grouping).", members)
            for t, members in groups.items()]


def cluster_papers(notes: list[PaperNotes], topic: str, llm: LLMClient) -> list[Theme]:
    amap = alias_map(notes)
    ok_notes = list(amap.values())
    if len(ok_notes) < 2:
        return group_by_type(ok_notes)
    low, high = theme_limits(len(ok_notes))
    prompt = build_cluster_prompt(topic, amap, low, high)
    try:
        return ask_json(llm, CLUSTER_SYSTEM, prompt,
                        lambda d: parse_themes(d, amap), label="clustering")
    except (LLMError, ValueError) as e:
        logger.warning("Clustering failed (%s); grouping by paper type instead", e)
        return group_by_type(ok_notes)


def build_gap_prompt(topic: str, entries: dict) -> str:
    lines = "\n".join(f"[{a}] {n.paper.title} ({n.paper.year}): " + "; ".join(n.open_problems)
                      for a, n in entries.items())
    return f"""Literature review topic: {topic}

Limitations and open problems reported by each paper:
{lines}

Return a JSON object of the form:
{{"gaps": [{{"title": "...", "description": "1-2 sentences", "papers": ["P1", "P4"]}}]}}
Return at most 6 gaps, and an empty list if no problem is shared by two or more papers."""


def parse_gaps(data: dict, allowed: dict) -> list[Gap]:
    raw = data.get("gaps")
    if not isinstance(raw, list):
        raise ValueError("missing 'gaps' list")
    gaps = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        aliases = item.get("papers")
        if not title or not isinstance(aliases, list):
            continue
        seen: list[str] = []
        for alias in aliases:
            alias = str(alias).strip()
            if alias in allowed and alias not in seen:
                seen.append(alias)
        if len(seen) >= 2:
            gaps.append(Gap(title, str(item.get("description", "")).strip(),
                            [allowed[a] for a in seen]))
    gaps.sort(key=lambda g: -len(g.notes))
    return gaps[:6]


def identify_gaps(notes: list[PaperNotes], topic: str, llm: LLMClient) -> list[Gap]:
    entries = {a: n for a, n in alias_map(notes).items() if n.open_problems}
    if len(entries) < 2:
        logger.info("Fewer than 2 papers report open problems; skipping gap analysis")
        return []
    prompt = build_gap_prompt(topic, entries)
    try:
        return ask_json(llm, GAP_SYSTEM, prompt,
                        lambda d: parse_gaps(d, entries), label="gap identification")
    except (LLMError, ValueError) as e:
        logger.warning("Gap identification failed: %s", e)
        return []
