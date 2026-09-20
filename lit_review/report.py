"""Assemble the final Markdown review in plain Python (no LLM involved).

Titles, authors, years and links come straight from arXiv metadata, so citations
are always real. Only the summaries are model-written.
"""

from __future__ import annotations

import datetime


def one_line(text):
    return " ".join(str(text).split()).lstrip("#").strip()

def short_cite(paper):
    if not paper.authors:
        who = "Unknown authors"
    elif len(paper.authors) == 1:
        who = paper.authors[0]
    elif len(paper.authors) == 2:
        who = f"{paper.authors[0]} and {paper.authors[1]}"
    else:
        who = f"{paper.authors[0]} et al."
    return f"[{who} ({paper.year})]({paper.abs_url})"

def bullet_list(items):
    return "\n".join(f"  - {one_line(item)}" for item in items)

def render_paper(n):
    p = n.paper
    lines = [f"### {one_line(p.title)}", "", p.citation(), ""]
    lines.append(f"- **Type:** {n.paper_type.replace('_', ' ')}")
    lines.append(f"- **Approach:** {one_line(n.core_method)}")
    if n.key_findings:
        lines.append("- **Key findings:**")
        lines.append(bullet_list(n.key_findings))
    if n.datasets:
        lines.append(f"- **Datasets:** {', '.join(one_line(d) for d in n.datasets)}")
    if n.relation_to_topic:
        lines.append(f"- **Relation to the topic:** {one_line(n.relation_to_topic)}")
    if n.open_problems:
        lines.append("- **Stated limitations:**")
        lines.append(bullet_list(n.open_problems))
    if n.source == "abstract_only":
        lines.append("- _Notes based on the abstract only (full text was unavailable)._")
    lines.append("")
    return "\n".join(lines)

def sort_key_first_author(p):
    last = p.authors[0].split()[-1].lower() if p.authors else ""
    return (last, p.year)

def build_markdown_review(topic, notes, themes, gaps, model_name=None):
    ok_notes = [n for n in notes if n.ok]
    failed = [n for n in notes if not n.ok]
    full_text = sum(1 for n in ok_notes if n.source == "full_text")

    header = f"_Generated on {datetime.date.today().isoformat()} from {len(notes)} arXiv papers"
    if model_name:
        header += f" using {model_name}"
    lines = [f"# Literature Review: {one_line(topic)}", "", header + "._", "", "## Overview", ""]
    lines.append(
        f"This review analyses {len(ok_notes)} papers ({full_text} from full text, long papers trimmed to their start and end; "
        f"{len(ok_notes) - full_text} from the abstract only), grouped into {len(themes)} themes:"
    )
    lines.append("")
    for t in themes:
        count = len(t.notes)
        lines.append(f"- **{one_line(t.name)}** ({count} paper{'s' if count != 1 else ''}): {one_line(t.description)}")
    lines.append("")

    for t in themes:
        lines += [f"## {one_line(t.name)}", "", one_line(t.description), ""]
        for n in sorted(t.notes, key=lambda x: (-x.relevance, -x.paper.year)):
            lines.append(render_paper(n))

    lines += ["## Recurring open problems", ""]
    if gaps:
        lines += ["Problems raised by at least two different papers:", ""]
        for g in gaps:
            lines += [f"### {one_line(g.title)}", "", one_line(g.description), ""]
            lines += ["Raised by: " + "; ".join(short_cite(n.paper) for n in g.notes), ""]
    else:
        lines += ["No problem shared by two or more papers was identified in this set.", ""]

    if failed:
        lines += ["## Papers that could not be processed", "",
                  "These papers were found, but automatic extraction failed, so they are not analysed above:", ""]
        lines += [f"- {n.paper.citation()}" for n in failed]
        lines.append("")

    lines += ["## References", ""]
    lines += [f"- {p.citation()}" for p in sorted((n.paper for n in notes), key=sort_key_first_author)]
    lines += ["", "---",
              "_This review was generated automatically by an LLM pipeline. Titles, authors, years and links come "
              "directly from arXiv, but the summaries are model-written and may contain errors. Verify important "
              "claims against the original papers._", ""]
    return "\n".join(lines)
