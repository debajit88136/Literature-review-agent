from lit_review.llm import LLMClient
from lit_review.models import Paper, PaperNotes


def mk_paper(i: int, abstract: str | None = None) -> Paper:
    return Paper(
        arxiv_id=f"2401.{i:05d}v1",
        title=f"Paper {i}",
        authors=[f"First{i} Last{i}", "Second Author"],
        abstract=abstract or f"Abstract {i}",
        year=2020 + i % 5,
        abs_url=f"https://arxiv.org/abs/2401.{i:05d}v1",
        pdf_url=f"https://arxiv.org/pdf/2401.{i:05d}v1",
    )


def mk_note(i: int, ptype: str = "new_method", problems=None, ok: bool = True,
            source: str = "full_text") -> PaperNotes:
    note = PaperNotes(mk_paper(i), source, ptype, f"method {i}", [f"finding {i}"], [f"DS{i}"],
                      f"relation {i}", problems or [], 4)
    if not ok:
        note.error = "failed"
    return note


class Canned(LLMClient):
    """Returns scripted replies in order (the last one repeats). Exceptions are raised."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.last_prompt = ""

    def complete(self, system, prompt, max_tokens=1500):
        self.calls += 1
        self.last_prompt = prompt
        reply = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply
