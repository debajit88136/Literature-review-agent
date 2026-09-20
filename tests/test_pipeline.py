import json
import re
import unittest
from unittest import mock

from helpers import mk_paper
from lit_review import pipeline
from lit_review.arxiv_client import ArxivError
from lit_review.llm import LLMClient, LLMError
from lit_review.models import PaperContent
from lit_review.pipeline import PipelineError, run_pipeline


class FakeLLM(LLMClient):
    """Answers each pipeline stage with valid JSON, decided by the system prompt."""

    def __init__(self, extraction_reply=None):
        self.calls = []
        self.extraction_reply = extraction_reply

    def complete(self, system, prompt, max_tokens=1500):
        ids = re.findall(r"^\[(P\d+)\]", prompt, flags=re.MULTILINE)
        if "choosing which papers" in system:
            stage, data = "select", {"selected": list(reversed(ids))}
        elif "careful research assistant" in system:
            stage = "extract"
            if self.extraction_reply is not None:
                self.calls.append(stage)
                if isinstance(self.extraction_reply, Exception):
                    raise self.extraction_reply
                return self.extraction_reply
            data = {"paper_type": "new_method", "core_method": "A method.", "key_findings": ["f"],
                    "datasets": ["D"], "relation_to_topic": "Related.",
                    "limitations_and_open_problems": ["shared limit"], "relevance": 4}
        elif "organising papers" in system:
            stage = "cluster"
            half = max(1, len(ids) // 2)
            data = {"themes": [{"name": "First half", "description": "d", "papers": ids[:half]},
                               {"name": "Second half", "description": "d", "papers": ids[half:]}]}
        else:
            stage = "gaps"
            data = {"gaps": [{"title": "Shared limit", "description": "Everyone has it.", "papers": ids[:3]}]}
        self.calls.append(stage)
        return json.dumps(data)


def fake_content(paper):
    return PaperContent(paper, "some paper text", "full_text")


class PipelineTests(unittest.TestCase):
    def patch_search(self, papers):
        p = mock.patch.object(pipeline, "search_arxiv", return_value=papers)
        self.search = p.start()
        self.addCleanup(p.stop)

    def patch_content(self, fn=fake_content):
        p = mock.patch.object(pipeline, "get_paper_content", side_effect=fn)
        p.start()
        self.addCleanup(p.stop)

    def test_end_to_end(self):
        self.patch_search([mk_paper(i) for i in range(1, 13)])
        self.patch_content()
        llm, seen = FakeLLM(), []
        result = run_pipeline("my topic", llm, num_papers=5, model_name="fake",
                              progress=lambda f, m: seen.append((f, m)))

        self.assertEqual(self.search.call_args.kwargs["max_results"], 10)
        self.assertEqual(llm.calls, ["select"] + ["extract"] * 5 + ["cluster", "gaps"])
        self.assertEqual(len(result.notes), 5)
        self.assertEqual((result.num_candidates, result.num_full_text, result.num_failed), (12, 5, 0))
        self.assertEqual(sum(len(t.notes) for t in result.themes), 5)
        self.assertEqual(len(result.gaps), 1)
        self.assertTrue(result.markdown.startswith("# Literature Review: my topic"))
        for n in result.notes:
            self.assertIn(n.paper.citation(), result.markdown)
        fractions = [f for f, _ in seen]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(fractions[-1], 1.0)
        self.assertTrue(all(0.0 <= f <= 1.0 for f in fractions))

    def test_input_validation(self):
        self.patch_search([mk_paper(1)])
        for topic, n in [("", 5), ("   ", 5), ("topic", 2), ("topic", 21)]:
            with self.subTest(topic=topic, n=n), self.assertRaises(ValueError):
                run_pipeline(topic, FakeLLM(), num_papers=n)
        self.search.assert_not_called()

    def test_search_problems_become_pipeline_errors(self):
        for exc in (ArxivError("down"), ValueError("no searchable words")):
            with self.subTest(exc=exc):
                with mock.patch.object(pipeline, "search_arxiv", side_effect=exc):
                    with self.assertRaises(PipelineError):
                        run_pipeline("topic", FakeLLM(), num_papers=5)
        self.patch_search([])
        with self.assertRaisesRegex(PipelineError, "No papers found"):
            run_pipeline("topic", FakeLLM(), num_papers=5)

    def test_stops_early_when_llm_keeps_failing(self):
        self.patch_search([mk_paper(i) for i in range(1, 13)])
        self.patch_content()
        llm = FakeLLM(extraction_reply=LLMError("quota exhausted"))
        with self.assertRaisesRegex(PipelineError, "quota exhausted"):
            run_pipeline("topic", llm, num_papers=8)
        self.assertEqual(llm.calls.count("extract"), 3)

    def test_isolated_failures_do_not_stop_the_run(self):
        self.patch_search([mk_paper(i) for i in range(1, 13)])
        self.patch_content()
        llm = FakeLLM()
        original = llm.complete
        state = {"n": 0}

        def flaky(system, prompt, max_tokens=1500):
            if "careful research assistant" in system:
                state["n"] += 1
                if state["n"] == 2:
                    raise LLMError("one-off failure")
            return original(system, prompt, max_tokens)

        llm.complete = flaky
        result = run_pipeline("topic", llm, num_papers=5)
        self.assertEqual((len(result.notes), result.num_failed), (5, 1))
        self.assertIn("Papers that could not be processed", result.markdown)

    def test_abstract_only_papers_are_counted(self):
        self.patch_search([mk_paper(i) for i in range(1, 13)])
        self.patch_content(lambda p: PaperContent(p, "abstract", "abstract_only", "HTTP 404"))
        result = run_pipeline("topic", FakeLLM(), num_papers=4)
        self.assertEqual(result.num_full_text, 0)
        self.assertIn("4 from the abstract only", result.markdown)

    def test_progress_callback_is_optional(self):
        self.patch_search([mk_paper(i) for i in range(1, 8)])
        self.patch_content()
        self.assertTrue(run_pipeline("topic", FakeLLM(), num_papers=3).markdown)


if __name__ == "__main__":
    unittest.main()
