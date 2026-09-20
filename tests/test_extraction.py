import json
import unittest

from helpers import Canned, mk_paper
from lit_review.extraction import as_str_list, build_notes, build_prompt, extract_notes
from lit_review.llm import LLMError
from lit_review.models import PaperContent


def content(source="full_text"):
    return PaperContent(mk_paper(1), "paper text here", source)


GOOD = json.dumps({
    "paper_type": "survey", "core_method": "We review things.",
    "key_findings": ["f1", "f2"], "datasets": ["D1"], "relation_to_topic": "Related.",
    "limitations_and_open_problems": ["l1", "not stated"], "relevance": 4,
})


class BuildNotesTests(unittest.TestCase):
    def test_placeholders_are_filtered(self):
        self.assertEqual(as_str_list(["not stated", "Real.", "N/A.", " ", "None"], 5), ["Real."])
        self.assertEqual(as_str_list("not a list", 3), [])
        self.assertEqual(as_str_list(["a", "b", "c"], 2), ["a", "b"])

    def test_valid_reply(self):
        n = build_notes(content(), json.loads(GOOD))
        self.assertTrue(n.ok)
        self.assertEqual((n.paper_type, n.relevance, n.open_problems), ("survey", 4, ["l1"]))

    def test_sanitises_odd_values(self):
        n = build_notes(content(), {"core_method": "m", "paper_type": "banana", "relevance": "9"})
        self.assertEqual((n.paper_type, n.relevance), ("other", 5))
        n = build_notes(content(), {"core_method": "m", "relevance": "high"})
        self.assertEqual(n.relevance, 0)

    def test_missing_core_method_is_rejected(self):
        with self.assertRaises(ValueError):
            build_notes(content(), {"paper_type": "survey"})

    def test_prompt_mentions_coverage_and_topic(self):
        self.assertIn("full text", build_prompt(content(), "topic-x"))
        self.assertIn("topic-x", build_prompt(content(), "topic-x"))
        self.assertIn("title and abstract only", build_prompt(content("abstract_only"), "t"))


class ExtractNotesTests(unittest.TestCase):
    def test_success(self):
        self.assertTrue(extract_notes(content(), "t", Canned(GOOD)).ok)

    def test_retry_then_success(self):
        llm = Canned("nonsense", GOOD)
        with self.assertLogs("lit_review.extraction", level="WARNING"):
            note = extract_notes(content(), "t", llm)
        self.assertTrue(note.ok)
        self.assertEqual(llm.calls, 2)

    def test_bad_output_twice_gives_error_note(self):
        with self.assertLogs("lit_review.extraction", level="WARNING"):
            note = extract_notes(content(), "t", Canned("nonsense"))
        self.assertFalse(note.ok)
        self.assertIn("bad model output", note.error)

    def test_api_error_is_not_retried(self):
        llm = Canned(LLMError("quota"))
        note = extract_notes(content(), "t", llm)
        self.assertEqual((note.ok, note.error, llm.calls), (False, "quota", 1))

    def test_unexpected_error_never_raises(self):
        with self.assertLogs("lit_review.extraction", level="ERROR"):
            note = extract_notes(content(), "t", Canned(RuntimeError("bug")))
        self.assertIn("unexpected", note.error)


if __name__ == "__main__":
    unittest.main()
