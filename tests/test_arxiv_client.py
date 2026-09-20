"""Offline tests for lit_review.arxiv_client.

None of these tests touch the real internet: pure functions are tested directly
on fixture XML files, and the network layer is tested with a fake `requests.get`.
Run with either:  python -m unittest discover -s tests -v    or    pytest
"""

import unittest
from pathlib import Path
from unittest import mock

import requests

from lit_review import arxiv_client
from lit_review.arxiv_client import ArxivError, build_query, parse_feed, search_arxiv

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class FakeResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class BuildQueryTests(unittest.TestCase):
    def test_joins_words_with_and(self):
        self.assertEqual(
            build_query("LLM hallucination detection"),
            "all:LLM AND all:hallucination AND all:detection",
        )

    def test_drops_stopwords_and_special_characters(self):
        # quotes / colons / parentheses have meaning in arXiv syntax -> stripped
        self.assertEqual(
            build_query('Recent advances in "graph: neural" (networks)'),
            "all:graph AND all:neural AND all:networks",
        )

    def test_keeps_hyphenated_terms(self):
        self.assertEqual(build_query("few-shot learning"), "all:few-shot AND all:learning")

    def test_empty_or_only_stopwords_raises(self):
        for bad in ["", "   ", "the of in"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    build_query(bad)


class ParseFeedTests(unittest.TestCase):
    def setUp(self):
        self.papers = parse_feed(load("sample_feed.xml"))

    def test_parses_all_entries(self):
        self.assertEqual(len(self.papers), 2)

    def test_fields_extracted_and_whitespace_normalised(self):
        p = self.papers[0]
        self.assertEqual(p.arxiv_id, "2401.00001v2")
        # title was hard-wrapped over two lines in the XML
        self.assertEqual(p.title, "Example Paper: Detecting Hallucinations with Sampling Consistency")
        self.assertEqual(p.year, 2024)
        self.assertEqual(p.authors, ["Alice Example", "Bob Sample", "Carol Test", "Dan Fixture"])
        self.assertEqual(p.categories, ["cs.CL", "cs.AI"])
        self.assertEqual(p.pdf_url, "https://arxiv.org/pdf/2401.00001v2")
        self.assertEqual(p.abs_url, "https://arxiv.org/abs/2401.00001v2")
        self.assertNotIn("\n", p.abstract)
        self.assertTrue(p.abstract.startswith("We study"))

    def test_old_style_id_and_pdf_url_fallback(self):
        p = self.papers[1]
        self.assertEqual(p.arxiv_id, "hep-th/9901001v1")
        self.assertEqual(p.year, 1999)
        # no <link title="pdf"> in the XML -> URL is constructed
        self.assertEqual(p.pdf_url, "https://arxiv.org/pdf/hep-th/9901001v1")

    def test_citation_uses_et_al_for_many_authors(self):
        self.assertIn("Alice Example et al. (2024)", self.papers[0].citation())
        self.assertIn("Eve Legacy (1999)", self.papers[1].citation())

    def test_empty_feed_returns_empty_list(self):
        empty = '<feed xmlns="http://www.w3.org/2005/Atom"><title>x</title></feed>'
        self.assertEqual(parse_feed(empty), [])

    def test_arxiv_error_entry_raises(self):
        with self.assertRaises(ArxivError) as ctx:
            parse_feed(load("error_feed.xml"))
        self.assertIn("incorrect id format", str(ctx.exception))

    def test_one_malformed_entry_is_skipped_not_fatal(self):
        with self.assertLogs("lit_review.arxiv_client", level="WARNING"):
            papers = parse_feed(load("malformed_entry_feed.xml"))
        self.assertEqual([p.arxiv_id for p in papers], ["2401.22222v1"])

    def test_invalid_xml_raises_arxiv_error(self):
        with self.assertRaises(ArxivError):
            parse_feed("<feed><unclosed>")


class FetchAndSearchTests(unittest.TestCase):
    """Network layer, with requests.get and time.sleep replaced by fakes."""

    def setUp(self):
        # Don't actually sleep during tests, and reset the rate-limit clock.
        patcher = mock.patch.object(arxiv_client.time, "sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)
        arxiv_client._last_request_time = float("-inf")

    def test_search_success(self):
        ok = FakeResponse(200, load("sample_feed.xml"))
        with mock.patch.object(arxiv_client.requests, "get", return_value=ok) as get:
            papers = search_arxiv("hallucination detection", max_results=2)
        self.assertEqual(len(papers), 2)
        sent = get.call_args.kwargs["params"]
        self.assertEqual(sent["search_query"], "all:hallucination AND all:detection")
        self.assertEqual(sent["max_results"], 2)
        self.assertIn("timeout", get.call_args.kwargs)  # never request without a timeout

    def test_retries_transient_errors_then_succeeds(self):
        responses = [
            FakeResponse(503),
            requests.Timeout("slow"),
            FakeResponse(200, load("sample_feed.xml")),
        ]
        with mock.patch.object(arxiv_client.requests, "get", side_effect=responses) as get:
            with self.assertLogs("lit_review.arxiv_client", level="WARNING"):
                papers = search_arxiv("hallucination detection")
        self.assertEqual(get.call_count, 3)
        self.assertEqual(len(papers), 2)

    def test_gives_up_after_max_retries(self):
        with mock.patch.object(
            arxiv_client.requests, "get", return_value=FakeResponse(503)
        ) as get:
            with self.assertLogs("lit_review.arxiv_client", level="WARNING"):
                with self.assertRaises(ArxivError):
                    search_arxiv("hallucination detection")
        self.assertEqual(get.call_count, arxiv_client.MAX_RETRIES)

    def test_permanent_error_is_not_retried(self):
        with mock.patch.object(
            arxiv_client.requests, "get", return_value=FakeResponse(400)
        ) as get:
            with self.assertRaises(ArxivError):
                search_arxiv("hallucination detection")
        self.assertEqual(get.call_count, 1)

    def test_rate_limiter_sleeps_between_back_to_back_requests(self):
        ok = FakeResponse(200, load("sample_feed.xml"))
        with mock.patch.object(arxiv_client.requests, "get", return_value=ok):
            search_arxiv("hallucination detection")
            self.sleep.assert_not_called()  # first request: no waiting needed
            search_arxiv("hallucination detection")
        self.sleep.assert_called_once()
        waited = self.sleep.call_args.args[0]
        self.assertGreater(waited, 0)
        self.assertLessEqual(waited, arxiv_client.MIN_SECONDS_BETWEEN_REQUESTS)

    def test_bad_arguments_rejected_before_any_network_call(self):
        with mock.patch.object(arxiv_client.requests, "get") as get:
            for kwargs in [{"max_results": 0}, {"max_results": 101}, {"sort_by": "banana"}]:
                with self.subTest(**kwargs):
                    with self.assertRaises(ValueError):
                        search_arxiv("hallucination detection", **kwargs)
            with self.assertRaises(ValueError):
                search_arxiv("")
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
