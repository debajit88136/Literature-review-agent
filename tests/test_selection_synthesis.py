import json
import unittest

from helpers import Canned, mk_note, mk_paper
from lit_review.llm import LLMError
from lit_review.selection import MAX_POOL, build_select_prompt, select_top_papers
from lit_review.synthesis import (alias_map, build_cluster_prompt, build_gap_prompt, cluster_papers,
                                  identify_gaps, theme_limits)


def titles(items):
    return [x.title for x in items]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.papers = [mk_paper(i) for i in range(1, 13)]

    def test_model_order_preserved(self):
        r = select_top_papers(self.papers, "t", Canned('{"selected": ["P7", "P2", "P11"]}'), keep=3)
        self.assertEqual(titles(r), ["Paper 7", "Paper 2", "Paper 11"])

    def test_invalid_and_duplicate_ids_dropped_and_capped(self):
        reply = '{"selected": ["P3", "P3", "P99", "banana", "P5", "P1", "P2"]}'
        r = select_top_papers(self.papers, "t", Canned(reply), keep=3)
        self.assertEqual(titles(r), ["Paper 3", "Paper 5", "Paper 1"])

    def test_model_may_return_fewer(self):
        r = select_top_papers(self.papers, "t", Canned('{"selected": ["P4"]}'), keep=5)
        self.assertEqual(titles(r), ["Paper 4"])

    def test_bad_replies_fall_back_to_arxiv_order(self):
        for bad in ["garbage", '{"selected": []}', '{"selected": ["P99"]}', '{"selected": "P1"}', '{"x": 1}']:
            with self.subTest(bad=bad), self.assertLogs("lit_review", level="WARNING"):
                r = select_top_papers(self.papers, "t", Canned(bad), keep=3)
            self.assertEqual(titles(r), ["Paper 1", "Paper 2", "Paper 3"])
        with self.assertLogs("lit_review", level="WARNING"):
            r = select_top_papers(self.papers, "t", Canned(LLMError("down")), keep=2)
        self.assertEqual(titles(r), ["Paper 1", "Paper 2"])

    def test_pool_is_capped(self):
        big = [mk_paper(i) for i in range(1, 61)]
        llm = Canned('{"selected": ["P40", "P41", "P1"]}')
        r = select_top_papers(big, "t", llm, keep=5)
        self.assertIn(f"[P{MAX_POOL}]", llm.last_prompt)
        self.assertNotIn(f"[P{MAX_POOL + 1}]", llm.last_prompt)
        self.assertEqual(titles(r), ["Paper 40", "Paper 1"])

    def test_prompt_braces(self):
        p = build_select_prompt("topic x", [("P1", self.papers[0])], 3)
        self.assertIn('{"selected": ["P4", "P1", "P9"]}', p)


class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.notes = [mk_note(i) for i in range(1, 6)]

    def test_good_reply(self):
        reply = json.dumps({"themes": [{"name": "A", "description": "d", "papers": ["P1", "P2"]},
                                       {"name": "B", "description": "d", "papers": ["P3", "P4", "P5"]}]})
        themes = cluster_papers(self.notes, "t", Canned(reply))
        self.assertEqual([len(t.notes) for t in themes], [2, 3])

    def test_every_paper_lands_in_exactly_one_theme(self):
        reply = json.dumps({"themes": [
            {"name": "Group A", "description": "x", "papers": ["P1", "P99", "P1"]},
            {"name": "", "description": "y", "papers": ["P2"]},
            "not a dict",
            {"name": "C", "description": "z", "papers": ["P1"]},
        ]})
        themes = cluster_papers(self.notes, "t", Canned(reply))
        self.assertEqual([t.name for t in themes], ["Group A", "Other approaches"])
        placed = [n.paper.title for t in themes for n in t.notes]
        self.assertEqual(sorted(placed), [f"Paper {i}" for i in range(1, 6)])

    def test_garbage_falls_back_to_grouping_by_type(self):
        notes = [mk_note(1, "survey"), mk_note(2, "new_method"), mk_note(3, "survey")]
        with self.assertLogs("lit_review", level="WARNING"):
            themes = cluster_papers(notes, "t", Canned("nope"))
        self.assertEqual({t.name: len(t.notes) for t in themes},
                         {"Surveys and overviews": 2, "New methods": 1})

    def test_failed_notes_excluded_and_empty_input(self):
        themes = cluster_papers([mk_note(1), mk_note(2, ok=False)], "t", Canned("never called"))
        self.assertEqual(sum(len(t.notes) for t in themes), 1)
        self.assertEqual(cluster_papers([], "t", Canned("x")), [])

    def test_retry_after_invalid_structure(self):
        good = json.dumps({"themes": [{"name": "A", "description": "d", "papers": ["P1", "P2", "P3", "P4", "P5"]}]})
        llm = Canned('{"themes": []}', good)
        with self.assertLogs("lit_review", level="WARNING"):
            themes = cluster_papers(self.notes, "t", llm)
        self.assertEqual((llm.calls, len(themes)), (2, 1))

    def test_theme_limits_and_prompt(self):
        self.assertEqual([theme_limits(n) for n in (1, 2, 5, 10, 15, 20, 60)],
                         [(1, 1), (1, 2), (2, 2), (2, 3), (3, 5), (4, 6), (4, 6)])
        self.assertIn('{"themes": [{"name": "..."', build_cluster_prompt("t", alias_map(self.notes), 2, 3))


class GapTests(unittest.TestCase):
    def setUp(self):
        self.notes = [mk_note(1, problems=["needs more models"]), mk_note(2),
                      mk_note(3, problems=["needs more models", "slow"]),
                      mk_note(4, problems=["slow"]), mk_note(5, problems=["only one"])]

    def test_only_gaps_backed_by_two_valid_papers_survive(self):
        reply = json.dumps({"gaps": [
            {"title": "Solo", "description": "d", "papers": ["P1"]},
            {"title": "Models", "description": "d", "papers": ["P1", "P3", "P3"]},
            {"title": "Speed", "description": "d", "papers": ["P3", "P4", "P2", "P99"]},
            {"title": "Ghost", "description": "d", "papers": ["P2", "P99"]},
        ]})
        gaps = identify_gaps(self.notes, "t", Canned(reply))
        self.assertEqual(sorted(g.title for g in gaps), ["Models", "Speed"])
        self.assertTrue(all(len(g.notes) == 2 for g in gaps))

    def test_failure_paths_return_empty_list(self):
        with self.assertLogs("lit_review", level="WARNING"):
            self.assertEqual(identify_gaps(self.notes, "t", Canned("nonsense")), [])
            self.assertEqual(identify_gaps(self.notes, "t", Canned(LLMError("down"))), [])
        few = [mk_note(1, problems=["x"]), mk_note(2)]
        self.assertEqual(identify_gaps(few, "t", Canned("never called")), [])

    def test_prompt_braces(self):
        p = build_gap_prompt("t", {"P1": self.notes[0]})
        self.assertIn('{"gaps": [{"title"', p)


if __name__ == "__main__":
    unittest.main()
