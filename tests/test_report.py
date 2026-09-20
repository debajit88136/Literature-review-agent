import unittest

from helpers import mk_note
from lit_review.models import Gap, Paper, PaperNotes, Theme
from lit_review.report import build_markdown_review, one_line, short_cite


def rich_note(i, source="full_text", problems=None):
    paper = Paper(f"2401.0000{i}v1", f"Title {i}", [f"First{i} Last{i}", "B", "C", "D"], "abs",
                  2020 + i, f"https://arxiv.org/abs/2401.0000{i}v1", "p")
    return PaperNotes(paper, source, "new_method", f"Method\n{i} text", [f"finding {i}", "# sneaky heading"],
                      [f"DS{i}"], f"relation {i}", problems or [], i)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.n1, self.n2, self.n3 = rich_note(1, problems=["a"]), rich_note(2, "abstract_only", ["a"]), rich_note(3)
        self.failed = mk_note(9, ok=False)
        self.themes = [Theme("Theme One", "About one.", [self.n1, self.n2]),
                       Theme("Theme Two", "About two.", [self.n3])]
        self.gaps = [Gap("Shared gap", "Both mention a.", [self.n1, self.n2])]
        self.md = build_markdown_review("test topic", [self.n1, self.n2, self.n3, self.failed],
                                        self.themes, self.gaps, model_name="fake-model")

    def test_header_and_overview(self):
        self.assertTrue(self.md.startswith("# Literature Review: test topic"))
        self.assertIn("using fake-model", self.md)
        self.assertIn("from 4 arXiv papers", self.md)
        self.assertIn("analyses 3 papers (2 from full text", self.md)
        self.assertIn("1 from the abstract only), grouped into 2 themes", self.md)
        self.assertIn("(1 paper)", self.md)

    def test_every_paper_cited_in_body_and_references(self):
        for n in (self.n1, self.n2, self.n3, self.failed):
            self.assertEqual(self.md.count(n.paper.citation()), 2)

    def test_section_order_and_paper_order(self):
        md = self.md
        self.assertLess(md.index("## Theme One"), md.index("## Theme Two"))
        self.assertLess(md.index("## Theme Two"), md.index("## Recurring open problems"))
        self.assertLess(md.index("## Recurring open problems"), md.index("## References"))
        self.assertLess(md.index("### Title 2"), md.index("### Title 1"))

    def test_gap_links_and_flags(self):
        self.assertIn("Raised by: [First1 Last1 et al. (2021)](https://arxiv.org/abs/2401.00001v1); "
                      "[First2 Last2 et al. (2022)]", self.md)
        self.assertEqual(self.md.count("Notes based on the abstract only"), 1)
        self.assertIn("Papers that could not be processed", self.md)

    def test_llm_text_cannot_inject_headings_or_newlines(self):
        self.assertNotIn("\n### sneaky", self.md)
        self.assertIn("  - sneaky heading", self.md)
        self.assertIn("- **Approach:** Method 1 text", self.md)

    def test_edge_cases(self):
        no_gaps = build_markdown_review("t", [self.n1], [Theme("Solo", "d", [self.n1])], [])
        self.assertIn("No problem shared by two or more papers", no_gaps)
        self.assertNotIn("could not be processed", no_gaps)
        self.assertNotIn(" using ", no_gaps.split("\n")[2])
        all_failed = build_markdown_review("t", [self.failed], [], [])
        self.assertIn("analyses 0 papers", all_failed)
        self.assertIn("could not be processed", all_failed)

    def test_helpers(self):
        self.assertEqual(one_line("## a\n  b\tc"), "a b c")
        self.assertIn("A and B (2020)", short_cite(Paper("i", "t", ["A", "B"], "", 2020, "u", "p")))
        self.assertIn("Unknown authors", short_cite(Paper("i", "t", [], "", 2020, "u", "p")))


if __name__ == "__main__":
    unittest.main()
