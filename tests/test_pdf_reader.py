import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import requests

from helpers import mk_paper
from lit_review import pdf_reader
from lit_review.pdf_reader import (HEAD_CHARS, TAIL_CHARS, PdfError, download_pdf, extract_text,
                                   get_paper_content, shorten, strip_references)


def fake_pymupdf(pages=None, needs_pass=False, open_error=None):
    class Doc:
        def __init__(self):
            self.needs_pass = needs_pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return iter(SimpleNamespace(get_text=lambda t=t: t) for t in (pages or []))

    def open_(stream, filetype):
        if open_error:
            raise open_error
        return Doc()

    module = types.ModuleType("pymupdf")
    module.open = open_
    return module


class TextHelperTests(unittest.TestCase):
    def test_strip_references_only_when_near_the_end(self):
        body = "x" * 1000
        self.assertEqual(strip_references(body + "\nReferences\n[1] cite"), body)
        early = "Intro\nReferences\n" + "y" * 1000
        self.assertEqual(strip_references(early), early)
        self.assertEqual(strip_references("no refs here"), "no refs here")

    def test_shorten_keeps_head_and_tail(self):
        short = "a" * 100
        self.assertEqual(shorten(short), short)
        long = "H" * HEAD_CHARS + "M" * 5000 + "T" * TAIL_CHARS
        out = shorten(long)
        self.assertTrue(out.startswith("H" * HEAD_CHARS))
        self.assertTrue(out.endswith("T" * TAIL_CHARS))
        self.assertNotIn("M", out)
        self.assertIn("omitted", out)


class ExtractTextTests(unittest.TestCase):
    def run_with(self, module):
        with mock.patch.dict(sys.modules, {"pymupdf": module}):
            return extract_text(b"%PDF-fake")

    def test_joins_pages_and_normalises_whitespace(self):
        text = self.run_with(fake_pymupdf(["Hello    world\n\n\n\nnext", "page two  "]))
        self.assertEqual(text, "Hello world\n\nnext\npage two")

    def test_password_protected(self):
        with self.assertRaises(PdfError):
            self.run_with(fake_pymupdf(needs_pass=True))

    def test_library_error_becomes_pdf_error(self):
        with self.assertRaises(PdfError):
            self.run_with(fake_pymupdf(open_error=RuntimeError("corrupt")))


class DownloadTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(pdf_reader, "wait_for_rate_limit")
        patcher.start()
        self.addCleanup(patcher.stop)

    def get(self, **response):
        return mock.patch.object(pdf_reader.requests, "get",
                                 return_value=SimpleNamespace(**response))

    def test_ok(self):
        with self.get(status_code=200, content=b"%PDF-1.5 data"):
            self.assertEqual(download_pdf("u"), b"%PDF-1.5 data")

    def test_failures_become_pdf_error(self):
        cases = [
            dict(status_code=404, content=b""),
            dict(status_code=200, content=b"<html>not a pdf</html>"),
            dict(status_code=200, content=b"%PDF" + b"0" * (pdf_reader.MAX_PDF_BYTES + 1)),
        ]
        for response in cases:
            with self.subTest(status=response["status_code"]), self.get(**response):
                with self.assertRaises(PdfError):
                    download_pdf("u")
        with mock.patch.object(pdf_reader.requests, "get", side_effect=requests.Timeout("slow")):
            with self.assertRaises(PdfError):
                download_pdf("u")


class GetPaperContentTests(unittest.TestCase):
    def setUp(self):
        self.paper = mk_paper(1, abstract="The abstract.")

    def patched(self, download=b"%PDF", text="word " * 1000, download_error=None):
        return (
            mock.patch.object(pdf_reader, "download_pdf", side_effect=download_error, return_value=download),
            mock.patch.object(pdf_reader, "extract_text", return_value=text),
        )

    def run_case(self, **kwargs):
        d, e = self.patched(**kwargs)
        with d, e:
            return get_paper_content(self.paper)

    def test_full_text(self):
        c = self.run_case()
        self.assertEqual(c.source, "full_text")
        self.assertIn("word", c.text)

    def test_too_little_text_falls_back(self):
        with self.assertLogs("lit_review.pdf_reader", level="WARNING"):
            c = self.run_case(text="tiny")
        self.assertEqual(c.source, "abstract_only")
        self.assertIn("The abstract.", c.text)
        self.assertIn("scanned", c.note)

    def test_pdf_error_falls_back(self):
        with self.assertLogs("lit_review.pdf_reader", level="WARNING"):
            c = self.run_case(download_error=PdfError("HTTP 404"))
        self.assertEqual((c.source, c.note), ("abstract_only", "HTTP 404"))

    def test_unexpected_error_still_never_raises(self):
        with self.assertLogs("lit_review.pdf_reader", level="ERROR"):
            c = self.run_case(download_error=ZeroDivisionError("bug"))
        self.assertEqual(c.source, "abstract_only")
        self.assertIn("unexpected", c.note)


if __name__ == "__main__":
    unittest.main()
