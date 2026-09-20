import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from helpers import Canned
from lit_review import llm as llm_module
from lit_review.llm import GeminiClient, LLMError, ask_json, parse_json_object


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"error {code}")
        self.code = code


def fake_google_modules(script):
    calls = []

    class Client:
        def __init__(self, api_key):
            self.models = SimpleNamespace(generate_content=self.generate_content)

        def generate_content(self, model, contents, config):
            calls.append(SimpleNamespace(model=model, contents=contents, config=config))
            item = script[min(len(calls) - 1, len(script) - 1)]
            if isinstance(item, Exception):
                raise item
            return SimpleNamespace(text=item)

    google = types.ModuleType("google")
    google.__path__ = []
    genai = types.ModuleType("google.genai")
    genai.Client = Client
    errors = types.ModuleType("google.genai.errors")
    errors.APIError = FakeAPIError
    type_mod = types.ModuleType("google.genai.types")
    type_mod.GenerateContentConfig = lambda **kw: SimpleNamespace(**kw)
    google.genai = genai
    genai.errors = errors
    genai.types = type_mod
    modules = {"google": google, "google.genai": genai,
               "google.genai.errors": errors, "google.genai.types": type_mod}
    return modules, calls


class GeminiClientTests(unittest.TestCase):
    def make(self, script, **kwargs):
        modules, calls = fake_google_modules(script)
        patcher = mock.patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        sleeper = mock.patch.object(llm_module.time, "sleep")
        self.sleep = sleeper.start()
        self.addCleanup(sleeper.stop)
        return GeminiClient("key", "model-x", **kwargs), calls

    def test_success_and_config(self):
        client, calls = self.make(['{"ok": true}'])
        self.assertEqual(client.complete("sys", "hello"), '{"ok": true}')
        cfg = calls[0].config
        self.assertEqual(cfg.system_instruction, "sys")
        self.assertEqual(cfg.response_mime_type, "application/json")
        self.assertGreaterEqual(cfg.max_output_tokens, 4096)
        self.assertEqual(calls[0].model, "model-x")

    def test_retries_rate_limit_then_succeeds(self):
        client, calls = self.make([FakeAPIError(429), FakeAPIError(503), "fine"])
        with self.assertLogs("lit_review.llm", level="WARNING"):
            self.assertEqual(client.complete("s", "p"), "fine")
        self.assertEqual(len(calls), 3)

    def test_permanent_error_not_retried(self):
        client, calls = self.make([FakeAPIError(400)])
        with self.assertRaises(LLMError):
            client.complete("s", "p")
        self.assertEqual(len(calls), 1)

    def test_gives_up_after_max_attempts(self):
        client, calls = self.make([FakeAPIError(429)], max_attempts=3)
        with self.assertLogs("lit_review.llm", level="WARNING"):
            with self.assertRaises(LLMError):
                client.complete("s", "p")
        self.assertEqual(len(calls), 3)

    def test_empty_reply_and_unexpected_error(self):
        client, _ = self.make(["   "])
        with self.assertRaises(LLMError):
            client.complete("s", "p")
        client, _ = self.make([ConnectionError("boom")])
        with self.assertRaises(LLMError):
            client.complete("s", "p")

    def test_client_spaces_out_back_to_back_calls(self):
        client, _ = self.make(["a", "b"], min_gap=6.0)
        client.complete("s", "p")
        self.sleep.assert_not_called()
        client.complete("s", "p")
        self.sleep.assert_called_once()
        self.assertTrue(0 < self.sleep.call_args.args[0] <= 6.0)


class JsonHelperTests(unittest.TestCase):
    def test_parse_json_object(self):
        self.assertEqual(parse_json_object('noise {"a": 1} tail'), {"a": 1})
        self.assertEqual(parse_json_object('```json\n{"a": [1, 2]}\n```'), {"a": [1, 2]})
        for bad in ["", "no json", "{broken", "{'a': 1}"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_json_object(bad)

    def test_ask_json_retries_bad_reply_then_succeeds(self):
        llm = Canned("garbage", '{"x": 5}')
        with self.assertLogs("lit_review.llm", level="WARNING"):
            self.assertEqual(ask_json(llm, "s", "p", lambda d: d["x"]), 5)
        self.assertEqual(llm.calls, 2)

    def test_ask_json_raises_after_attempts(self):
        with self.assertLogs("lit_review.llm", level="WARNING"):
            with self.assertRaises(ValueError):
                ask_json(Canned("garbage"), "s", "p", lambda d: d)

    def test_build_function_can_reject_reply(self):
        def build(d):
            raise ValueError("wrong shape")
        with self.assertLogs("lit_review.llm", level="WARNING"):
            with self.assertRaises(ValueError):
                ask_json(Canned('{"a": 1}'), "s", "p", build)

    def test_llm_error_propagates_without_retry(self):
        llm = Canned(LLMError("api down"))
        with self.assertRaises(LLMError):
            ask_json(llm, "s", "p", lambda d: d)
        self.assertEqual(llm.calls, 1)


if __name__ == "__main__":
    unittest.main()
