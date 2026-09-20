"""LLM access behind a tiny interface, plus helpers for structured JSON replies.

Every other module talks to `LLMClient.complete()` only. Swapping providers
means writing one new subclass; nothing else changes.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMError(Exception):
    pass


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, prompt: str, max_tokens: int = 1500) -> str:
        ...


class GeminiClient(LLMClient):
    RETRYABLE_CODES = (429, 500, 503, 504)

    def __init__(self, api_key: str, model: str, min_gap: float = 6.0,
                 max_attempts: int = 4, json_mode: bool = True):
        from google import genai
        from google.genai import errors, types

        self.model = model
        self.min_gap = min_gap
        self.max_attempts = max_attempts
        self.json_mode = json_mode
        self._types = types
        self._api_error = errors.APIError
        self._client = genai.Client(api_key=api_key)
        self._last_call = float("-inf")

    def _wait_turn(self) -> None:
        gap = time.monotonic() - self._last_call
        if gap < self.min_gap:
            time.sleep(self.min_gap - gap)
        self._last_call = time.monotonic()

    def complete(self, system: str, prompt: str, max_tokens: int = 1500) -> str:
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max(max_tokens, 4096),
            temperature=0.2,
            response_mime_type="application/json" if self.json_mode else None,
        )
        last_error = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            self._wait_turn()
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config)
            except self._api_error as e:
                code = getattr(e, "code", None)
                last_error = f"Gemini API error {code}: {e}"
                if code not in self.RETRYABLE_CODES:
                    raise LLMError(last_error) from e
                if attempt < self.max_attempts:
                    wait = 15 * attempt
                    logger.warning("%s | retrying in %ds (%d/%d)",
                                   last_error[:120], wait, attempt, self.max_attempts - 1)
                    time.sleep(wait)
                continue
            except Exception as e:
                raise LLMError(f"unexpected LLM failure: {e}") from e
            text = response.text or ""
            if not text.strip():
                raise LLMError("empty response from model (possibly blocked)")
            return text
        raise LLMError(f"Gemini failed after {self.max_attempts} attempts: {last_error}")


def parse_json_object(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e


def ask_json(llm: LLMClient, system: str, prompt: str, build: Callable[[dict], T],
             attempts: int = 2, label: str = "LLM call") -> T:
    """Ask for JSON, validate it with `build`, retry once if the reply is unusable.

    LLMError (API-level failure) propagates immediately: the client already retried.
    ValueError (reply was unusable) is retried, then re-raised.
    """
    last_error = "unknown error"
    for attempt in range(1, attempts + 1):
        try:
            return build(parse_json_object(llm.complete(system, prompt)))
        except ValueError as e:
            last_error = str(e)
            logger.warning("%s: attempt %d/%d failed (%s)", label, attempt, attempts, e)
    raise ValueError(f"{label} failed after {attempts} attempts: {last_error}")
