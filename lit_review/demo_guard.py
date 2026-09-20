"""Protects a shared API key: limits demo runs per day and allows one run at a time.

Pure Python (no Streamlit), so it can be tested with a fake clock. The app keeps one
instance alive for the whole server process, shared by all visitors.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Callable, Optional


class DemoGuard:
    def __init__(self, max_runs_per_day: int, max_run_seconds: int = 900,
                 clock: Optional[Callable[[], datetime]] = None):
        self.max_runs_per_day = max_runs_per_day
        self.max_run_seconds = max_run_seconds
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._day = None
        self._runs_today = 0
        self._running_since: Optional[datetime] = None

    def _roll_day(self, now: datetime) -> None:
        if self._day != now.date():
            self._day = now.date()
            self._runs_today = 0

    @property
    def runs_left_today(self) -> int:
        with self._lock:
            self._roll_day(self._clock())
            return max(0, self.max_runs_per_day - self._runs_today)

    def try_start(self) -> tuple[bool, str]:
        with self._lock:
            now = self._clock()
            self._roll_day(now)
            if self._running_since is not None:
                if (now - self._running_since).total_seconds() < self.max_run_seconds:
                    return False, "Another demo review is running right now. Please try again in a few minutes."
            if self._runs_today >= self.max_runs_per_day:
                return False, ("The daily limit for demo runs has been reached. Try again tomorrow, "
                               "or paste your own free Gemini API key in the sidebar to remove the limit.")
            self._running_since = now
            self._runs_today += 1
            return True, ""

    def finish(self) -> None:
        with self._lock:
            self._running_since = None
