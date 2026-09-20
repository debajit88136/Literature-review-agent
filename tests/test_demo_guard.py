import threading
import unittest
from datetime import datetime, timedelta, timezone

from lit_review.demo_guard import DemoGuard


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)


class DemoGuardTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.guard = DemoGuard(max_runs_per_day=2, max_run_seconds=600, clock=self.clock)

    def test_allows_a_run_then_blocks_a_concurrent_one(self):
        self.assertEqual(self.guard.try_start(), (True, ""))
        ok, reason = self.guard.try_start()
        self.assertFalse(ok)
        self.assertIn("running right now", reason)

    def test_next_run_allowed_after_finish(self):
        self.guard.try_start()
        self.guard.finish()
        self.assertTrue(self.guard.try_start()[0])

    def test_daily_limit_and_message_mentions_own_key(self):
        for _ in range(2):
            self.assertTrue(self.guard.try_start()[0])
            self.guard.finish()
        ok, reason = self.guard.try_start()
        self.assertFalse(ok)
        self.assertIn("own free Gemini API key", reason)
        self.assertEqual(self.guard.runs_left_today, 0)

    def test_limit_resets_next_day(self):
        for _ in range(2):
            self.guard.try_start()
            self.guard.finish()
        self.assertFalse(self.guard.try_start()[0])
        self.clock.advance(days=1)
        self.assertEqual(self.guard.runs_left_today, 2)
        self.assertTrue(self.guard.try_start()[0])

    def test_stuck_run_expires(self):
        self.guard.try_start()
        self.clock.advance(seconds=599)
        self.assertFalse(self.guard.try_start()[0])
        self.clock.advance(seconds=2)
        self.assertTrue(self.guard.try_start()[0])

    def test_runs_left_counts_down(self):
        self.assertEqual(self.guard.runs_left_today, 2)
        self.guard.try_start()
        self.assertEqual(self.guard.runs_left_today, 1)

    def test_only_one_of_many_simultaneous_visitors_starts(self):
        guard = DemoGuard(max_runs_per_day=100)
        results = []
        barrier = threading.Barrier(20)

        def visitor():
            barrier.wait()
            results.append(guard.try_start()[0])

        threads = [threading.Thread(target=visitor) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results.count(True), 1)


if __name__ == "__main__":
    unittest.main()
