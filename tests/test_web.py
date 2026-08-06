from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research_agent import web


class WebSecurityTests(unittest.TestCase):
    def setUp(self):
        with web.RUNS_LOCK:
            web.RUNS.clear()

    def test_run_status_does_not_retain_api_key(self):
        sentinel = "sentinel-secret-key"
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with web.RUNS_LOCK:
                web.RUNS["test"] = {
                    "id": "test", "goal": "goal", "mode": "live", "provider": "groq",
                    "status": "starting", "answer": None, "error": None,
                    "trace_path": str(run_dir / "trace.jsonl"),
                }

            class FakePlanner:
                def __init__(self, model, api_key):
                    self.model = model
                    self.received_key = api_key

            class FakeAgent:
                def __init__(self, planner, tools, path):
                    self.planner = planner

                def run(self, goal):
                    self.assert_key()
                    return SimpleNamespace(
                        status="complete",
                        final_answer="done",
                        step=1,
                        provider_retries=2,
                        tool_retries=1,
                        validation_failures=3,
                    )

                def assert_key(self):
                    if self.planner.received_key != sentinel:
                        raise AssertionError("key was not passed to the selected provider")

            with patch.object(web, "GroqPlanner", FakePlanner), patch.object(web, "ResearchAgent", FakeAgent):
                web._run_agent("test", "goal", False, "groq", "model", run_dir, sentinel)

        serialized = json.dumps(web.RUNS["test"])
        self.assertNotIn(sentinel, serialized)
        self.assertEqual(web.RUNS["test"]["status"], "complete")
        self.assertEqual(web.RUNS["test"]["provider_retries"], 2)
        self.assertEqual(web.RUNS["test"]["tool_retries"], 1)
        self.assertEqual(web.RUNS["test"]["validation_failures"], 3)

    def test_ui_does_not_use_browser_storage_for_keys(self):
        self.assertNotIn("localStorage", web.HTML)
        self.assertNotIn("sessionStorage", web.HTML)
        self.assertIn('autocomplete="off"', web.HTML)
        self.assertIn('id="providerRetries"', web.HTML)
        self.assertIn('id="validationFailures"', web.HTML)
        self.assertIn('id="goalCoverage"', web.HTML)
        self.assertIn('id="citationCoverage"', web.HTML)
        self.assertIn('<option value="libra">Libra / Company</option>', web.HTML)
        self.assertIn("libra:['gpt-5.6-sol','Libra']", web.HTML)

    def test_libra_run_routes_key_without_retaining_it(self):
        sentinel = "sentinel-libra-key"
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with web.RUNS_LOCK:
                web.RUNS["libra-test"] = {
                    "id": "libra-test", "goal": "goal", "mode": "live", "provider": "libra",
                    "status": "starting", "answer": None, "error": None,
                    "trace_path": str(run_dir / "trace.jsonl"),
                }

            class FakeLibraPlanner:
                def __init__(self, model, api_key):
                    self.model = model
                    self.received_key = api_key

            class FakeAgent:
                def __init__(self, planner, tools, path):
                    if planner.received_key != sentinel or planner.model != "gpt-5.6-sol":
                        raise AssertionError("Libra routing did not preserve the selected model and key")

                def run(self, goal):
                    return SimpleNamespace(
                        status="complete", final_answer="done", step=1,
                        provider_retries=0, tool_retries=0, validation_failures=0,
                    )

            with patch.object(web, "LibraPlanner", FakeLibraPlanner), patch.object(web, "ResearchAgent", FakeAgent):
                web._run_agent("libra-test", "goal", False, "libra", "gpt-5.6-sol", run_dir, sentinel)

        serialized = json.dumps(web.RUNS["libra-test"])
        self.assertNotIn(sentinel, serialized)
        self.assertEqual(web.RUNS["libra-test"]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
