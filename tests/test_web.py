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
                def __init__(self, planner, tools, path, max_steps=12):
                    self.planner = planner
                    self.max_steps = max_steps

                def run(self, goal, resume=False):
                    self.assert_key()
                    if resume or self.max_steps != 12:
                        raise AssertionError("unexpected resume configuration")
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
        self.assertIn('id="runMode"', web.HTML)
        self.assertIn('id="savedRun"', web.HTML)
        self.assertIn('id="maxSteps"', web.HTML)
        self.assertIn("fetch('/api/runs')", web.HTML)
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
                def __init__(self, planner, tools, path, max_steps=12):
                    if planner.received_key != sentinel or planner.model != "gpt-5.6-sol":
                        raise AssertionError("Libra routing did not preserve the selected model and key")
                    if max_steps != 12:
                        raise AssertionError("unexpected step budget")

                def run(self, goal, resume=False):
                    if resume:
                        raise AssertionError("unexpected resume")
                    return SimpleNamespace(
                        status="complete", final_answer="done", step=1,
                        provider_retries=0, tool_retries=0, validation_failures=0,
                    )

            with patch.object(web, "LibraPlanner", FakeLibraPlanner), patch.object(web, "ResearchAgent", FakeAgent):
                web._run_agent("libra-test", "goal", False, "libra", "gpt-5.6-sol", run_dir, sentinel)

        serialized = json.dumps(web.RUNS["libra-test"])
        self.assertNotIn(sentinel, serialized)
        self.assertEqual(web.RUNS["libra-test"]["status"], "complete")

    def test_saved_runs_lists_only_valid_resumable_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(web, "RUNS_ROOT", Path(tmp)):
            resumable_dir = web._safe_run_dir("abcde12345")
            web._write_run_config(resumable_dir, "abcde12345", "live", "groq", "model-x")
            (resumable_dir / "checkpoint.json").write_text(
                json.dumps({"goal": "resume this research", "status": "budget_exhausted", "step": 12}),
                encoding="utf-8",
            )

            complete_dir = web._safe_run_dir("fffff11111")
            web._write_run_config(complete_dir, "fffff11111", "demo", "openai", "demo")
            (complete_dir / "checkpoint.json").write_text(
                json.dumps({"goal": "already done", "status": "complete", "step": 7}),
                encoding="utf-8",
            )

            saved = web._saved_runs()
            metadata_text = (resumable_dir / "run.json").read_text(encoding="utf-8")

        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["id"], "abcde12345")
        self.assertEqual(saved[0]["minimum_max_steps"], 13)
        self.assertEqual(saved[0]["provider"], "groq")
        self.assertNotIn("api_key", metadata_text.lower())

    def test_saved_run_path_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(web, "RUNS_ROOT", Path(tmp)):
            for unsafe_id in ("../escape", "not/flat", "ABC", ""):
                with self.subTest(run_id=unsafe_id), self.assertRaisesRegex(ValueError, "invalid saved run ID"):
                    web._safe_run_dir(unsafe_id)

    def test_resume_worker_forwards_resume_and_total_budget(self):
        sentinel = "sentinel-resume-key"
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "checkpoint.json").write_text(json.dumps({"step": 4}), encoding="utf-8")
            with web.RUNS_LOCK:
                web.RUNS["resume-test"] = {
                    "id": "resume-test", "goal": "goal", "mode": "live", "provider": "groq",
                    "status": "starting", "answer": None, "error": None,
                    "trace_path": str(run_dir / "trace.jsonl"),
                }

            class FakePlanner:
                def __init__(self, model, api_key):
                    if model != "model-x" or api_key != sentinel:
                        raise AssertionError("resume provider configuration changed")

            class FakeAgent:
                def __init__(self, planner, tools, path, max_steps=12):
                    if path != run_dir or max_steps != 20:
                        raise AssertionError("resume path or budget changed")

                def run(self, goal, resume=False):
                    if goal != "goal" or not resume:
                        raise AssertionError("resume flag or goal changed")
                    return SimpleNamespace(
                        status="complete", final_answer="done", step=5,
                        provider_retries=0, tool_retries=0, validation_failures=0, quality_checks={},
                    )

            with patch.object(web, "GroqPlanner", FakePlanner), patch.object(web, "ResearchAgent", FakeAgent):
                web._run_agent(
                    "resume-test", "goal", False, "groq", "model-x", run_dir, sentinel, True, 20
                )

        serialized = json.dumps(web.RUNS["resume-test"])
        self.assertNotIn(sentinel, serialized)
        self.assertEqual(web.RUNS["resume-test"]["status"], "complete")

    def test_offline_demo_resumes_from_budget_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            exhausted = web.ResearchAgent(
                web.ScriptedPlanner(web._demo_actions()[:4]),
                web.LocalCorpusTools(web.ROOT / "evals" / "corpus"),
                run_dir,
                max_steps=4,
            ).run("Compare checkpointing and retry logic for resilient agents.")
            with web.RUNS_LOCK:
                web.RUNS["demo-resume"] = {
                    "id": "demo-resume", "goal": exhausted.goal, "mode": "demo", "provider": "openai",
                    "status": "starting", "answer": None, "error": None,
                    "trace_path": str(run_dir / "trace.jsonl"),
                }
            web._run_agent(
                "demo-resume", exhausted.goal, True, "openai", "demo", run_dir, "", True, 7
            )
            trace = (run_dir / "trace.jsonl").read_text(encoding="utf-8")

        self.assertEqual(exhausted.status, "budget_exhausted")
        self.assertEqual(web.RUNS["demo-resume"]["status"], "complete")
        self.assertIn('"event": "run_resumed"', trace)


if __name__ == "__main__":
    unittest.main()
