from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from research_agent import web
from research_agent.metrics import render_metrics
from research_agent.qa import inspect_run, judge_with_deepeval, publish_qa_to_langfuse


class MetricsTests(unittest.TestCase):
    def test_web_exposes_prometheus_metrics(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(web, "RUNS_ROOT", Path(tmp)):
            response = MagicMock()
            web.Handler.do_GET(SimpleNamespace(path="/metrics", _send=response))
        self.assertIn("research_agent_run_checkpoints", response.call_args.args[1])
        self.assertIn("text/plain", response.call_args.args[2])

    def test_aggregate_metrics_exclude_user_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "abc"
            run.mkdir()
            run.joinpath("checkpoint.json").write_text(json.dumps({
                "goal": "private research goal", "final_answer": "private answer",
                "status": "failed", "step": 3, "provider_retries": 2,
                "tool_retries": 1, "validation_failures": 4,
            }), encoding="utf-8")
            bad = root / "bad"
            bad.mkdir()
            bad.joinpath("checkpoint.json").write_text("not json", encoding="utf-8")
            output = render_metrics(root)
        self.assertIn('research_agent_run_checkpoints{status="failed"} 1', output)
        self.assertIn("research_agent_recorded_provider_retries 2", output)
        self.assertIn("research_agent_recorded_tool_retries 1", output)
        self.assertIn("research_agent_malformed_checkpoints 1", output)
        self.assertNotIn("private", output)
        self.assertNotIn("abc", output)


class SavedRunQATests(unittest.TestCase):
    def test_optional_langfuse_trace_excludes_run_content(self):
        payloads = []

        class Span:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def update(self, **kwargs):
                payloads.append(kwargs)

        client = SimpleNamespace(start_as_current_observation=lambda **_kwargs: Span(), flush=lambda: None)
        module = types.ModuleType("langfuse")
        module.get_client = lambda: client
        with patch.dict(sys.modules, {"langfuse": module}):
            publish_qa_to_langfuse({
                "run_id": "run-1", "goal": "private goal", "answer": "private answer",
                "structural_pass": True, "checks": {"completed": True},
                "judge": {"faithfulness": {"score": 0.9, "reason": "private evidence"}},
            })
        serialized = json.dumps(payloads)
        self.assertIn("run-1", serialized)
        self.assertNotIn("private", serialized)

    def test_optional_deepeval_adapter_returns_scores(self):
        class Metric:
            def __init__(self, **_kwargs):
                self.score, self.reason = 0.8, "supported"

            def measure(self, case):
                if not case.retrieval_context:
                    raise AssertionError("missing evidence")

            def is_successful(self):
                return True

        class Case:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        package = types.ModuleType("deepeval")
        metrics = types.ModuleType("deepeval.metrics")
        metrics.AnswerRelevancyMetric = Metric
        metrics.FaithfulnessMetric = Metric
        cases = types.ModuleType("deepeval.test_case")
        cases.LLMTestCase = Case
        with patch.dict(sys.modules, {
            "deepeval": package, "deepeval.metrics": metrics, "deepeval.test_case": cases,
        }):
            result = judge_with_deepeval({
                "goal": "goal", "answer": "answer", "evidence_excerpts": ["evidence"],
            })
        self.assertTrue(all(item["passed"] for item in result.values()))

    def test_complete_cited_run_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            run.joinpath("checkpoint.json").write_text(json.dumps({
                "goal": "Explain checkpointing.", "run_id": "run-1", "step": 4,
                "status": "complete", "evidence": [{"source_id": "S1", "title": "Checkpoint",
                    "locator": "https://example.test/one", "excerpt": "A checkpoint preserves state."}],
                "final_answer": "A checkpoint preserves state [S1].",
            }), encoding="utf-8")
            run.joinpath("trace.jsonl").write_text(json.dumps({
                "run_id": "run-1", "step": 4, "event": "run_completed", "detail": {},
            }) + "\n", encoding="utf-8")
            report = inspect_run(run)
        self.assertTrue(report["structural_pass"])

    def test_unsupported_citation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            run.joinpath("checkpoint.json").write_text(json.dumps({
                "goal": "Explain checkpointing.", "run_id": "run-2", "status": "complete",
                "evidence": [{"source_id": "S1", "title": "Checkpoint",
                    "locator": "https://example.test/one", "excerpt": "A checkpoint preserves state."}],
                "final_answer": "A checkpoint preserves state [S2].",
            }), encoding="utf-8")
            run.joinpath("trace.jsonl").write_text(json.dumps({
                "run_id": "run-2", "step": 4, "event": "run_completed", "detail": {},
            }) + "\n", encoding="utf-8")
            report = inspect_run(run)
        self.assertFalse(report["structural_pass"])
        self.assertFalse(report["checks"]["all_citations_admitted"])


if __name__ == "__main__":
    unittest.main()
