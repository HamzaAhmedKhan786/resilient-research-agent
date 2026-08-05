from __future__ import annotations

import tempfile
import unittest
import json
from io import BytesIO
from unittest.mock import patch
from pathlib import Path

from research_agent.agent import ResearchAgent
from research_agent.model import GroqPlanner, OpenAIPlanner, ProviderHTTPError, ScriptedPlanner
from research_agent.tools import FlakyTools, LocalCorpusTools
from research_agent.types import Action, AgentState


CORPUS = Path(__file__).resolve().parents[1] / "evals" / "corpus"


class AgentTests(unittest.TestCase):
    @staticmethod
    def _response(payload: dict):
        class Response(BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        return Response(json.dumps(payload).encode())

    def test_openai_request_body_is_an_object_not_double_encoded(self):
        response = {"output_text": json.dumps({
            "kind": "finish",
            "args": {"answer": "Insufficient evidence.", "query": None, "source_id": None, "excerpt": None},
            "rationale": "No evidence available.",
        })}
        with patch("urllib.request.urlopen", return_value=self._response(response)) as urlopen:
            action = OpenAIPlanner(api_key="sentinel-test-key").next_action("system", "state")
        request_body = json.loads(urlopen.call_args.args[0].data)
        self.assertIsInstance(request_body, dict)
        self.assertEqual(request_body["model"], "gpt-5-mini")
        self.assertEqual(action.kind, "finish")

    def test_groq_planner_uses_plain_text_search_without_tools(self):
        payload = {"choices": [{"message": {"content": "bridge collapse"}}]}

        class Response(BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        with patch("urllib.request.urlopen", return_value=Response(json.dumps(payload).encode())) as urlopen:
            state = "Goal and compact working state:\n" + json.dumps({
                "sources": {},
                "progress": {"searched_queries": [], "unread_source_ids": [], "read_without_evidence": [], "evidence_source_ids": []},
            })
            action = GroqPlanner(api_key="sentinel-test-key").next_action("system", state)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(request.get_header("User-agent"), "resilient-research-agent/0.1")
        request_body = json.loads(request.data)
        self.assertNotIn("tool_choice", request_body)
        self.assertNotIn("tools", request_body)
        self.assertEqual(request_body["max_completion_tokens"], 512)
        self.assertEqual(request_body["reasoning_effort"], "low")
        self.assertFalse(request_body["include_reasoning"])
        self.assertEqual(action.kind, "search")
        self.assertEqual(action.args["query"], "bridge collapse")

    def test_groq_empty_search_uses_distinct_title_fallback_without_api_call(self):
        previous = "Tacoma Narrows Bridge collapse 1940 aeroelastic"
        state = "Goal and compact working state:\n" + json.dumps({
            "goal": "Explain why the Tacoma Narrows Bridge collapsed and distinguish aeroelastic flutter from resonance.",
            "sources": {},
            "progress": {
                "searched_queries": [previous], "unread_source_ids": [], "read_without_evidence": [],
                "evidence_source_ids": [], "minimum_sources": 2, "uncovered_goal_terms": ["aeroelastic", "resonance"],
            },
        })
        with patch("urllib.request.urlopen") as urlopen:
            action = GroqPlanner(api_key="sentinel-test-key").next_action("system", state)
        urlopen.assert_not_called()
        self.assertEqual(action.kind, "search")
        self.assertEqual(action.args["query"], "Tacoma Narrows Bridge")
        self.assertNotEqual(action.args["query"], previous)

    def test_groq_planner_forces_note_after_read(self):
        state = "Goal and compact working state:\n" + json.dumps({
            "sources": {"S1": {"title": "Bridge"}},
            "progress": {
                "searched_queries": ["bridge"],
                "unread_source_ids": ["S2"],
                "read_without_evidence": ["S1"],
                "evidence_source_ids": [],
            },
        })
        self.assertEqual(GroqPlanner._forced_tool(state), "note")

    def test_groq_note_is_bound_to_pending_source_not_prior_evidence(self):
        state = "Goal and compact working state:\n" + json.dumps({
            "sources": {"S1": {"title": "First"}, "S2": {"title": "Second"}},
            "progress": {
                "searched_queries": ["bridge"], "unread_source_ids": [], "read_without_evidence": ["S2"],
                "evidence_source_ids": ["S1"], "minimum_sources": 2, "uncovered_goal_terms": ["resonance"],
            },
            "read_extracts": {"S2": "A verbatim passage about torsional flutter and resonance."},
            "evidence": [{"source_id": "S1", "title": "First", "excerpt": "old excerpt that must not be copied"}],
        })
        response = {"choices": [{"message": {"content": "A verbatim passage about torsional flutter and resonance."}}]}
        with patch("urllib.request.urlopen", return_value=self._response(response)) as urlopen:
            action = GroqPlanner(api_key="sentinel-test-key").next_action("system", state)
        request_body = json.loads(urlopen.call_args.args[0].data)
        self.assertNotIn("tools", request_body)
        self.assertNotIn("old excerpt that must not be copied", request_body["messages"][0]["content"])
        self.assertEqual(request_body["max_completion_tokens"], 512)
        self.assertEqual(action.args["source_id"], "S2")

    def test_groq_can_abandon_source_without_relevant_excerpt(self):
        state = "Goal and compact working state:\n" + json.dumps({
            "sources": {"S1": {"title": "Weak source"}},
            "progress": {
                "searched_queries": ["bridge"], "unread_source_ids": [], "read_without_evidence": ["S1"],
                "evidence_source_ids": [], "minimum_sources": 1, "uncovered_goal_terms": ["resonance"],
            },
            "read_extracts": {"S1": "This source discusses an unrelated bridge."},
            "evidence": [],
        })
        response = {"choices": [{"message": {"content": "NO_RELEVANT_EXCERPT"}}]}
        with patch("urllib.request.urlopen", return_value=self._response(response)):
            action = GroqPlanner(api_key="sentinel-test-key").next_action("system", state)
        self.assertEqual(action.kind, "skip")
        self.assertEqual(action.args["source_id"], "S1")

    def test_abandoned_source_does_not_block_next_unread_source(self):
        class TwoSources:
            def search(self, query: str):
                return [{"title": "Weak", "locator": "weak"}, {"title": "Checkpointing", "locator": "strong"}]

            def read(self, locator: str):
                return "Unrelated material." if locator == "weak" else "Checkpointing preserves completed work."

        plan = [
            Action("search", {"query": "checkpointing"}),
            Action("read", {"source_id": "S1"}),
            Action("skip", {"source_id": "S1"}, "No relevant excerpt."),
            Action("read", {"source_id": "S2"}),
            Action("note", {"source_id": "S2", "excerpt": "Checkpointing preserves completed work."}),
            Action("finish", {"answer": "Checkpointing preserves completed work [S2]."}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchAgent(ScriptedPlanner(plan), TwoSources(), Path(tmp)).run("Explain checkpointing")
            trace = (Path(tmp) / "trace.jsonl").read_text(encoding="utf-8")
        self.assertEqual(state.status, "complete")
        self.assertEqual(state.abandoned_ids, ["S1"])
        self.assertIn('"event": "source_abandoned"', trace)

    def test_groq_read_phase_uses_no_provider_request(self):
        state = "Goal and compact working state:\n" + json.dumps({
            "sources": {"S1": {"title": "First"}},
            "progress": {
                "searched_queries": ["bridge"], "unread_source_ids": ["S1"], "read_without_evidence": [],
                "evidence_source_ids": [], "minimum_sources": 2, "uncovered_goal_terms": ["aeroelastic"],
            },
        })
        with patch("urllib.request.urlopen") as urlopen:
            action = GroqPlanner(api_key="sentinel-test-key").next_action("system", state)
        urlopen.assert_not_called()
        self.assertEqual(action, Action("read", {"source_id": "S1"}, "Selected deterministically from valid unread sources."))

    def test_groq_searches_for_uncovered_term_before_reading_unrelated_results(self):
        state = "Goal and compact working state:\n" + json.dumps({
            "sources": {"S1": {"title": "Bridge"}, "S2": {"title": "Unrelated structure"}},
            "progress": {
                "searched_queries": ["bridge collapse"],
                "unread_source_ids": ["S2"],
                "read_without_evidence": [],
                "evidence_source_ids": ["S1"],
                "minimum_sources": 2,
                "uncovered_goal_terms": ["resonance"],
                "coverage_candidate_ids": [],
            },
        })
        self.assertEqual(GroqPlanner._forced_tool(state), "search")

        compact = json.loads(state.split("\n", 1)[1])
        compact["progress"]["coverage_candidate_ids"] = ["S3"]
        compact["progress"]["unread_source_ids"].append("S3")
        self.assertEqual(GroqPlanner._allowed_source_ids("read", compact), ["S3"])

    def test_groq_plain_text_adapter_completes_multistep_run(self):
        responses = [
            self._response({"choices": [{"message": {"content": "checkpoint durable state"}}]}),
            self._response({"choices": [{"message": {"content": "A checkpoint lets a process resume from durable state instead of repeating all prior work."}}]}),
            self._response({"choices": [{"message": {"content": "bounded retries transient failures"}}]}),
            self._response({"choices": [{"message": {"content": "Retries should be bounded and reserved for failures likely to be transient."}}]}),
            self._response({"choices": [{"message": {"content": "Checkpointing preserves work [S1], while retries handle transient faults [S2]."}}]}),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch("urllib.request.urlopen", side_effect=responses) as urlopen:
            state = ResearchAgent(
                GroqPlanner(api_key="sentinel-test-key", min_request_interval=0), LocalCorpusTools(CORPUS), Path(tmp)
            ).run("Compare checkpointing and retry logic for resilient agents.")
        self.assertEqual(state.status, "complete")
        self.assertEqual(state.step, 7)
        self.assertEqual(urlopen.call_count, 5)
        for request_call in urlopen.call_args_list:
            self.assertNotIn("tools", json.loads(request_call.args[0].data))

    def test_groq_planner_finishes_after_two_evidence_sources(self):
        state = "Goal and compact working state:\n" + json.dumps({
            "sources": {"S1": {}, "S2": {}, "S3": {}},
            "progress": {
                "searched_queries": ["bridge"],
                "unread_source_ids": ["S3"],
                "read_without_evidence": [],
                "evidence_source_ids": ["S1", "S2"],
            },
        })
        self.assertEqual(GroqPlanner._forced_tool(state), "finish")

    def test_groq_final_synthesis_uses_plain_text_without_tools(self):
        state = "Goal and compact working state:\n" + json.dumps({
            "sources": {"S1": {}, "S2": {}, "S9": {"title": "Unsaved"}},
            "progress": {
                "searched_queries": ["bridge"], "unread_source_ids": [], "read_without_evidence": [],
                "evidence_source_ids": ["S1", "S2"], "minimum_sources": 2, "uncovered_goal_terms": [],
            },
            "evidence": [
                {"source_id": "S1", "excerpt": "first"},
                {"source_id": "S2", "excerpt": "second"},
            ],
        })
        response = {"choices": [{"message": {"content": "Supported answer [S1] [S2]."}}]}
        with patch("urllib.request.urlopen", return_value=self._response(response)) as urlopen:
            action = GroqPlanner(api_key="sentinel-test-key").next_action("system", state)
        request_body = json.loads(urlopen.call_args.args[0].data)
        self.assertNotIn("tools", request_body)
        self.assertEqual(request_body["max_completion_tokens"], 1200)
        self.assertEqual(request_body["reasoning_effort"], "low")
        self.assertFalse(request_body["include_reasoning"])
        prompt = request_body["messages"][0]["content"]
        self.assertIn('"allowed_citation_ids":["S1","S2"]', prompt)
        self.assertNotIn("S9", prompt)
        self.assertEqual(action.kind, "finish")

    def test_nonretryable_provider_error_fails_once(self):
        class InvalidRequest:
            def __init__(self):
                self.calls = 0

            def next_action(self, system: str, state: str) -> Action:
                self.calls += 1
                raise ProviderHTTPError("OpenAI", 400, "expected object", "invalid_type")

        planner = InvalidRequest()
        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchAgent(planner, LocalCorpusTools(CORPUS), Path(tmp)).run("research")
            trace = (Path(tmp) / "trace.jsonl").read_text(encoding="utf-8")
        self.assertEqual(planner.calls, 1)
        self.assertEqual(state.status, "failed")
        self.assertIn('"event": "terminal_provider_error"', trace)

    def test_exhausted_credit_429_is_not_retried(self):
        error = ProviderHTTPError("OpenAI", 429, "no credits", "credit_balance_exhausted")
        self.assertFalse(error.retryable)

    def test_finish_requires_explicit_goal_term_coverage(self):
        tacoma_goal = (
            "Explain why the Tacoma Narrows Bridge collapsed and distinguish the modern aeroelastic explanation "
            "from the commonly repeated resonance explanation."
        )
        self.assertEqual(ResearchAgent._goal_terms(tacoma_goal), ["aeroelastic", "resonance"])
        plan = [
            Action("search", {"query": "checkpoint"}),
            Action("read", {"source_id": "S1"}),
            Action("note", {"source_id": "S1", "excerpt": "A checkpoint lets a process resume from durable state instead of repeating all prior work."}),
            Action("finish", {"answer": "Checkpointing preserves progress [S1]."}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchAgent(ScriptedPlanner(plan), LocalCorpusTools(CORPUS), Path(tmp)).run("Compare checkpointing and resonance")
        self.assertEqual(state.status, "failed")
        self.assertTrue(any("resonance" in error for error in state.errors))

    def test_relevant_extract_centers_late_goal_evidence(self):
        document = "A" * 6000 + " Resonance is not the modern explanation for the collapse. " + "B" * 6000
        extract = ResearchAgent._relevant_extract(document, ["resonance"])
        self.assertLessEqual(len(extract), 3600)
        self.assertIn("Resonance is not the modern explanation", extract)
        self.assertIn(extract, document)

    def test_evidence_verification_recovers_exact_source_typography(self):
        document = (
            "The bridge collapsed because moderate winds produced aeroelastic flutter that was "
            "self-exciting and unbounded at about 35 mph."
        )
        proposed = (
            "the bridge collapsed because moderate winds produced aeroelastic flutter that was "
            "self‑exciting   and unbounded at about 35\u00a0mph."
        )
        verified = ResearchAgent._verified_excerpt(document, proposed)
        self.assertEqual(verified, document)
        self.assertIn(verified, document)

        self.assertEqual(
            ResearchAgent._verified_excerpt(document, "The bridge collapsed because of resonance."),
            "",
        )

    def test_source_ranking_prefers_exact_goal_subject_title(self):
        state = AgentState(
            goal="Explain why the Tacoma Narrows Bridge collapsed in 1940 and distinguish aeroelasticity from resonance.",
            run_id="ranking",
            search_results={
                "S1": {"title": "Suspension bridge", "locator": "one", "snippet": "Aeroelastic resonance destroyed the Tacoma Narrows Bridge."},
                "S2": {"title": "Tacoma Narrows Bridge (1940)", "locator": "two", "snippet": "The original suspension bridge."},
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            view = ResearchAgent(ScriptedPlanner([]), LocalCorpusTools(CORPUS), Path(tmp))._view(state)
        self.assertEqual(view["progress"]["coverage_candidate_ids"][0], "S2")

    def test_note_rejects_analogous_event_then_accepts_goal_subject(self):
        class OneSource:
            def search(self, query: str):
                return [{"title": "Suspension bridges", "locator": "bridge"}]

            def read(self, locator: str):
                return (
                    "Broughton Bridge collapsed because troops induced mechanical resonance. "
                    "The Tacoma Narrows Bridge collapsed after aeroelastic instability."
                )

        plan = [
            Action("search", {"query": "Tacoma bridge"}),
            Action("read", {"source_id": "S1"}),
            Action("note", {"source_id": "S1", "excerpt": "Broughton Bridge collapsed because troops induced mechanical resonance."}),
            Action("note", {"source_id": "S1", "excerpt": "The Tacoma Narrows Bridge collapsed after aeroelastic instability."}),
            Action("finish", {"answer": "The Tacoma Narrows Bridge collapsed after aeroelastic instability [S1]."}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchAgent(ScriptedPlanner(plan), OneSource(), Path(tmp)).run(
                "Explain why the Tacoma Narrows Bridge collapsed."
            )
        self.assertEqual(state.status, "complete")
        self.assertEqual(state.validation_failures, 1)
        self.assertTrue(any("does not mention the goal subject" in error for error in state.errors))

    def test_subject_titled_source_allows_contextual_bridge_excerpt(self):
        class TacomaSource:
            def search(self, query: str):
                return [{"title": "Tacoma Narrows Bridge (1940)", "locator": "tacoma"}]

            def read(self, locator: str):
                return "The bridge collapsed because moderate winds produced aeroelastic flutter, not elementary resonance."

        excerpt = "The bridge collapsed because moderate winds produced aeroelastic flutter, not elementary resonance."
        plan = [
            Action("search", {"query": "Tacoma bridge"}),
            Action("read", {"source_id": "S1"}),
            Action("note", {"source_id": "S1", "excerpt": excerpt}),
            Action("finish", {"answer": "The collapse involved aeroelastic flutter rather than elementary resonance [S1]."}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchAgent(ScriptedPlanner(plan), TacomaSource(), Path(tmp)).run(
                "Explain why the Tacoma Narrows Bridge collapsed and distinguish aeroelastic flutter from resonance."
            )
        self.assertEqual(state.status, "complete")
        self.assertEqual(state.validation_failures, 0)

    def test_finish_requires_minimum_distinct_citations_then_recovers(self):
        plan = [
            Action("search", {"query": "checkpoint durable state"}),
            Action("read", {"source_id": "S1"}),
            Action("note", {"source_id": "S1", "excerpt": "A checkpoint lets a process resume from durable state instead of repeating all prior work."}),
            Action("search", {"query": "bounded retries transient failures"}),
            Action("read", {"source_id": "S2"}),
            Action("note", {"source_id": "S2", "excerpt": "Retries should be bounded and reserved for failures likely to be transient."}),
            Action("finish", {"answer": "Only checkpointing is discussed [S1]."}),
            Action("finish", {"answer": "Checkpointing preserves work [S1], while bounded retries handle transient faults [S2]."}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchAgent(ScriptedPlanner(plan), LocalCorpusTools(CORPUS), Path(tmp)).run(
                "Use at least two relevant sources."
            )
        self.assertEqual(state.status, "complete")
        self.assertEqual(state.step, 8)
        self.assertEqual(state.validation_failures, 1)
        self.assertTrue(any("at least 2 distinct saved sources" in error for error in state.errors))

    def test_explicit_claim_level_citation_requirement_rejects_uncited_sentence(self):
        excerpt = "A checkpoint lets a process resume from durable state instead of repeating all prior work."
        plan = [
            Action("search", {"query": "checkpoint durable state"}),
            Action("read", {"source_id": "S1"}),
            Action("note", {"source_id": "S1", "excerpt": excerpt}),
            Action("finish", {"answer": "Checkpointing preserves work [S1]. It also improves reliability."}),
            Action("finish", {"answer": "Checkpointing preserves completed work [S1]."}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchAgent(ScriptedPlanner(plan), LocalCorpusTools(CORPUS), Path(tmp)).run(
                "Explain checkpointing and cite every substantive claim."
            )
        self.assertEqual(state.status, "complete")
        self.assertEqual(state.validation_failures, 1)
        self.assertTrue(any("every substantive sentence" in error for error in state.errors))

    def test_rejects_unread_evidence_then_recovers(self):
        plan = [
            Action("search", {"query": "checkpoint"}),
            Action("note", {"source_id": "S1", "excerpt": "made up"}),
            Action("read", {"source_id": "S1"}),
            Action("note", {"source_id": "S1", "excerpt": "A checkpoint lets a process resume from durable state instead of repeating all prior work."}),
            Action("finish", {"answer": "Checkpointing preserves progress [S1]."}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchAgent(ScriptedPlanner(plan), LocalCorpusTools(CORPUS), Path(tmp)).run("research")
        self.assertEqual(state.status, "complete")
        self.assertTrue(any("must be read" in error for error in state.errors))
        self.assertEqual(state.validation_failures, 1)

    def test_step_budget_terminates(self):
        plan = [Action("search", {"query": "checkpoint"})] * 4
        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchAgent(ScriptedPlanner(plan), LocalCorpusTools(CORPUS), Path(tmp), max_steps=2).run("research")
        self.assertEqual(state.status, "budget_exhausted")

    def test_interruption_resumes_from_checkpoint(self):
        class InterruptAfterTwo:
            def __init__(self):
                self.inner = iter([
                    Action("search", {"query": "checkpoint"}),
                    Action("read", {"source_id": "S1"}),
                ])

            def next_action(self, system: str, state: str) -> Action:
                try:
                    return next(self.inner)
                except StopIteration:
                    raise KeyboardInterrupt("simulated process interruption")

        resumed_actions = [
            Action("read", {"source_id": "S1"}),
            Action("note", {"source_id": "S1", "excerpt": "A checkpoint lets a process resume from durable state instead of repeating all prior work."}),
            Action("finish", {"answer": "Checkpointing preserves prior work [S1]."}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            agent = ResearchAgent(InterruptAfterTwo(), LocalCorpusTools(CORPUS), run_dir)
            with self.assertRaises(KeyboardInterrupt):
                agent.run("research")
            checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["step"], 2)
            resumed = ResearchAgent(ScriptedPlanner(resumed_actions), LocalCorpusTools(CORPUS), run_dir).run("", resume=True)
            trace = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
        self.assertEqual(resumed.status, "complete")
        self.assertEqual(trace.count('"event": "run_started"'), 2)

    def test_repeated_planner_outage_opens_circuit(self):
        class OfflinePlanner:
            def next_action(self, system: str, state: str) -> Action:
                raise ConnectionError("planner unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchAgent(OfflinePlanner(), LocalCorpusTools(CORPUS), Path(tmp), retries=0).run("research")
            trace = (Path(tmp) / "trace.jsonl").read_text(encoding="utf-8")
        self.assertEqual(state.status, "failed")
        self.assertEqual(state.step, 0)
        self.assertIn('"event": "circuit_opened"', trace)

    def test_provider_retries_do_not_consume_logical_steps(self):
        class RateLimitedThenReady:
            def __init__(self):
                self.calls = 0

            def next_action(self, system: str, state: str) -> Action:
                self.calls += 1
                if self.calls < 3:
                    raise ProviderHTTPError("Groq", 429, "rate limited", "rate_limit", retry_after=0)
                return Action("finish", {"answer": "Insufficient evidence; no supported conclusion."})

        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchAgent(RateLimitedThenReady(), LocalCorpusTools(CORPUS), Path(tmp), retries=2).run("research")
            trace = (Path(tmp) / "trace.jsonl").read_text(encoding="utf-8")
        self.assertEqual(state.status, "complete")
        self.assertEqual(state.step, 1)
        self.assertEqual(state.provider_retries, 2)
        self.assertEqual(state.tool_retries, 0)
        self.assertEqual(trace.count('"event": "provider_retry"'), 2)

    def test_tool_retries_are_persisted(self):
        plan = [
            Action("search", {"query": "checkpoint"}),
            Action("finish", {"answer": "Insufficient evidence; no supported conclusion."}),
        ]
        tools = FlakyTools(LocalCorpusTools(CORPUS), {"search": 1})
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state = ResearchAgent(ScriptedPlanner(plan), tools, run_dir).run("research")
            checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertEqual(state.tool_retries, 1)
        self.assertEqual(checkpoint["tool_retries"], 1)

    def test_rate_limits_do_not_consume_tool_generation_retry_budget(self):
        class MixedProviderFailures:
            def __init__(self):
                self.failures = iter([
                    ProviderHTTPError("Groq", 429, "cool down", "rate_limit_exceeded", retry_after=0),
                    ProviderHTTPError("Groq", 400, "malformed tool call", "tool_use_failed", retry_after=0),
                    ProviderHTTPError("Groq", 429, "cool down again", "rate_limit_exceeded", retry_after=0),
                ])
                self.calls = 0

            def next_action(self, system: str, state: str) -> Action:
                self.calls += 1
                failure = next(self.failures, None)
                if failure:
                    raise failure
                return Action("finish", {"answer": "Insufficient evidence; no supported conclusion."})

        planner = MixedProviderFailures()
        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchAgent(planner, LocalCorpusTools(CORPUS), Path(tmp), retries=2).run("research")
            events = [json.loads(line) for line in (Path(tmp) / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        retries = [event for event in events if event["event"] == "provider_retry"]
        self.assertEqual(state.status, "complete")
        self.assertEqual(state.step, 1)
        self.assertEqual(planner.calls, 4)
        self.assertEqual(state.provider_retries, 3)
        self.assertEqual([event["detail"]["category"] for event in retries], ["rate_limit", "tool_generation", "rate_limit"])

    def test_repeated_read_is_rejected_then_run_recovers(self):
        plan = [
            Action("search", {"query": "checkpoint"}),
            Action("read", {"source_id": "S1"}),
            Action("read", {"source_id": "S1"}),
            Action("note", {"source_id": "S1", "excerpt": "A checkpoint lets a process resume from durable state instead of repeating all prior work."}),
            Action("finish", {"answer": "Checkpointing preserves progress [S1]."}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchAgent(ScriptedPlanner(plan), LocalCorpusTools(CORPUS), Path(tmp)).run("research")
        self.assertEqual(state.status, "complete")
        self.assertTrue(any("source already read" in error for error in state.errors))


if __name__ == "__main__":
    unittest.main()
