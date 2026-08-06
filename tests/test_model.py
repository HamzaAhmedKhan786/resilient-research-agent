from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from research_agent.model import GroqPlanner, LibraPlanner, ResponsesPlanner


class ModelHelperTests(unittest.TestCase):
    def test_libra_uses_company_responses_endpoint_and_model(self):
        planner = LibraPlanner(api_key="sentinel-company-key")
        self.assertEqual(planner.provider, "Libra")
        self.assertEqual(planner.model, "gpt-5.6-sol")
        self.assertEqual(
            planner.endpoint,
            "https://libra-ai-interviews.services.ai.azure.com/"
            "api/projects/proj-default/openai/v1/responses",
        )

    def test_libra_reads_only_its_dedicated_environment_key(self):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "wrong-openai-key",
                "GROQ_API_KEY": "wrong-groq-key",
                "LIBRA_INTERVIEW_API_KEY": "expected-libra-key",
            },
            clear=True,
        ):
            planner = LibraPlanner()
        self.assertEqual(planner.api_key, "expected-libra-key")

    def test_libra_request_uses_company_url_and_bearer_key(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "output_text": json.dumps({
                "kind": "search",
                "args": {"query": "bridge", "source_id": None, "excerpt": None, "answer": None},
                "rationale": "find sources",
            })
        }).encode()
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            action = LibraPlanner(api_key="sentinel-company-key").next_action("system", "state")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, LibraPlanner.ENDPOINT)
        self.assertEqual(request.get_header("Authorization"), "Bearer sentinel-company-key")
        self.assertEqual(action.kind, "search")

    def test_groq_request_pacing_waits_between_calls(self):
        planner = GroqPlanner(api_key="sentinel-test-key", min_request_interval=0.75)
        planner._last_request_at = 10.0
        with patch("research_agent.model.time.monotonic", return_value=10.2), patch(
            "research_agent.model.time.sleep"
        ) as sleep:
            with patch.object(ResponsesPlanner, "_post_json", return_value={}):
                planner._post_json({})
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 0.55)
        self.assertAlmostEqual(planner._last_request_at, 10.75)

    def test_search_query_is_bounded_to_six_meaningful_terms(self):
        query = GroqPlanner._clean_query(
            "Tacoma Narrows Bridge collapse 1940 aeroelastic flutter versus resonance Wikipedia articles"
        )
        self.assertEqual(query, "Tacoma Narrows Bridge collapse 1940 aeroelastic")

    def test_fallback_query_never_repeats_prior_search(self):
        compact = {
            "goal": "Explain the Tacoma Narrows Bridge collapse and distinguish aeroelastic flutter from resonance.",
            "progress": {"uncovered_goal_terms": ["aeroelastic", "resonance"]},
        }
        first = GroqPlanner._fallback_query(compact, [])
        second = GroqPlanner._fallback_query(compact, [first])
        self.assertEqual(first, "Tacoma Narrows Bridge aeroelastic resonance")
        self.assertEqual(second, "Tacoma Narrows Bridge collapse aeroelastic flutter")

    def test_followup_search_cannot_omit_uncovered_goal_concept(self):
        planner = GroqPlanner(api_key="sentinel-test-key", min_request_interval=0)
        compact = {
            "goal": "Explain why the Tacoma Narrows Bridge collapsed and distinguish aeroelastic flutter from resonance.",
            "sources": {"S1": {"title": "Tacoma Narrows Bridge"}},
            "progress": {
                "searched_queries": ["Tacoma Narrows Bridge collapse aeroelastic"],
                "evidence_source_ids": ["S1"],
                "subject_terms": ["tacoma", "narrows", "bridge"],
                "uncovered_goal_terms": ["resonance"],
            },
        }
        response = {"choices": [{"message": {"content": "Tacoma Narrows Bridge"}}]}
        with patch.object(planner, "_post_json", return_value=response):
            action = planner._search(compact)
        self.assertEqual(action.args["query"], "tacoma narrows bridge resonance")

    def test_empty_chat_content_reports_finish_reason(self):
        with self.assertRaisesRegex(ValueError, "finish_reason=length"):
            GroqPlanner._chat_text({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})


if __name__ == "__main__":
    unittest.main()
