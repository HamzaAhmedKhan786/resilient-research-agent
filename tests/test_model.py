from __future__ import annotations

import unittest
from unittest.mock import patch

from research_agent.model import GroqPlanner, ResponsesPlanner


class ModelHelperTests(unittest.TestCase):
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
        self.assertEqual(first, "Tacoma Narrows Bridge")
        self.assertEqual(second, "aeroelastic resonance")

    def test_empty_chat_content_reports_finish_reason(self):
        with self.assertRaisesRegex(ValueError, "finish_reason=length"):
            GroqPlanner._chat_text({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})


if __name__ == "__main__":
    unittest.main()
