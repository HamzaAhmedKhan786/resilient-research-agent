from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_agent.agent import ResearchAgent
from research_agent.model import ScriptedPlanner
from research_agent.tools import LocalCorpusTools
from research_agent.types import Action


CORPUS = Path(__file__).resolve().parents[1] / "evals" / "corpus"


class AgentTests(unittest.TestCase):
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

    def test_step_budget_terminates(self):
        plan = [Action("search", {"query": "checkpoint"})] * 4
        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchAgent(ScriptedPlanner(plan), LocalCorpusTools(CORPUS), Path(tmp), max_steps=2).run("research")
        self.assertEqual(state.status, "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
