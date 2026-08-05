from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_agent.agent import ResearchAgent
from research_agent.model import ScriptedPlanner
from research_agent.tools import FlakyTools, LocalCorpusTools
from research_agent.types import Action


@dataclass
class Result:
    name: str
    passed: bool
    status: str
    steps: int
    checks: dict[str, bool]


class AdaptivePlanner:
    """A deterministic policy that derives each action from current state."""

    def next_action(self, system: str, state_prompt: str) -> Action:
        state = json.loads(state_prompt.split("\n", 1)[1])
        sources = state["sources"]
        read_ids = set(sources) - set(state["progress"]["unread_source_ids"])
        evidence_ids = {item["source_id"] for item in state["evidence"]}
        errors = " ".join(state["recent_errors"])
        if not sources:
            return Action("search", {"query": "checkpoint durable state"})
        if "S1" not in read_ids:
            return Action("read", {"source_id": "S1"})
        if "S1" not in evidence_ids:
            return Action("note", {"source_id": "S1", "excerpt": "A checkpoint lets a process resume from durable state instead of repeating all prior work."})
        if len(sources) == 1:
            return Action("search", {"query": "bounded retries transient failures"})
        if "S2" not in read_ids:
            return Action("read", {"source_id": "S2"})
        if "S2" not in evidence_ids:
            return Action("note", {"source_id": "S2", "excerpt": "Retries should be bounded and reserved for failures likely to be transient."})
        if "citation" not in errors:
            return Action("finish", {"answer": "Checkpointing and retries address complementary failure modes."})
        return Action("finish", {"answer": "Checkpointing preserves completed work [S1], while bounded retries handle transient faults [S2]."})


def actions(valid_citation: bool = True) -> list[Action]:
    citation = "[S1]" if valid_citation else ""
    return [
        Action("search", {"query": "checkpoint durable state"}),
        Action("read", {"source_id": "S1"}),
        Action("note", {"source_id": "S1", "excerpt": "A checkpoint lets a process resume from durable state instead of repeating all prior work."}),
        Action("finish", {"answer": f"Durable checkpoints avoid repeating completed work {citation}."}),
    ]


def irrelevant_source_actions() -> list[Action]:
    return [
        Action("search", {"query": "observability"}),
        Action("read", {"source_id": "S1"}),
        Action("skip", {"source_id": "S1"}, "No checkpointing evidence in this source."),
        Action("search", {"query": "checkpoint"}),
        Action("read", {"source_id": "S2"}),
        Action("note", {"source_id": "S2", "excerpt": "A checkpoint lets a process resume from durable state instead of repeating all prior work."}),
        Action("finish", {"answer": "Checkpointing avoids repeating completed work [S2]."}),
    ]


def run_case(
    name: str,
    plan: list[Action],
    failures: dict[str, int] | None = None,
    max_steps: int = 8,
    planner=None,
    expected_event: str | None = None,
    expected_error: str | None = None,
) -> Result:
    run_dir = ROOT / ".runs" / "evals" / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    tools = LocalCorpusTools(ROOT / "evals" / "corpus")
    wrapped = FlakyTools(tools, failures or {})
    state = ResearchAgent(planner or ScriptedPlanner(plan), wrapped, run_dir, max_steps=max_steps).run("Explain why checkpointing helps.")
    trace = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    checks = {
        "completed": state.status == "complete",
        "has_evidence": bool(state.evidence),
        "trace_complete": any(e["event"] == "run_completed" for e in trace),
        "checkpoint_written": (run_dir / "checkpoint.json").exists(),
    }
    if failures:
        checks["retry_observed"] = any(e["event"] == "retry" for e in trace)
    if expected_event:
        checks[f"{expected_event}_observed"] = any(e["event"] == expected_event for e in trace)
    if expected_error:
        checks["expected_validation_error_observed"] = any(expected_error in error for error in state.errors)
    return Result(name, all(checks.values()), state.status, state.step, checks)


def main() -> int:
    results = [
        run_case("happy_path", actions()),
        run_case("transient_search_failure", actions(), {"search": 1}),
        run_case("transient_read_failure", actions(), {"read": 2}),
        run_case(
            "invalid_citation_recovery",
            actions(False) + [Action("finish", {"answer": "Durable checkpoints avoid repeated work [S1]."})],
            expected_error="every final citation",
        ),
        run_case(
            "adaptive_long_horizon", [], {"search": 1, "read": 1}, max_steps=10,
            planner=AdaptivePlanner(), expected_error="every final citation",
        ),
        run_case("irrelevant_source_recovery", irrelevant_source_actions(), expected_event="source_abandoned"),
    ]
    output = {"summary": {"passed": sum(r.passed for r in results), "total": len(results)}, "results": [asdict(r) for r in results]}
    result_path = ROOT / "evals" / "results.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
