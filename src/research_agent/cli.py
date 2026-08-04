from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import ResearchAgent
from .model import OpenAIPlanner, ScriptedPlanner
from .tools import HttpTools, LocalCorpusTools
from .types import Action


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the resilient research agent")
    parser.add_argument("goal", nargs="?", help="research goal")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--run-dir", type=Path, default=Path(".runs/live"))
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--demo", action="store_true", help="offline, deterministic example")
    args = parser.parse_args()

    if args.demo:
        goal = args.goal or "Compare checkpointing and retry logic for resilient agents."
        planner = ScriptedPlanner(_demo_actions())
        tools = LocalCorpusTools(Path("evals/corpus"))
    else:
        if not args.goal and not args.resume:
            parser.error("goal is required unless --resume is used")
        goal = args.goal or ""
        planner = OpenAIPlanner(args.model)
        tools = HttpTools()

    state = ResearchAgent(planner, tools, args.run_dir, args.max_steps).run(goal, args.resume)
    print(state.final_answer)
    print(f"\nstatus={state.status} steps={state.step} trace={args.run_dir / 'trace.jsonl'}")


def _demo_actions() -> list[Action]:
    return [
        Action("search", {"query": "checkpoint durable state"}, "Find checkpoint material."),
        Action("read", {"source_id": "S1"}, "Inspect the strongest result."),
        Action("note", {"source_id": "S1", "excerpt": "A checkpoint lets a process resume from durable state instead of repeating all prior work."}),
        Action("search", {"query": "bounded retries transient failures"}, "Find complementary retry material."),
        Action("read", {"source_id": "S2"}, "Get a complementary source."),
        Action("note", {"source_id": "S2", "excerpt": "Retries should be bounded and reserved for failures likely to be transient."}),
        Action("finish", {"answer": "Checkpointing limits repeated work after interruption [S1], while bounded retries absorb transient faults without allowing endless loops [S2]. Together they address different failure modes."}),
    ]


if __name__ == "__main__":
    main()
