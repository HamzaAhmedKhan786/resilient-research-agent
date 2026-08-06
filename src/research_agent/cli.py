from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import ResearchAgent
from .model import GroqPlanner, LibraPlanner, OpenAIPlanner, ScriptedPlanner
from .tools import HttpTools, LocalCorpusTools
from .types import Action


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the resilient research agent")
    parser.add_argument("goal", nargs="?", help="research goal")
    parser.add_argument("--provider", choices=["openai", "groq", "libra"], default="openai")
    parser.add_argument("--model", help="provider model ID")
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
        if args.provider == "groq":
            planner = GroqPlanner(args.model or "openai/gpt-oss-20b")
        elif args.provider == "libra":
            planner = LibraPlanner(args.model or "gpt-5.6-sol")
        else:
            planner = OpenAIPlanner(args.model or "gpt-5-mini")
        tools = HttpTools()

    state = ResearchAgent(planner, tools, args.run_dir, args.max_steps).run(goal, args.resume)
    print(state.final_answer)
    print(
        f"\nstatus={state.status} steps={state.step} "
        f"provider_retries={state.provider_retries} "
        f"tool_retries={state.tool_retries} "
        f"validation_failures={state.validation_failures} "
        f"quality_checks={json.dumps(state.quality_checks, separators=(',', ':'))} "
        f"trace={args.run_dir / 'trace.jsonl'}"
    )


def _demo_actions() -> list[Action]:
    return [
        Action("search", {"query": "checkpoint durable state"}, "Find checkpoint material."),
        Action("read", {"source_id": "S1"}, "Inspect the strongest result."),
        Action("note", {"source_id": "S1", "excerpt": "A checkpoint lets a process resume from durable state instead of repeating all prior work."}),
        Action("search", {"query": "bounded retries transient failures"}, "Find complementary retry material."),
        Action("read", {"source_id": "S2"}, "Get a complementary source."),
        Action("note", {"source_id": "S2", "excerpt": "Retries should be bounded and reserved for failures likely to be transient."}),
        Action("finish", {"answer": "Checkpointing limits repeated work after interruption [S1], while bounded retries absorb transient faults without allowing endless loops [S2]. Together they address different failure modes [S1][S2]."}),
    ]


if __name__ == "__main__":
    main()
