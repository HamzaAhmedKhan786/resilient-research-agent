from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_agent.agent import ResearchAgent
from research_agent.model import GroqPlanner, LibraPlanner, OpenAIPlanner
from research_agent.tools import HttpTools


DEFAULT_GOAL = (
    "Explain why the Tacoma Narrows Bridge collapsed in 1940 and distinguish the modern "
    "aeroelastic explanation from the commonly repeated resonance explanation. Use at least "
    "two relevant Wikipedia sources and cite every substantive claim."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one real model-in-the-loop evaluation")
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--provider", choices=["openai", "groq", "libra"], default="openai")
    parser.add_argument("--model", help="provider model ID")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--run-dir", type=Path, default=ROOT / ".runs" / "live-eval")
    parser.add_argument("--result", type=Path, default=ROOT / "evals" / "live-results.json")
    args = parser.parse_args()

    key_env = {
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "libra": "LIBRA_INTERVIEW_API_KEY",
    }[args.provider]
    api_key = os.environ.get(key_env) or getpass.getpass(f"{args.provider.title()} API key (not stored): ")
    if not api_key:
        print("No API key supplied.", file=sys.stderr)
        return 2

    started = time.perf_counter()
    defaults = {"openai": "gpt-5-mini", "groq": "openai/gpt-oss-20b", "libra": "gpt-5.6-sol"}
    model = args.model or defaults[args.provider]
    if args.provider == "groq":
        planner = GroqPlanner(model, api_key=api_key)
    elif args.provider == "libra":
        planner = LibraPlanner(model, api_key=api_key)
    else:
        planner = OpenAIPlanner(model, api_key=api_key)
    state = ResearchAgent(
        planner,
        HttpTools(),
        args.run_dir,
        max_steps=args.max_steps,
    ).run(args.goal)
    elapsed = round(time.perf_counter() - started, 2)
    cited = sorted(set(re.findall(r"\[(S\d+)\]", state.final_answer or "")))
    evidence_ids = sorted({item.source_id for item in state.evidence})
    uncited_sentences = ResearchAgent._uncited_substantive_sentences(state.final_answer or "")
    checks = {
        "completed": state.status == "complete",
        "multi_step": state.step >= 4,
        "two_sources_admitted": len(evidence_ids) >= 2,
        "all_citations_admitted": bool(cited) and set(cited).issubset(evidence_ids),
        "every_substantive_sentence_cited": not uncited_sentences,
        "requested_goal_terms_covered": (
            state.quality_checks.get("goal_terms_covered") == state.quality_checks.get("goal_terms_total")
            and bool(state.quality_checks)
        ),
        "checkpoint_written": (args.run_dir / "checkpoint.json").exists(),
        "trace_written": (args.run_dir / "trace.jsonl").exists(),
    }
    result = {
        "kind": "live_model_in_the_loop",
        "provider": args.provider,
        "goal": args.goal,
        "model": model,
        "status": state.status,
        "steps": state.step,
        "elapsed_seconds": elapsed,
        "evidence_sources": [
            {"source_id": item.source_id, "title": item.title, "locator": item.locator}
            for item in state.evidence
        ],
        "cited_source_ids": cited,
        "semantic_error_count": len(state.errors),
        "uncited_substantive_sentences": uncited_sentences,
        "quality_checks": state.quality_checks,
        "answer": state.final_answer,
        "checks": checks,
        "passed": all(checks.values()),
        "limitations": [
            "Structural checks do not independently judge factual correctness or entailment.",
            "Wikipedia and model availability make this run non-deterministic.",
            "Token usage and dollar cost are not yet captured by the minimal adapter.",
        ],
    }
    args.result.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nRaw trace (ignored by Git): {args.run_dir / 'trace.jsonl'}")
    print(f"Sanitized result: {args.result}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
