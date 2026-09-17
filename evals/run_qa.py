"""Run deterministic QA, with optional DeepEval judge and Langfuse summary trace."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_agent.qa import inspect_run, judge_with_deepeval, publish_qa_to_langfuse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="directory containing checkpoint.json and trace.jsonl")
    parser.add_argument("--judge", action="store_true", help="use DeepEval; sends goal, answer, and evidence to the judge model")
    parser.add_argument("--judge-model", help="optional DeepEval model ID")
    parser.add_argument("--show-reasons", action="store_true", help="print judge explanations, which may quote run content")
    parser.add_argument("--langfuse", action="store_true", help="publish content-free QA scores to Langfuse")
    args = parser.parse_args()
    if args.judge and not os.environ.get("OPENAI_API_KEY"):
        parser.error("--judge requires OPENAI_API_KEY for the default DeepEval judge")
    if args.langfuse and not all(os.environ.get(name) for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")):
        parser.error("--langfuse requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")

    try:
        report = inspect_run(args.run_dir)
        if args.judge and report["structural_pass"]:
            # DeepEval otherwise auto-loads .env files from the working directory.
            os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
            report["judge"] = judge_with_deepeval(report, args.judge_model)
        if args.langfuse:
            publish_qa_to_langfuse(report)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"QA could not run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    # Avoid writing the goal, answer, or evidence to stdout or a second result file.
    summary = {key: report[key] for key in ("run_id", "status", "checks", "structural_pass")}
    if "judge" in report:
        summary["judge"] = {
            name: verdict if args.show_reasons else {"score": verdict["score"], "passed": verdict["passed"]}
            for name, verdict in report["judge"].items()
        }
    print(json.dumps(summary, indent=2))
    return 0 if report["structural_pass"] and all(
        item["passed"] for item in report.get("judge", {}).values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
