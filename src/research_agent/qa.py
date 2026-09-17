"""Independent, read-only QA of one saved agent run."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .agent import ResearchAgent
from .types import AgentState


def inspect_run(run_dir: Path) -> dict:
    """Check trace integrity and grounding structure without calling a model."""
    state = AgentState.from_dict(json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8")))
    events = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    answer = state.final_answer or ""
    cited = set(re.findall(r"\[(S\d+)\]", answer))
    admitted = {item.source_id for item in state.evidence}
    minimum = ResearchAgent._minimum_sources(state.goal)
    checks = {
        "completed": state.status == "complete",
        "trace_run_ids_match": bool(events) and all(event.get("run_id") == state.run_id for event in events),
        "trace_records_completion": any(event.get("event") == "run_completed" for event in events),
        "answer_present": bool(answer.strip()),
        "enough_sources": len(admitted) >= minimum,
        "all_citations_admitted": bool(cited) and cited.issubset(admitted),
        "cites_required_sources": len(cited) >= minimum,
        "evidence_excerpts_present": bool(state.evidence) and all(item.excerpt.strip() for item in state.evidence),
    }
    if "cite every substantive claim" in state.goal.lower():
        checks["all_substantive_sentences_cited"] = not ResearchAgent._uncited_substantive_sentences(answer)
    return {
        "run_id": state.run_id,
        "status": state.status,
        "checks": checks,
        "structural_pass": all(checks.values()),
        "goal": state.goal,
        "answer": answer,
        "evidence_excerpts": [item.excerpt for item in state.evidence],
    }


def judge_with_deepeval(report: dict, model: str | None = None) -> dict:
    """Optional LLM judge; sends goal, answer, and excerpts to the configured judge."""
    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise RuntimeError("Install optional QA dependencies: pip install -r requirements-qa.txt") from exc

    case = LLMTestCase(
        input=report["goal"],
        actual_output=report["answer"],
        retrieval_context=report["evidence_excerpts"],
    )
    options = {"threshold": 0.7, "include_reason": True}
    if model:
        options["model"] = model
    metrics = {
        "faithfulness": FaithfulnessMetric(**options),
        "answer_relevancy": AnswerRelevancyMetric(**options),
    }
    verdicts = {}
    for name, metric in metrics.items():
        metric.measure(case)
        verdicts[name] = {
            "score": metric.score,
            "passed": metric.is_successful(),
            "reason": metric.reason,
        }
    return verdicts


def publish_qa_to_langfuse(report: dict) -> None:
    """Opt-in QA trace containing only IDs, check names, and scores—not user text."""
    try:
        from langfuse import get_client
    except ImportError as exc:
        raise RuntimeError("Install optional QA dependencies: pip install -r requirements-qa.txt") from exc

    client = get_client()
    with client.start_as_current_observation(as_type="span", name="research-agent-qa") as span:
        span.update(
            input={"run_id": report["run_id"]},
            output={
                "structural_pass": report["structural_pass"],
                "checks": report["checks"],
                "judge_scores": {
                    name: verdict["score"] for name, verdict in report.get("judge", {}).items()
                },
            },
        )
    client.flush()
