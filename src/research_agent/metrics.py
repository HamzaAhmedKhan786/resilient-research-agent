"""Aggregate, content-free Prometheus metrics from durable UI checkpoints."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

STATUSES = ("running", "complete", "failed", "budget_exhausted")
COUNTERS = ("step", "provider_retries", "tool_retries", "validation_failures")
METRIC_NAMES = {
    "step": "research_agent_recorded_steps",
    "provider_retries": "research_agent_recorded_provider_retries",
    "tool_retries": "research_agent_recorded_tool_retries",
    "validation_failures": "research_agent_recorded_validation_failures",
}


def render_metrics(runs_root: Path) -> str:
    statuses: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    malformed = 0
    for path in runs_root.glob("*/checkpoint.json"):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or state.get("status") not in STATUSES:
                raise ValueError("invalid checkpoint status")
            values = {key: state.get(key, 0) for key in COUNTERS}
            if any(type(value) is not int or value < 0 for value in values.values()):
                raise ValueError("invalid checkpoint counter")
        except (OSError, ValueError):
            malformed += 1
            continue
        statuses[state["status"]] += 1
        totals.update(values)

    lines = [
        "# HELP research_agent_run_checkpoints Saved UI runs by latest checkpoint status.",
        "# TYPE research_agent_run_checkpoints gauge",
    ]
    lines.extend(f'research_agent_run_checkpoints{{status="{status}"}} {statuses[status]}' for status in STATUSES)
    for key in COUNTERS:
        name = METRIC_NAMES[key]
        lines.extend((
            f"# HELP {name} Sum of {key} values across saved UI checkpoints.",
            f"# TYPE {name} gauge",
            f"{name} {totals[key]}",
        ))
    lines.extend((
        "# HELP research_agent_malformed_checkpoints Checkpoints skipped due to invalid data.",
        "# TYPE research_agent_malformed_checkpoints gauge",
        f"research_agent_malformed_checkpoints {malformed}",
    ))
    return "\n".join(lines) + "\n"
