from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class Action:
    """A model-selected next action. Kept intentionally tiny and inspectable."""

    kind: Literal["search", "read", "note", "skip", "finish"]
    args: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class Evidence:
    source_id: str
    title: str
    locator: str
    excerpt: str


@dataclass
class Event:
    step: int
    event: str
    detail: dict[str, Any]


@dataclass
class AgentState:
    goal: str
    run_id: str
    step: int = 0
    status: Literal["running", "complete", "budget_exhausted", "failed"] = "running"
    search_results: dict[str, dict[str, str]] = field(default_factory=dict)
    search_queries: list[str] = field(default_factory=list)
    read_ids: list[str] = field(default_factory=list)
    abandoned_ids: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    provider_retries: int = 0
    tool_retries: int = 0
    validation_failures: int = 0
    final_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AgentState":
        raw = dict(raw)
        raw["evidence"] = [Evidence(**item) for item in raw.get("evidence", [])]
        return cls(**raw)
