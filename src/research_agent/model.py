from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol

from .types import Action


class Planner(Protocol):
    def next_action(self, system: str, state: str) -> Action: ...


class OpenAIPlanner:
    """Minimal Responses API client; avoids hiding the agent loop in a framework."""

    def __init__(self, model: str = "gpt-5-mini", timeout: int = 60, api_key: str | None = None):
        self.model = model
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for a live run")

    def next_action(self, system: str, state: str) -> Action:
        payload = json.dumps({
            "model": self.model,
            "instructions": system,
            "input": state,
            "text": {"format": {
                "type": "json_schema",
                "name": "research_action",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["search", "read", "note", "finish"]},
                        "args": {"type": "object", "additionalProperties": {"type": "string"}},
                        "rationale": {"type": "string"},
                    },
                    "required": ["kind", "args", "rationale"],
                    "additionalProperties": False,
                },
            }},
        }).encode()
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"model HTTP {exc.code}: {detail[:300]}") from exc
        text = _response_text(body)
        raw = json.loads(text)
        return Action(kind=raw["kind"], args=raw.get("args", {}), rationale=raw.get("rationale", ""))


def _response_text(body: dict) -> str:
    if body.get("output_text"):
        return body["output_text"]
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content["text"]
    raise ValueError("model response contained no output text")


class ScriptedPlanner:
    """Deterministic planner used by evals and the checked-in example run."""

    def __init__(self, actions: list[Action]):
        self.actions = iter(actions)

    def next_action(self, system: str, state: str) -> Action:
        try:
            return next(self.actions)
        except StopIteration as exc:
            raise RuntimeError("scripted planner ran out of actions") from exc
