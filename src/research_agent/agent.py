from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from .model import Planner
from .prompts import SYSTEM_PROMPT, state_prompt
from .tools import ResearchTools
from .types import Action, AgentState, Event, Evidence


class ResearchAgent:
    def __init__(self, planner: Planner, tools: ResearchTools, run_dir: Path, max_steps: int = 12, retries: int = 2):
        self.planner, self.tools = planner, tools
        self.run_dir, self.max_steps, self.retries = run_dir, max_steps, retries
        run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = run_dir / "trace.jsonl"
        self.checkpoint_path = run_dir / "checkpoint.json"
        self._documents: dict[str, str] = {}

    def run(self, goal: str, resume: bool = False) -> AgentState:
        state = self._load() if resume and self.checkpoint_path.exists() else AgentState(goal, uuid.uuid4().hex[:12])
        self._event(state, "run_started", {"goal": state.goal, "resume": resume})
        while state.status == "running" and state.step < self.max_steps:
            state.step += 1
            try:
                action = self._retry("plan", lambda: self.planner.next_action(SYSTEM_PROMPT, state_prompt(self._view(state))), state)
                self._validate(action, state)
                self._event(state, "action_selected", {"kind": action.kind, "args": action.args, "rationale": action.rationale})
                self._execute(action, state)
            except Exception as exc:
                message = f"step {state.step}: {type(exc).__name__}: {exc}"
                state.errors.append(message)
                self._event(state, "step_error", {"error": message})
            self._save(state)
        if state.status == "running":
            state.status = "budget_exhausted"
            state.final_answer = "I could not complete the research within the step budget."
            self._event(state, "budget_exhausted", {"max_steps": self.max_steps})
            self._save(state)
        return state

    def _execute(self, action: Action, state: AgentState) -> None:
        if action.kind == "search":
            results = self._retry("search", lambda: self.tools.search(action.args["query"]), state)
            for item in results:
                source_id = f"S{len(state.search_results) + 1}"
                state.search_results[source_id] = item
            self._event(state, "search_completed", {"query": action.args["query"], "count": len(results)})
        elif action.kind == "read":
            sid = action.args["source_id"]
            doc = self._retry("read", lambda: self.tools.read(state.search_results[sid]["locator"]), state)
            self._documents[sid] = doc
            if sid not in state.read_ids:
                state.read_ids.append(sid)
            self._event(state, "source_read", {"source_id": sid, "characters": len(doc)})
        elif action.kind == "note":
            sid, excerpt = action.args["source_id"], action.args["excerpt"].strip()
            if excerpt not in self._documents.get(sid, ""):
                raise ValueError("excerpt is not present verbatim in the read source")
            item = state.search_results[sid]
            state.evidence.append(Evidence(sid, item["title"], item["locator"], excerpt))
            self._event(state, "evidence_saved", {"source_id": sid, "excerpt": excerpt})
        else:
            answer = action.args["answer"].strip()
            cited = set(re.findall(r"\[(S\d+)\]", answer))
            saved = {item.source_id for item in state.evidence}
            if state.evidence and (not cited or not cited.issubset(saved)):
                raise ValueError("every final citation must refer to saved evidence")
            state.final_answer, state.status = answer, "complete"
            self._event(state, "run_completed", {"evidence_count": len(state.evidence)})

    def _retry(self, operation: str, fn, state: AgentState):
        for attempt in range(self.retries + 1):
            try:
                return fn()
            except (TimeoutError, ConnectionError) as exc:
                self._event(state, "retry", {"operation": operation, "attempt": attempt + 1, "error": str(exc)})
                if attempt == self.retries:
                    raise
                time.sleep(0.01 * (2**attempt))

    @staticmethod
    def _validate(action: Action, state: AgentState) -> None:
        if action.kind not in {"search", "read", "note", "finish"}:
            raise ValueError(f"unknown action: {action.kind}")
        required = {"search": "query", "read": "source_id", "note": "excerpt", "finish": "answer"}[action.kind]
        if not isinstance(action.args.get(required), str) or not action.args[required].strip():
            raise ValueError(f"{action.kind} requires non-empty {required}")
        if action.kind in {"read", "note"} and action.args.get("source_id") not in state.search_results:
            raise ValueError("source_id has not been discovered")
        if action.kind == "note" and action.args["source_id"] not in state.read_ids:
            raise ValueError("source must be read before evidence is saved")

    def _view(self, state: AgentState) -> dict:
        return {
            "goal": state.goal,
            "step": state.step,
            "remaining_steps": self.max_steps - state.step,
            "sources": state.search_results,
            "read_extracts": {sid: self._documents[sid][:3000] for sid in state.read_ids if sid in self._documents},
            "evidence": [e.__dict__ for e in state.evidence],
            "recent_errors": state.errors[-3:],
        }

    def _event(self, state: AgentState, event: str, detail: dict) -> None:
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(Event(state.step, event, detail).__dict__, ensure_ascii=False) + "\n")

    def _save(self, state: AgentState) -> None:
        temp = self.checkpoint_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.checkpoint_path)

    def _load(self) -> AgentState:
        return AgentState.from_dict(json.loads(self.checkpoint_path.read_text(encoding="utf-8")))
