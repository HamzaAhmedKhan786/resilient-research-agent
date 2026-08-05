from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.error
import uuid
from pathlib import Path

from .model import Planner, ProviderHTTPError
from .prompts import SYSTEM_PROMPT, state_prompt
from .tools import ResearchTools
from .types import Action, AgentState, Event, Evidence


class ResearchAgent:
    MAX_SEARCHES = 4
    MAX_RATE_LIMIT_RETRIES = 4
    MAX_RATE_LIMIT_WAIT = 45.0
    GOAL_STOPWORDS = {
        "about", "after", "against", "already", "also", "and", "answer", "are", "claim", "claims",
        "cite", "cites", "commonly", "compare", "comparison", "distinguish", "evaluate", "every", "explain",
        "explanation", "finish", "from", "give", "helps", "important", "into", "least", "modern", "provide",
        "logic", "relevant", "repeated", "report", "research", "resilient", "source", "sources", "substantive", "that", "their",
        "them", "these", "this", "uncertainties", "uncertainty", "use", "using", "what", "when", "where",
        "which", "why", "with", "wikipedia", "would", "your",
    }

    def __init__(self, planner: Planner, tools: ResearchTools, run_dir: Path, max_steps: int = 12, retries: int = 2):
        self.planner, self.tools = planner, tools
        self.run_dir, self.max_steps, self.retries = run_dir, max_steps, retries
        run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = run_dir / "trace.jsonl"
        self.checkpoint_path = run_dir / "checkpoint.json"
        self._documents: dict[str, str] = {}

    def run(self, goal: str, resume: bool = False) -> AgentState:
        state = self._load() if resume and self.checkpoint_path.exists() else AgentState(goal, uuid.uuid4().hex[:12])
        previous_error: str | None = None
        repeated_errors = 0
        self._event(state, "run_started", {"goal": state.goal, "resume": resume})
        if resume:
            self._restore_documents(state)
        while state.status == "running" and state.step < self.max_steps:
            try:
                action = self._retry("plan", lambda: self.planner.next_action(SYSTEM_PROMPT, state_prompt(self._view(state))), state)
                state.step += 1
                self._validate(action, state)
                self._event(state, "action_selected", {"kind": action.kind, "args": action.args, "rationale": action.rationale})
                self._execute(action, state)
                previous_error, repeated_errors = None, 0
            except Exception as exc:
                message = f"step {state.step}: {type(exc).__name__}: {exc}"
                state.errors.append(message)
                self._event(state, "step_error", {"error": message})
                if isinstance(exc, ValueError) and not isinstance(exc, ProviderHTTPError):
                    state.validation_failures += 1
                fingerprint = f"{type(exc).__name__}: {exc}"
                repeated_errors = repeated_errors + 1 if fingerprint == previous_error else 1
                previous_error = fingerprint
                if isinstance(exc, ProviderHTTPError):
                    state.status = "failed"
                    state.final_answer = f"The {exc.provider} request failed after bounded retries: {exc}."
                    self._event(state, "terminal_provider_error", {"provider": exc.provider, "status": exc.status, "error": str(exc)})
                elif repeated_errors >= 3:
                    state.status = "failed"
                    state.final_answer = "The run stopped after the same unrecoverable error occurred three times. Check the trace and external-service connectivity, then retry."
                    self._event(state, "circuit_opened", {"error": fingerprint, "occurrences": repeated_errors})
            self._save(state)
        if state.status == "running":
            state.status = "budget_exhausted"
            state.final_answer = "I could not complete the research within the step budget."
            self._event(state, "budget_exhausted", {"max_steps": self.max_steps})
            self._save(state)
        return state

    def _execute(self, action: Action, state: AgentState) -> None:
        if action.kind == "search":
            query = action.args["query"].strip()
            if len(state.search_queries) >= self.MAX_SEARCHES:
                raise ValueError("search budget reached; read discovered sources and save evidence instead")
            normalized = self._normalize_query(query)
            if normalized in {self._normalize_query(item) for item in state.search_queries}:
                raise ValueError("search query already completed; read an existing source or use a materially different query")
            results = self._retry("search", lambda: self.tools.search(query), state)
            state.search_queries.append(query)
            known_locators = {item["locator"] for item in state.search_results.values()}
            added = 0
            for item in results:
                if item["locator"] in known_locators:
                    continue
                source_id = f"S{len(state.search_results) + 1}"
                state.search_results[source_id] = item
                known_locators.add(item["locator"])
                added += 1
            self._event(state, "search_completed", {"query": query, "returned": len(results), "new_sources": added})
        elif action.kind == "read":
            sid = action.args["source_id"]
            if sid in state.read_ids and sid in self._documents:
                raise ValueError("source already read; save evidence from it or choose an unread source")
            doc = self._retry("read", lambda: self.tools.read(state.search_results[sid]["locator"]), state)
            self._documents[sid] = doc
            if sid not in state.read_ids:
                state.read_ids.append(sid)
            self._event(state, "source_read", {"source_id": sid, "characters": len(doc)})
        elif action.kind == "note":
            sid, excerpt = action.args["source_id"], action.args["excerpt"].strip()
            excerpt = self._verified_excerpt(self._documents.get(sid, ""), excerpt)
            if not excerpt:
                raise ValueError("excerpt is not present verbatim in the read source")
            item = state.search_results[sid]
            subject_terms = self._subject_terms(state.goal)
            identity_terms = subject_terms[:2]
            excerpt_stems = self._token_stems(excerpt)
            title_stems = self._token_stems(item.get("title", ""))
            title_identifies_subject = all(self._stem(term) in title_stems for term in identity_terms)
            excerpt_names_subject = bool({self._stem(term) for term in identity_terms}.intersection(excerpt_stems))
            if identity_terms and not (title_identifies_subject or excerpt_names_subject):
                raise ValueError(
                    "evidence excerpt does not mention the goal subject; choose direct evidence or abandon the source"
                )
            if any(saved.source_id == sid and saved.excerpt == excerpt for saved in state.evidence):
                raise ValueError("this evidence excerpt is already saved")
            state.evidence.append(Evidence(sid, item["title"], item["locator"], excerpt))
            self._event(state, "evidence_saved", {"source_id": sid, "excerpt": excerpt})
        elif action.kind == "skip":
            sid = action.args["source_id"]
            if sid not in state.abandoned_ids:
                state.abandoned_ids.append(sid)
            self._documents.pop(sid, None)
            self._event(state, "source_abandoned", {"source_id": sid, "reason": action.rationale or "no relevant verbatim evidence"})
        else:
            answer = action.args["answer"].strip()
            cited = set(re.findall(r"\[(S\d+)\]", answer))
            saved = {item.source_id for item in state.evidence}
            if state.evidence and (not cited or not cited.issubset(saved)):
                raise ValueError("every final citation must refer to saved evidence")
            minimum_sources = self._minimum_sources(state.goal)
            admits_insufficient_evidence = not state.evidence and "insufficient evidence" in answer.lower()
            if len(saved) < minimum_sources and not admits_insufficient_evidence:
                raise ValueError(f"the goal requires evidence from at least {minimum_sources} sources")
            if state.evidence and len(cited) < minimum_sources:
                raise ValueError(f"the final answer must cite at least {minimum_sources} distinct saved sources")
            if "cite every substantive claim" in state.goal.lower():
                uncited = self._uncited_substantive_sentences(answer)
                if uncited:
                    raise ValueError(
                        "every substantive sentence must contain a saved-evidence citation; "
                        f"uncited sentence: {uncited[0]}"
                    )
            uncovered = self._uncovered_goal_terms(state)
            if state.evidence and uncovered:
                raise ValueError(f"saved evidence does not cover these goal terms: {', '.join(uncovered)}")
            state.final_answer, state.status = answer, "complete"
            self._event(state, "run_completed", {"evidence_count": len(state.evidence)})

    def _retry(self, operation: str, fn, state: AgentState):
        failure_counts: dict[str, int] = {}
        rate_limit_wait = 0.0
        while True:
            try:
                return fn()
            except ProviderHTTPError as exc:
                if not exc.retryable:
                    raise
                category = "rate_limit" if exc.status == 429 else "tool_generation" if exc.status == 400 else "server"
                failure_counts[category] = failure_counts.get(category, 0) + 1
                limit = self.MAX_RATE_LIMIT_RETRIES if category == "rate_limit" else self.retries
                delay = min(15.0, max(0.1, exc.retry_after))
                if failure_counts[category] > limit or (category == "rate_limit" and rate_limit_wait + delay > self.MAX_RATE_LIMIT_WAIT):
                    raise
                if category == "rate_limit":
                    rate_limit_wait += delay
                state.provider_retries += 1
                self._event(state, "provider_retry", {
                    "provider": exc.provider,
                    "status": exc.status,
                    "operation": operation,
                    "category": category,
                    "attempt": failure_counts[category],
                    "wait_seconds": delay,
                })
                time.sleep(delay)
            except (TimeoutError, ConnectionError, urllib.error.URLError) as exc:
                category = "transport"
                failure_counts[category] = failure_counts.get(category, 0) + 1
                if failure_counts[category] > self.retries:
                    raise
                delay = getattr(exc, "retry_after", 0.01 * (2 ** (failure_counts[category] - 1)))
                state.tool_retries += 1
                self._event(state, "retry", {"operation": operation, "attempt": failure_counts[category], "error": str(exc)})
                time.sleep(min(10.0, delay))

    def _restore_documents(self, state: AgentState) -> None:
        for source_id in state.read_ids:
            if source_id in state.abandoned_ids:
                continue
            item = state.search_results.get(source_id)
            if not item:
                continue
            try:
                document = self._retry("restore_read", lambda: self.tools.read(item["locator"]), state)
                self._documents[source_id] = document
                self._event(state, "source_restored", {"source_id": source_id, "characters": len(document)})
            except Exception as exc:
                self._event(state, "restore_error", {"source_id": source_id, "error": f"{type(exc).__name__}: {exc}"})

    @staticmethod
    def _validate(action: Action, state: AgentState) -> None:
        if action.kind not in {"search", "read", "note", "skip", "finish"}:
            raise ValueError(f"unknown action: {action.kind}")
        required = {"search": "query", "read": "source_id", "note": "excerpt", "skip": "source_id", "finish": "answer"}[action.kind]
        if not isinstance(action.args.get(required), str) or not action.args[required].strip():
            raise ValueError(f"{action.kind} requires non-empty {required}")
        if action.kind in {"read", "note", "skip"} and action.args.get("source_id") not in state.search_results:
            raise ValueError("source_id has not been discovered")
        if action.kind in {"note", "skip"} and action.args["source_id"] not in state.read_ids:
            raise ValueError("source must be read before it can be noted or abandoned")

    def _view(self, state: AgentState) -> dict:
        evidence_ids = {item.source_id for item in state.evidence}
        read_without_evidence = [sid for sid in state.read_ids if sid not in evidence_ids and sid not in state.abandoned_ids]
        uncovered = self._uncovered_goal_terms(state)
        uncovered_stems = {self._stem(term) for term in uncovered}
        subject_terms = self._subject_terms(state.goal)
        subject_stems = {self._stem(term) for term in subject_terms}
        goal_stems = self._token_stems(state.goal)
        scored_candidates = []
        for sid, item in state.search_results.items():
            if sid in state.read_ids:
                continue
            title_stems = self._token_stems(item.get("title", ""))
            snippet_stems = self._token_stems(item.get("snippet", ""))
            searchable_stems = title_stems | snippet_stems
            score = (
                5 * len(subject_stems.intersection(title_stems))
                + len(subject_stems.intersection(snippet_stems))
                + 2 * len(goal_stems.intersection(title_stems))
                + 2 * len(uncovered_stems.intersection(searchable_stems))
            )
            if score:
                scored_candidates.append((score, sid))
        scored_candidates.sort(key=lambda item: (-item[0], item[1]))
        candidate_ids = [sid for _, sid in scored_candidates]
        return {
            "goal": state.goal,
            "step": state.step,
            "remaining_steps": self.max_steps - state.step,
            "sources": state.search_results,
            "progress": {
                "searched_queries": state.search_queries,
                "unread_source_ids": [sid for sid in state.search_results if sid not in state.read_ids],
                "read_without_evidence": read_without_evidence,
                "evidence_source_ids": sorted(evidence_ids),
                "abandoned_source_ids": state.abandoned_ids,
                "minimum_sources": self._minimum_sources(state.goal),
                "subject_terms": subject_terms,
                "uncovered_goal_terms": uncovered,
                "coverage_candidate_ids": candidate_ids,
                "next_step_hint": "Save evidence after a read. If goal terms remain uncovered, search for them or read a matching candidate.",
            },
            "read_extracts": {
                sid: self._relevant_extract(
                    self._documents[sid],
                    [*subject_terms, *subject_terms, *(uncovered or self._goal_terms(state.goal))],
                )
                for sid in read_without_evidence
                if sid in self._documents
            },
            "evidence": [e.__dict__ for e in state.evidence],
            "recent_errors": state.errors[-3:],
        }

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", query.lower()))

    @staticmethod
    def _verified_excerpt(document: str, proposed: str) -> str:
        """Match harmless typography changes, then return the exact source span."""
        if proposed in document:
            return proposed

        def canonical_with_map(value: str) -> tuple[str, list[int]]:
            canonical: list[str] = []
            source_indexes: list[int] = []
            translations = {
                "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
                "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            }
            for index, character in enumerate(value):
                normalized = unicodedata.normalize("NFKC", character).lower()
                for item in normalized:
                    item = translations.get(item, item)
                    if item.isspace():
                        if canonical and canonical[-1] != " ":
                            canonical.append(" ")
                            source_indexes.append(index)
                    else:
                        canonical.append(item)
                        source_indexes.append(index)
            return "".join(canonical), source_indexes

        canonical_document, indexes = canonical_with_map(document)
        canonical_proposed, _ = canonical_with_map(proposed)
        canonical_proposed = canonical_proposed.strip()
        if not canonical_proposed:
            return ""
        position = canonical_document.find(canonical_proposed)
        if position < 0:
            return ""
        start = indexes[position]
        end = indexes[position + len(canonical_proposed) - 1] + 1
        return document[start:end]

    @staticmethod
    def _relevant_extract(document: str, terms: list[str], limit: int = 3600) -> str:
        """Return one exact, bounded window centered on the densest relevant passage."""
        if len(document) <= limit:
            return document
        lowered = document.lower()
        normalized_terms = [term.lower() for term in terms if term.strip()]
        positions = [
            match.start()
            for term in normalized_terms
            for match in re.finditer(rf"\b{re.escape(term)}\w*\b", lowered)
        ]
        if not positions:
            return document[:limit]

        best: tuple[int, int, str] | None = None
        for position in positions:
            start = max(0, min(position - limit // 3, len(document) - limit))
            window = document[start:start + limit]
            score = sum(len(re.findall(rf"\b{re.escape(term)}\w*\b", window.lower())) for term in normalized_terms)
            candidate = (score, -start, window)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        return best[2] if best else document[:limit]

    @classmethod
    def _goal_terms(cls, goal: str) -> list[str]:
        lowered = goal.lower()
        segments: list[str] = []
        distinguish = re.search(r"\bdistinguish\s+(.+?)\s+from\s+(.+?)(?:[.;]|\buse\b|\bcite\b|$)", lowered)
        compare = re.search(r"\bcompare\s+(.+?)\s+(?:with|and|versus|vs\.?)+\s+(.+?)(?:[.;]|\bfor\b|\buse\b|$)", lowered)
        if distinguish:
            segments.extend(distinguish.groups())
        elif compare:
            segments.extend(compare.groups())
        else:
            segments.append(lowered)

        terms: list[str] = []
        for segment in segments:
            for word in re.findall(r"[a-z0-9]+", segment):
                if len(word) < 5 or word in cls.GOAL_STOPWORDS or word.isdigit():
                    continue
                if word not in terms:
                    terms.append(word)
        return terms[:6] if len(segments) > 1 else terms[:3]

    @classmethod
    def _subject_terms(cls, goal: str) -> list[str]:
        lowered = goal.lower()
        boundary = re.search(r"\b(?:distinguish|compare|evaluate)\b", lowered)
        prefix = lowered[:boundary.start()] if boundary else lowered
        subject = re.search(
            r"\b(?:why|how)\s+(?:the\s+)?(.+?)\s+"
            r"(?:collapsed|failed|fails|works|worked|happened|occurred)\b",
            prefix,
        )
        if not subject:
            return []
        segment = subject.group(1)
        terms = []
        for word in re.findall(r"[a-z0-9]+", segment):
            if len(word) < 5 or word in cls.GOAL_STOPWORDS or word.isdigit():
                continue
            if word not in terms:
                terms.append(word)
        return terms[:4]

    @staticmethod
    def _stem(word: str) -> str:
        if word.endswith("ies") and len(word) > 5:
            return word[:-3] + "y"
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                return word[:-len(suffix)]
        return word

    @classmethod
    def _token_stems(cls, text: str) -> set[str]:
        return {cls._stem(word) for word in re.findall(r"[a-z0-9]+", text.lower())}

    @classmethod
    def _minimum_sources(cls, goal: str) -> int:
        match = re.search(r"at least\s+(\d+|one|two|three|four)", goal.lower())
        if not match:
            return 1
        values = {"one": 1, "two": 2, "three": 3, "four": 4}
        return int(match.group(1)) if match.group(1).isdigit() else values[match.group(1)]

    @staticmethod
    def _uncited_substantive_sentences(answer: str) -> list[str]:
        """Return factual-looking sentences without an inline [S#] citation."""
        text = re.sub(r"^#{1,6}\s+", "", answer, flags=re.MULTILINE)
        text = re.sub(r"\*\*([^*]+)\**", r"\1", text)
        sentences = re.split(r"(?<=[.!?])(?:\s+|$)", text.strip())
        uncited: list[str] = []
        for sentence in sentences:
            sentence = sentence.strip().lstrip("-• ")
            words = re.findall(r"[A-Za-z0-9]+", sentence)
            if len(words) >= 4 and not re.search(r"\[S\d+\]", sentence):
                uncited.append(sentence)
        return uncited

    @classmethod
    def _uncovered_goal_terms(cls, state: AgentState) -> list[str]:
        covered_text = " ".join(f"{item.title} {item.excerpt}" for item in state.evidence)
        covered = cls._token_stems(covered_text)
        return [term for term in cls._goal_terms(state.goal) if cls._stem(term) not in covered]

    def _event(self, state: AgentState, event: str, detail: dict) -> None:
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(Event(state.step, event, detail).__dict__, ensure_ascii=False) + "\n")

    def _save(self, state: AgentState) -> None:
        temp = self.checkpoint_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        for attempt in range(5):
            try:
                temp.replace(self.checkpoint_path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01 * (2**attempt))

    def _load(self) -> AgentState:
        return AgentState.from_dict(json.loads(self.checkpoint_path.read_text(encoding="utf-8")))
