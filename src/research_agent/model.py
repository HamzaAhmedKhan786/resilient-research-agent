from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Protocol

from .types import Action


ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["search", "read", "note", "skip", "finish"]},
        "args": {
            "type": "object",
            "properties": {
                "query": {"type": ["string", "null"]},
                "source_id": {"type": ["string", "null"]},
                "excerpt": {"type": ["string", "null"]},
                "answer": {"type": ["string", "null"]},
            },
            "required": ["query", "source_id", "excerpt", "answer"],
            "additionalProperties": False,
        },
        "rationale": {"type": "string"},
    },
    "required": ["kind", "args", "rationale"],
    "additionalProperties": False,
}


class ProviderHTTPError(RuntimeError):
    def __init__(self, provider: str, status: int, message: str, code: str = "", retry_after: float = 1.0):
        self.provider = provider
        self.status = status
        self.retry_after = retry_after
        permanent_quota_codes = {"credit_balance_exhausted", "insufficient_quota"}
        self.retryable = (status == 429 and code not in permanent_quota_codes) or 500 <= status < 600 or code == "tool_use_failed"
        label = f"{code}: " if code else ""
        super().__init__(f"{provider} HTTP {status}: {label}{message}")


class Planner(Protocol):
    def next_action(self, system: str, state: str) -> Action: ...


class ResponsesPlanner:
    """Minimal provider-compatible Responses API client."""

    def __init__(self, model: str, api_key: str, endpoint: str, provider: str, timeout: int = 60):
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.endpoint = endpoint
        self.provider = provider
        if not self.api_key:
            raise ValueError(f"{provider} API key is required for a live run")

    def next_action(self, system: str, state: str) -> Action:
        payload = {
            "model": self.model,
            "instructions": system,
            "input": state,
            "text": {"format": {
                "type": "json_schema",
                "name": "research_action",
                "strict": True,
                "schema": ACTION_SCHEMA,
            }},
            "max_output_tokens": 400,
        }
        body = self._post_json(payload)
        text = _response_text(body)
        return _parse_action(text)

    def _post_json(self, payload: dict) -> dict:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "resilient-research-agent/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            message, code = _safe_error(detail)
            raise ProviderHTTPError(self.provider, exc.code, message, code, _retry_after(exc, message)) from exc
        return body


class OpenAIPlanner(ResponsesPlanner):
    def __init__(self, model: str = "gpt-5-mini", timeout: int = 60, api_key: str | None = None):
        super().__init__(model, api_key or os.environ.get("OPENAI_API_KEY", ""), "https://api.openai.com/v1/responses", "OpenAI", timeout)


class LibraPlanner(ResponsesPlanner):
    """Company-provided GPT-5.6 Sol access through its fixed Responses endpoint."""

    ENDPOINT = (
        "https://libra-ai-interviews.services.ai.azure.com/"
        "api/projects/proj-default/openai/v1/responses"
    )

    def __init__(self, model: str = "gpt-5.6-sol", timeout: int = 60, api_key: str | None = None):
        super().__init__(
            model,
            api_key or os.environ.get("LIBRA_INTERVIEW_API_KEY", ""),
            self.ENDPOINT,
            "Libra",
            timeout,
        )


class GroqPlanner(ResponsesPlanner):
    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        timeout: int = 60,
        api_key: str | None = None,
        min_request_interval: float = 0.75,
    ):
        super().__init__(model, api_key or os.environ.get("GROQ_API_KEY", ""), "https://api.groq.com/openai/v1/chat/completions", "Groq", timeout)
        self.min_request_interval = max(0.0, min_request_interval)
        self._last_request_at = 0.0

    def _post_json(self, payload: dict) -> dict:
        now = time.monotonic()
        wait = max(0.0, self._last_request_at + self.min_request_interval - now)
        if wait:
            time.sleep(wait)
        self._last_request_at = now + wait
        return super()._post_json(payload)

    def next_action(self, system: str, state: str) -> Action:
        compact = self._state_data(state)
        phase = self._forced_tool(state)
        if phase == "finish":
            return self._finish(compact)
        allowed_ids = self._allowed_source_ids(phase, compact)
        if phase == "read":
            if not allowed_ids:
                raise ValueError("no valid unread source is available")
            return Action("read", {"source_id": allowed_ids[0]}, "Selected deterministically from valid unread sources.")
        if phase == "search":
            return self._search(compact)
        if phase == "note":
            if not allowed_ids:
                raise ValueError("no read source is awaiting evidence")
            return self._note(compact, allowed_ids[0])
        raise ValueError(f"unsupported Groq phase: {phase}")

    def _search(self, compact: dict) -> Action:
        progress = compact.get("progress", {})
        searched = list(progress.get("searched_queries", []))
        if searched and not compact.get("sources"):
            fallback = self._fallback_query(compact, searched)
            if fallback:
                return Action("search", {"query": fallback}, "Shortened deterministically after an empty search.")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write one concise Wikipedia search query for the research goal. Focus on uncovered goal terms. "
                        "Do not repeat an already searched query. Return only the query text with no quotes, bullets, JSON, or explanation.\n"
                        + json.dumps({
                            "goal": compact.get("goal", ""),
                            "uncovered_goal_terms": progress.get("uncovered_goal_terms", []),
                            "searched_queries": searched,
                        }, ensure_ascii=False, separators=(",", ":"))
                    ),
                },
            ],
            "temperature": 0,
            "reasoning_effort": "low",
            "include_reasoning": False,
            "max_completion_tokens": 512,
        }
        query = self._clean_query(self._chat_text(self._post_json(payload)).splitlines()[0])
        if not query:
            raise ValueError("Groq returned an empty search query")
        normalized_searched = {self._normalize_query(item) for item in searched}
        uncovered = [str(term) for term in progress.get("uncovered_goal_terms", [])]
        query_stems = {self._term_key(word) for word in re.findall(r"[a-z0-9]+", query.lower())}
        uncovered_stems = {self._term_key(term.lower()) for term in uncovered}
        if progress.get("evidence_source_ids") and uncovered_stems and not query_stems.intersection(uncovered_stems):
            query = self._fallback_query(compact, searched)
        if self._normalize_query(query) in normalized_searched:
            query = self._fallback_query(compact, searched)
        if not query:
            raise ValueError("no distinct Wikipedia search query remains")
        return Action("search", {"query": query}, "Query proposed by Groq plain-text completion.")

    @staticmethod
    def _clean_query(query: str) -> str:
        omitted = {"article", "articles", "explanation", "wikipedia", "versus"}
        words = re.findall(r"[A-Za-z0-9]+", query.strip().strip('"\''))
        filtered = [word for word in words if word.lower() not in omitted]
        return " ".join(filtered[:6])

    @classmethod
    def _fallback_query(cls, compact: dict, searched: list[str]) -> str:
        omitted = {
            "and", "cite", "commonly", "compare", "distinguish", "every", "explain", "explanation", "from",
            "important", "least", "modern", "relevant", "repeated", "source", "sources", "substantive", "the",
            "use", "versus", "wikipedia", "why", "with",
        }
        goal_words = [
            word for word in re.findall(r"[A-Za-z0-9]+", str(compact.get("goal", "")))
            if len(word) > 2 and word.lower() not in omitted and not word.isdigit()
        ]
        uncovered = [str(term) for term in compact.get("progress", {}).get("uncovered_goal_terms", [])]
        subject = [str(term) for term in compact.get("progress", {}).get("subject_terms", [])]
        if not subject:
            subject = goal_words[:3]
        candidates = [
            " ".join([*subject, *uncovered][:6]),
            " ".join([*goal_words[:3], *uncovered][:6]),
            " ".join(goal_words[:6]),
            " ".join(uncovered),
        ]
        normalized_searched = {cls._normalize_query(item) for item in searched}
        return next((item for item in candidates if item and cls._normalize_query(item) not in normalized_searched), "")

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", query.lower()))

    @staticmethod
    def _term_key(term: str) -> str:
        if term.endswith("ies") and len(term) > 5:
            return term[:-3] + "y"
        if term.endswith("s") and not term.endswith("ss") and len(term) > 4:
            return term[:-1]
        return term

    def _note(self, compact: dict, source_id: str) -> Action:
        extract = compact.get("read_extracts", {}).get(source_id, "")
        if not extract:
            raise ValueError(f"no extract is available for {source_id}")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Copy one contiguous, claim-bearing excerpt from the supplied source text. It must directly concern the named subject "
                        "in the research goal and help cover the listed goal terms; do not use an analogous event or a different subject. "
                        "It only needs to cover one relevant term, not every listed term. Copy the characters verbatim: do not paraphrase, "
                        "add ellipses, add quotation marks, or explain. If the text has no relevant excerpt, return exactly NO_RELEVANT_EXCERPT.\n"
                        + json.dumps({
                            "goal": compact.get("goal", ""),
                            "source_id": source_id,
                            "source_title": compact.get("sources", {}).get(source_id, {}).get("title", ""),
                            "required_subject_terms": compact.get("progress", {}).get("subject_terms", []),
                            "uncovered_goal_terms": compact.get("progress", {}).get("uncovered_goal_terms", []),
                            "recent_errors": compact.get("recent_errors", []),
                            "source_text": extract,
                        }, ensure_ascii=False, separators=(",", ":"))
                    ),
                },
            ],
            "temperature": 0,
            "reasoning_effort": "low",
            "include_reasoning": False,
            "max_completion_tokens": 512,
        }
        excerpt = self._chat_text(self._post_json(payload)).strip().strip('"')
        if excerpt == "NO_RELEVANT_EXCERPT":
            return Action("skip", {"source_id": source_id}, "Groq found no relevant verbatim evidence in the source.")
        if not excerpt:
            raise ValueError("Groq returned an empty evidence excerpt")
        return Action("note", {"source_id": source_id, "excerpt": excerpt}, "Excerpt proposed by Groq plain-text completion.")

    def _finish(self, compact: dict) -> Action:
        evidence = compact.get("evidence", [])
        allowed_ids = sorted({item.get("source_id", "") for item in evidence if item.get("source_id")})
        finish_state = {
            "goal": compact.get("goal", ""),
            "minimum_sources": compact.get("progress", {}).get("minimum_sources", 1),
            "allowed_citation_ids": allowed_ids,
            "evidence": evidence,
            "recent_errors": compact.get("recent_errors", []),
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write the final research answer using only the saved evidence below. "
                        "Cite every substantive claim using only an ID in allowed_citation_ids. "
                        "Use the exact ASCII citation format [S1], [S2], etc.; never use superscript numbers, footnotes, or bare numbers. "
                        "Place each citation before the sentence-ending punctuation: correct `Claim [S1].`; incorrect `Claim. [S1]`. "
                        "Put a citation in every factual sentence, including uncertainty statements. "
                        "Do not add general background, definitions, causal details, or uncertainty unless the evidence states them. "
                        "Prefer a shorter fully supported answer over a broader answer containing reasonable but unsupported knowledge. "
                        "Never cite a discovered-but-unsaved source or invent an ID. Use at least minimum_sources distinct IDs. "
                        "State uncertainty only when the saved evidence supports it. Return only the answer text, not JSON and not a tool call.\n"
                        + json.dumps(finish_state, ensure_ascii=False, separators=(",", ":"))
                    ),
                },
            ],
            "temperature": 0,
            "reasoning_effort": "low",
            "include_reasoning": False,
            "max_completion_tokens": 1200,
        }
        answer = self._chat_text(self._post_json(payload))
        if not answer:
            raise ValueError("Groq final synthesis returned no answer text")
        return Action("finish", {"answer": answer}, "Synthesized by Groq from saved evidence.")

    @staticmethod
    def _chat_text(body: dict) -> str:
        choice = body.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            finish_reason = choice.get("finish_reason", "unknown")
            raise ValueError(f"Groq response contained no plain-text content (finish_reason={finish_reason})")
        return content.strip()

    @staticmethod
    def _state_data(state: str) -> dict:
        try:
            return json.loads(state.split("\n", 1)[-1])
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _state_text(compact: dict) -> str:
        return "Goal and compact working state:\n" + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _allowed_source_ids(forced_tool: str, compact: dict) -> list[str]:
        progress = compact.get("progress", {})
        if forced_tool == "note":
            return list(progress.get("read_without_evidence", []))
        if forced_tool != "read":
            return []
        candidates = progress.get("coverage_candidate_ids", [])
        if candidates:
            return list(candidates)
        return list(progress.get("unread_source_ids", []))

    @staticmethod
    def _forced_tool(state: str) -> str:
        """Choose the next valid phase; the model still supplies the action arguments."""
        try:
            compact = GroqPlanner._state_data(state)
        except TypeError:
            return "search"

        progress = compact.get("progress", {})
        if progress.get("read_without_evidence"):
            return "note"
        if not compact.get("sources"):
            return "search"
        evidence_sources = progress.get("evidence_source_ids", [])
        minimum_sources = progress.get("minimum_sources", 2)
        if (
            evidence_sources
            and progress.get("uncovered_goal_terms")
            and not progress.get("coverage_candidate_ids")
            and len(progress.get("searched_queries", [])) < 4
        ):
            return "search"
        if len(evidence_sources) >= minimum_sources:
            if not progress.get("uncovered_goal_terms"):
                return "finish"
            if progress.get("coverage_candidate_ids"):
                return "read"
            if len(progress.get("searched_queries", [])) < 4:
                return "search"
        if progress.get("unread_source_ids"):
            return "read"
        if len(progress.get("searched_queries", [])) < 4:
            return "search"
        return "finish"


def _parse_action(text: str) -> Action:
    raw = json.loads(text)
    return Action(kind=raw["kind"], args=raw.get("args", {}), rationale=raw.get("rationale", ""))


def _safe_error(detail: str) -> tuple[str, str]:
    try:
        error = json.loads(detail).get("error", {})
        message = str(error.get("message", "request failed"))
        code = str(error.get("code", ""))
    except json.JSONDecodeError:
        message, code = detail.strip() or "request failed", ""
    message = re.sub(r"organization `[^`]+`", "organization <redacted>", message)
    message = re.sub(r"https?://\S+", "<url>", message)
    return message[:240], code


def _retry_after(exc: urllib.error.HTTPError, message: str) -> float:
    header = exc.headers.get("retry-after") if exc.headers else None
    match = re.search(r"try again in ([0-9.]+)s", message, re.IGNORECASE)
    try:
        return min(10.0, max(0.1, float(header or (match.group(1) if match else 1.0))))
    except ValueError:
        return 1.0


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
