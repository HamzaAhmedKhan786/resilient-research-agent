from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ResearchTools(Protocol):
    def search(self, query: str) -> list[dict[str, str]]: ...
    def read(self, locator: str) -> str: ...


@dataclass
class LocalCorpusTools:
    corpus_dir: Path

    def search(self, query: str) -> list[dict[str, str]]:
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for path in self.corpus_dir.glob("*.txt"):
            text = path.read_text(encoding="utf-8")
            score = sum(text.lower().count(term) for term in terms)
            if score:
                scored.append((score, path, text))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [
            {
                "title": text.splitlines()[0].lstrip("# "),
                "locator": str(path.resolve()),
                "snippet": " ".join(text.split())[:240],
            }
            for _, path, text in scored[:5]
        ]

    def read(self, locator: str) -> str:
        path = Path(locator).resolve()
        root = self.corpus_dir.resolve()
        if root not in path.parents:
            raise ValueError("read target is outside the configured corpus")
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("local source is empty")
        return text


@dataclass
class HttpTools:
    """Small live web adapter using Wikipedia's public API and page extracts."""

    timeout: int = 20
    user_agent: str = "resilient-research-agent/0.1 (evaluation project)"

    def search(self, query: str) -> list[dict[str, str]]:
        params = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query, "format": "json", "utf8": 1})
        data = self._get_json("https://en.wikipedia.org/w/api.php?" + params)
        results = []
        for item in data.get("query", {}).get("search", [])[:5]:
            title = item["title"]
            results.append({
                "title": title,
                "locator": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                "snippet": re.sub(r"<[^>]+>", "", item.get("snippet", "")),
            })
        return results

    def read(self, locator: str) -> str:
        title = urllib.parse.unquote(locator.rsplit("/", 1)[-1]).replace("_", " ")
        params = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": title, "format": "json", "utf8": 1})
        data = self._get_json("https://en.wikipedia.org/w/api.php?" + params)
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            raise ValueError("Wikipedia returned no page")
        page = next(iter(pages.values()))
        if "missing" in page:
            raise ValueError("Wikipedia page does not exist")
        extract = page.get("extract", "")
        if not extract.strip():
            raise ValueError("Wikipedia page extract is empty")
        return extract[:30000]

    def _get_json(self, url: str) -> dict:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)


class FlakyTools:
    """Failure-injection decorator: fail named operations a bounded number of times."""

    def __init__(self, inner: ResearchTools, failures: dict[str, int]):
        self.inner, self.failures = inner, dict(failures)

    def _maybe_fail(self, name: str) -> None:
        if self.failures.get(name, 0) > 0:
            self.failures[name] -= 1
            raise TimeoutError(f"injected transient {name} failure")

    def search(self, query: str) -> list[dict[str, str]]:
        self._maybe_fail("search")
        return self.inner.search(query)

    def read(self, locator: str) -> str:
        self._maybe_fail("read")
        return self.inner.read(locator)
