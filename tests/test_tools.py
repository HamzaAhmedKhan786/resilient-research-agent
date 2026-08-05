from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.tools import HttpTools, LocalCorpusTools


class ToolTests(unittest.TestCase):
    def test_local_corpus_ranks_matching_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "weak.txt").write_text("# Weak\ncheckpoint", encoding="utf-8")
            (root / "strong.txt").write_text("# Strong\ncheckpoint checkpoint durable", encoding="utf-8")
            results = LocalCorpusTools(root).search("checkpoint durable")
        self.assertEqual(results[0]["title"], "Strong")

    def test_local_corpus_rejects_reads_outside_root(self):
        with tempfile.TemporaryDirectory() as corpus_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            outside = Path(outside_tmp) / "secret.txt"
            outside.write_text("not allowed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the configured corpus"):
                LocalCorpusTools(Path(corpus_tmp)).read(str(outside))

    def test_local_corpus_rejects_empty_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty.txt"
            empty.write_text("   ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "local source is empty"):
                LocalCorpusTools(root).read(str(empty))

    def test_wikipedia_search_builds_encoded_locators(self):
        tools = HttpTools()
        payload = {"query": {"search": [{
            "title": "Tacoma Narrows Bridge (1940)",
            "snippet": "A <span class=\"searchmatch\">bridge</span> article",
        }]}}
        with patch.object(tools, "_get_json", return_value=payload):
            results = tools.search("Tacoma bridge")
        self.assertEqual(results[0]["locator"], "https://en.wikipedia.org/wiki/Tacoma_Narrows_Bridge_%281940%29")
        self.assertEqual(results[0]["snippet"], "A bridge article")

    def test_wikipedia_read_rejects_missing_or_empty_pages(self):
        tools = HttpTools()
        missing = {"query": {"pages": {"-1": {"missing": True}}}}
        with patch.object(tools, "_get_json", return_value=missing):
            with self.assertRaisesRegex(ValueError, "does not exist"):
                tools.read("https://en.wikipedia.org/wiki/Missing")

        empty = {"query": {"pages": {"1": {"extract": ""}}}}
        with patch.object(tools, "_get_json", return_value=empty):
            with self.assertRaisesRegex(ValueError, "extract is empty"):
                tools.read("https://en.wikipedia.org/wiki/Empty")


if __name__ == "__main__":
    unittest.main()
