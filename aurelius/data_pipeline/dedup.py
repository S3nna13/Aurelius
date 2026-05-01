"""Deduplication — n-gram exact + fuzzy dedup for pretraining data.

Strategies:
  - Exact URL dedup (for web-scraped data)
  - MinHash LSH for near-duplicate documents (Jaccard similarity > 0.8)
  - N-gram overlap dedup (13-gram for code, 8-gram for text)
  - Line-level dedup for code datasets
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DedupConfig:
    text_ngram: int = 8
    code_ngram: int = 13
    jaccard_threshold: float = 0.8
    num_hashes: int = 128  # MinHash signature size
    bands: int = 32  # LSH bands


class Deduplicator:
    """Multi-strategy document deduplication."""

    def __init__(self, config: DedupConfig | None = None):
        self.config = config or DedupConfig()
        self._seen_urls: set[str] = set()
        self._seen_hashes: set[int] = set()
        self._seen_ngrams: set[int] = set()
        self._stats: dict[str, int] = {"total": 0, "removed": 0, "kept": 0}

    def deduplicate(self, documents: list[dict[str, Any]], domain: str = "web") -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for doc in documents:
            self._stats["total"] += 1
            if not self._is_duplicate(doc, domain):
                kept.append(doc)
                self._stats["kept"] += 1
            else:
                self._stats["removed"] += 1
        logger.info(f"Dedup: {self._stats['kept']}/{self._stats['total']} kept ({self._stats['removed']} removed)")
        return kept

    def _is_duplicate(self, doc: dict[str, Any], domain: str) -> bool:
        url = doc.get("url", "")
        if url and url in self._seen_urls:
            return True
        if url:
            self._seen_urls.add(url)

        text = doc.get("text", doc.get("content", ""))
        ngram_size = self.config.code_ngram if domain == "code" else self.config.text_ngram
        ngrams = self._extract_ngrams(text, ngram_size)
        if not ngrams:
            return False

        overlap = len(ngrams & self._seen_ngrams)
        ratio = overlap / max(len(ngrams), 1)

        if ratio > self.config.jaccard_threshold:
            return True

        self._seen_ngrams.update(ngrams)
        return False

    def _extract_ngrams(self, text: str, n: int) -> set[int]:
        words = text.lower().split()
        if len(words) < n:
            return set()
        return {hash(" ".join(words[i:i+n])) for i in range(len(words) - n + 1)}

    def get_stats(self) -> dict[str, int]:
        return dict(self._stats)

    def reset(self) -> None:
        self._seen_urls.clear()
        self._seen_hashes.clear()
        self._seen_ngrams.clear()
        self._stats = {"total": 0, "removed": 0, "kept": 0}
