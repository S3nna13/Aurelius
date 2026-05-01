"""Data collection — sources, crawling, ingestion for 30T token pretraining corpus.

Sources:
  - CommonCrawl / FineWeb (web text, filtered)
  - The Stack v2 (code, 619 languages)
  - arXiv (science papers)
  - Wikipedia + Wikidata (encyclopedic)
  - GitHub issues + PRs (code discussions)
  - OpenWebMath (math)
  - Reddit / StackExchange (Q&A)
  - Books3 + PG-19 (books)
  - PubMed (biomedical)
  - Legal corpora (court cases, contracts)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    name: str
    url: str
    estimated_tokens: int
    domain: str
    priority: int = 10
    weight: float = 1.0


PRETRAINING_SOURCES: list[DataSource] = [
    DataSource("fineweb", "https://huggingface.co/datasets/HuggingFaceFW/fineweb", 15_000_000_000_000, "web", 1, 0.50),
    DataSource("the_stack_v2", "https://huggingface.co/datasets/bigcode/the-stack-v2", 5_000_000_000_000, "code", 2, 0.17),
    DataSource("arxiv", "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T", 2_000_000_000_000, "science", 3, 0.07),
    DataSource("wikipedia", "https://huggingface.co/datasets/wikipedia", 500_000_000_000, "web", 4, 0.02),
    DataSource("openwebmath", "https://huggingface.co/datasets/open-web-math/open-web-math", 1_000_000_000_000, "math", 5, 0.03),
    DataSource("github_clean", "https://huggingface.co/datasets/codeparrot/github-code-clean", 3_000_000_000_000, "code", 6, 0.10),
    DataSource("stackexchange", "https://huggingface.co/datasets/stack-exchange", 500_000_000_000, "web", 7, 0.02),
    DataSource("pubmed", "https://huggingface.co/datasets/pubmed", 500_000_000_000, "science", 8, 0.02),
    DataSource("books", "https://huggingface.co/datasets/bookcorpus", 1_000_000_000_000, "web", 9, 0.03),
    DataSource("legal", "https://huggingface.co/datasets/lex_glue", 500_000_000_000, "science", 10, 0.02),
    DataSource("dolmino", "https://huggingface.co/datasets/allenai/dolmino", 1_000_000_000_000, "web", 1, 0.03),
]


class DataCollector:
    """Collects data from configured sources with chunked streaming."""

    def __init__(self, output_dir: str | Path = "data/pretrain", chunk_size_tokens: int = 1_000_000_000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size_tokens
        self.collected: dict[str, dict[str, Any]] = {}

    def collect_source(self, source: DataSource, limit_tokens: int | None = None) -> dict[str, Any]:
        """Simulate collecting data from a source. In prod, downloads from HF/direct."""
        tokens = limit_tokens or source.estimated_tokens
        chunks = max(1, tokens // self.chunk_size)

        source_dir = self.output_dir / source.name
        source_dir.mkdir(exist_ok=True)

        manifest = {
            "source": source.name,
            "domain": source.domain,
            "tokens": tokens,
            "chunks": chunks,
            "files": [],
        }

        for i in range(chunks):
            chunk_path = source_dir / f"chunk_{i:04d}.jsonl"
            manifest["files"].append(str(chunk_path))

        self.collected[source.name] = manifest
        logger.info(f"Collected {source.name}: {tokens:,} tokens in {chunks} chunks")
        return manifest

    def collect_all(self, limit_per_source: int | None = None) -> dict[str, Any]:
        results = {}
        for source in PRETRAINING_SOURCES:
            results[source.name] = self.collect_source(source, limit_per_source)
        total_tokens = sum(r["tokens"] for r in results.values())
        return {"sources": results, "total_tokens": total_tokens}

    def get_domain_breakdown(self) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for manifest in self.collected.values():
            domain = manifest["domain"]
            breakdown[domain] = breakdown.get(domain, 0) + manifest["tokens"]
        return breakdown
