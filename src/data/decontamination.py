"""Decontamination pipeline for training data.

From Phi-4 §1.1, Appendix B.1:
Ensures benchmark test data is not present in training data by
n-gram matching against known benchmark datasets. Contaminated
samples are either removed or flagged.

Supports:
  - N-gram overlap detection (8-gram, 13-gram for code)
  - Multiple benchmark dataset formats
  - JSONL, text, and parquet input formats
  - Reporting with contamination statistics
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ContaminationResult:
    sample_id: str
    contaminated: bool
    benchmark: str = ""
    max_ngram_overlap: int = 0
    overlap_ratio: float = 0.0


@dataclass
class BenchmarkFingerprint:
    name: str
    ngram_size: int = 8
    fingerprints: set[int] = field(default_factory=set)


class Decontaminator:
    """Training data decontamination via n-gram fingerprint matching.

    Args:
        ngram_size: Default n-gram size for fingerprinting (default: 8).
        code_ngram_size: N-gram size for code benchmarks (default: 13).
        threshold: Minimum n-gram overlap to flag contamination (default: 0.5).
    """

    def __init__(
        self,
        ngram_size: int = 8,
        code_ngram_size: int = 13,
        threshold: float = 0.5,
    ):
        self.ngram_size = ngram_size
        self.code_ngram_size = code_ngram_size
        self.threshold = threshold
        self.benchmarks: list[BenchmarkFingerprint] = []

    def add_benchmark(
        self,
        name: str,
        texts: list[str],
        is_code: bool = False,
    ) -> BenchmarkFingerprint:
        ngram_size = self.code_ngram_size if is_code else self.ngram_size
        fp = BenchmarkFingerprint(name=name, ngram_size=ngram_size)

        for text in texts:
            normalized = self._normalize(text)
            fp.fingerprints.update(self._ngrams(normalized, ngram_size))

        self.benchmarks.append(fp)
        logger.info(
            "Loaded benchmark '%s': %d %d-gram fingerprints",
            name, len(fp.fingerprints), ngram_size,
        )
        return fp

    def add_benchmark_file(
        self,
        name: str,
        path: str,
        is_code: bool = False,
    ) -> BenchmarkFingerprint:
        path = Path(path)
        texts: list[str] = []

        if path.suffix == ".jsonl":
            with open(path) as f:
                for line in f:
                    data = json.loads(line)
                    texts.append(data.get("text", data.get("content", "")))
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = [item.get("text", item.get("content", str(item))) for item in data]
                else:
                    texts = [str(data)]
        else:
            texts = [path.read_text()]

        return self.add_benchmark(name, texts, is_code)

    def check_sample(self, text: str) -> ContaminationResult:
        sample_id = hashlib.sha256(text.encode()).hexdigest()[:12]
        normalized = self._normalize(text)
        sample_ngrams = self._ngrams(normalized, self.ngram_size)

        best_overlap = 0
        best_benchmark = ""

        for bm in self.benchmarks:
            if not sample_ngrams:
                continue
            overlap = len(sample_ngrams & bm.fingerprints)
            ratio = overlap / len(sample_ngrams)
            if ratio > best_overlap:
                best_overlap = ratio
                best_benchmark = bm.name

        return ContaminationResult(
            sample_id=sample_id,
            contaminated=best_overlap >= self.threshold,
            benchmark=best_benchmark,
            max_ngram_overlap=best_overlap,
            overlap_ratio=best_overlap,
        )

    def filter_dataset(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> tuple[list[dict], list[ContaminationResult]]:
        cleaned: list[dict] = []
        contaminated: list[ContaminationResult] = []
        input_path = Path(input_path)

        with open(input_path) as f:
            for line in f:
                sample = json.loads(line)
                text = sample.get("text", sample.get("content", ""))
                result = self.check_sample(text)

                if result.contaminated:
                    contaminated.append(result)
                else:
                    cleaned.append(sample)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for sample in cleaned:
                    f.write(json.dumps(sample) + "\n")

        logger.info(
            "Decontamination: %d kept, %d removed (%.1f%%)",
            len(cleaned), len(contaminated),
            len(contaminated) / max(len(cleaned) + len(contaminated), 1) * 100,
        )
        return cleaned, contaminated

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    @staticmethod
    def _ngrams(text: str, n: int) -> set[int]:
        words = text.split()
        if len(words) < n:
            if words:
                return {hash(" ".join(words))}
            return set()
        return {hash(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}
