from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetStats:
    n_examples: int
    avg_prompt_len: float
    avg_response_len: float
    max_prompt_len: int
    min_prompt_len: int
    vocab_size: int
    domain_distribution: dict


class DatasetAnalyzer:
    def __init__(self) -> None:
        pass

    def analyze(
        self,
        examples: list[dict],
        prompt_key: str = "prompt",
        response_key: str = "response",
        domain_key: str = "domain",
    ) -> DatasetStats:
        if not examples:
            return DatasetStats(
                n_examples=0,
                avg_prompt_len=0.0,
                avg_response_len=0.0,
                max_prompt_len=0,
                min_prompt_len=0,
                vocab_size=0,
                domain_distribution={},
            )
        prompt_lens = [len(e.get(prompt_key, "")) for e in examples]
        response_lens = [len(e.get(response_key, "")) for e in examples]
        vocab: set = set()
        for e in examples:
            vocab.update(e.get(prompt_key, "").split())
            vocab.update(e.get(response_key, "").split())
        domain_dist: dict[str, int] = {}
        for e in examples:
            d = e.get(domain_key, "")
            if d:
                domain_dist[d] = domain_dist.get(d, 0) + 1
        return DatasetStats(
            n_examples=len(examples),
            avg_prompt_len=sum(prompt_lens) / len(prompt_lens),
            avg_response_len=sum(response_lens) / len(response_lens),
            max_prompt_len=max(prompt_lens),
            min_prompt_len=min(prompt_lens),
            vocab_size=len(vocab),
            domain_distribution=domain_dist,
        )

    def length_histogram(
        self,
        examples: list[dict],
        key: str = "prompt",
        n_bins: int = 10,
    ) -> dict[str, int]:
        lengths = [len(e.get(key, "")) for e in examples]
        if not lengths:
            return {}
        lo = min(lengths)
        hi = max(lengths)
        if lo == hi:
            label = f"{lo}-{hi}"
            return {label: len(lengths)}
        bin_width = (hi - lo) / n_bins
        bins: dict[str, int] = {}
        for i in range(n_bins):
            bin_lo = lo + i * bin_width
            bin_hi = lo + (i + 1) * bin_width
            label = f"{int(bin_lo)}-{int(bin_hi)}"
            bins[label] = 0
        for length in lengths:
            if length == hi:
                idx = n_bins - 1
            else:
                idx = int((length - lo) / bin_width)
                idx = min(idx, n_bins - 1)
            bin_lo = lo + idx * bin_width
            bin_hi = lo + (idx + 1) * bin_width
            label = f"{int(bin_lo)}-{int(bin_hi)}"
            bins[label] += 1
        return bins

    def detect_duplicates(self, examples: list[dict], key: str = "prompt") -> list[int]:
        seen: dict[str, int] = {}
        duplicates: list[int] = []
        for i, e in enumerate(examples):
            val = e.get(key, "")
            if val in seen:
                duplicates.append(i)
            else:
                seen[val] = i
        return duplicates

    def quality_score(
        self,
        example: dict,
        prompt_key: str = "prompt",
        response_key: str = "response",
    ) -> float:
        prompt_len = max(len(example.get(prompt_key, "")), 1)
        response_len = len(example.get(response_key, ""))
        score = response_len / prompt_len
        return min(score, 1.0)


DATASET_ANALYZER_REGISTRY: dict = {"default": DatasetAnalyzer}
