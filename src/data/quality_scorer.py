"""Dataset quality scoring pipeline: multi-signal ranking for training examples.

Combines perplexity, length normalization, n-gram novelty, near-duplicate
detection (SimHash), and instruction-following relevance into a composite score.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

STOPWORDS: set[str] = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "to",
    "of",
    "and",
    "in",
    "that",
    "for",
    "on",
    "with",
    "as",
    "at",
    "by",
    "from",
    "this",
    "it",
    "its",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QualitySignals:
    perplexity: float  # model perplexity on the response
    length_score: float  # score in [0,1] based on ideal length
    novelty_score: float  # n-gram novelty vs reference corpus
    dedup_score: float  # 1.0 = unique, 0.0 = duplicate
    instruction_relevance: float  # heuristic: does response contain instruction keywords?
    composite: float  # weighted combination


@dataclass
class ScoredExample:
    text: str
    instruction: str
    response: str
    signals: QualitySignals
    rank: int = 0  # rank in dataset (lower = better quality)


# ---------------------------------------------------------------------------
# Length scoring
# ---------------------------------------------------------------------------


def compute_length_score(
    text: str,
    min_len: int = 50,
    max_len: int = 2000,
    ideal_min: int = 100,
    ideal_max: int = 800,
) -> float:
    """Score in [0,1]. 1.0 = in ideal range. Linearly decays outside."""
    n = len(text)

    # Outside hard bounds -> 0
    if n < min_len or n > max_len:
        return 0.0

    # Inside ideal range -> 1
    if ideal_min <= n <= ideal_max:
        return 1.0

    # Between min_len and ideal_min: linearly ramp 0->1
    if n < ideal_min:
        span = ideal_min - min_len
        if span == 0:
            return 1.0
        return (n - min_len) / span

    # Between ideal_max and max_len: linearly decay 1->0
    span = max_len - ideal_max
    if span == 0:
        return 1.0
    return (max_len - n) / span


# ---------------------------------------------------------------------------
# N-gram novelty
# ---------------------------------------------------------------------------


def _get_ngrams(text: str, n: int) -> set[tuple]:
    words = text.lower().split()
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def compute_ngram_novelty(
    text: str,
    reference_texts: list[str],
    n: int = 4,
) -> float:
    """Fraction of n-grams in text not appearing in any reference_text."""
    text_ngrams = _get_ngrams(text, n)
    if not text_ngrams:
        return 1.0  # no n-grams -> trivially novel

    reference_ngrams: set[tuple] = set()
    for ref in reference_texts:
        reference_ngrams.update(_get_ngrams(ref, n))

    novel = text_ngrams - reference_ngrams
    return len(novel) / len(text_ngrams)


# ---------------------------------------------------------------------------
# SimHash
# ---------------------------------------------------------------------------


def simhash(text: str, n_bits: int = 64) -> int:
    """
    SimHash fingerprint of text.
    1. Tokenize to words
    2. For each word, compute hash, get bit vector
    3. Sum bit vectors weighted by word frequency
    4. Threshold to 0/1
    Returns: integer fingerprint
    """
    words = text.lower().split()
    if not words:
        return 0

    # Count word frequencies
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Accumulate weighted bit vector
    vector = [0] * n_bits
    for word, weight in freq.items():
        h = hash(word)
        for i in range(n_bits):
            bit = (h >> i) & 1
            vector[i] += weight if bit else -weight

    # Threshold: bit=1 if vector[i] > 0
    fingerprint = 0
    for i in range(n_bits):
        if vector[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int, n_bits: int = 64) -> int:
    """Hamming distance between two SimHash values."""
    xor = a ^ b
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count


def compute_dedup_score(
    text: str,
    seen_hashes: set[int],
    similarity_threshold: int = 3,
    n_bits: int = 64,
) -> tuple[float, int]:
    """
    Returns (dedup_score, simhash_of_text).
    dedup_score=1.0 if unique, 0.0 if near-duplicate.
    Adds hash to seen_hashes in-place.
    """
    h = simhash(text, n_bits=n_bits)
    is_duplicate = any(hamming_distance(h, s, n_bits) <= similarity_threshold for s in seen_hashes)
    seen_hashes.add(h)
    score = 0.0 if is_duplicate else 1.0
    return score, h


# ---------------------------------------------------------------------------
# Instruction relevance
# ---------------------------------------------------------------------------


def compute_instruction_relevance(instruction: str, response: str) -> float:
    """
    Heuristic: what fraction of instruction keywords appear in response?
    Uses simple word overlap after stopword removal.
    Returns float in [0, 1].
    """

    def keywords(text: str) -> set[str]:
        return {
            w.lower().strip(".,!?;:\"'")
            for w in text.split()
            if w.lower().strip(".,!?;:\"'") not in STOPWORDS and w.strip(".,!?;:\"'")
        }

    instr_kws = keywords(instruction)
    if not instr_kws:
        return 1.0  # no keywords to check

    resp_kws = keywords(response)
    overlap = instr_kws & resp_kws
    return len(overlap) / len(instr_kws)


# ---------------------------------------------------------------------------
# PerplexityScorer
# ---------------------------------------------------------------------------


class PerplexityScorer:
    """Compute language model perplexity on text for quality estimation."""

    def __init__(
        self,
        model: nn.Module,
        encode_fn: Callable[[str], list[int]],
        device: str = "cpu",
        max_seq_len: int = 512,
    ) -> None:
        self.model = model
        self.encode_fn = encode_fn
        self.device = device
        self.max_seq_len = max_seq_len
        self.model.eval()

    def score(self, text: str) -> float:
        """Return perplexity (exp of avg cross-entropy). Lower = more fluent."""
        token_ids = self.encode_fn(text)
        if not token_ids:
            return float("inf")

        token_ids = token_ids[: self.max_seq_len]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            loss, _logits, _extra = self.model(input_ids)

        return float(math.exp(loss.item()))

    def score_batch(self, texts: list[str]) -> list[float]:
        return [self.score(t) for t in texts]


# ---------------------------------------------------------------------------
# DatasetQualityScorer
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = {
    "perplexity": 0.3,
    "length": 0.2,
    "novelty": 0.2,
    "dedup": 0.2,
    "relevance": 0.1,
}


def _perplexity_to_score(perplexity: float, scale: float = 100.0) -> float:
    """Map perplexity to [0,1]. Moderate perplexity is most informative.
    Uses exp(-|log(ppl/scale)|) so score peaks when ppl == scale."""
    if perplexity <= 0 or math.isinf(perplexity) or math.isnan(perplexity):
        return 0.0
    return float(math.exp(-abs(math.log(perplexity / scale))))


class DatasetQualityScorer:
    def __init__(
        self,
        perplexity_scorer: PerplexityScorer | None = None,
        weights: dict[str, float] | None = None,
        reference_texts: list[str] | None = None,
    ) -> None:
        self.perplexity_scorer = perplexity_scorer
        self.weights = weights if weights is not None else dict(_DEFAULT_WEIGHTS)
        self.reference_texts = reference_texts or []
        self._seen_hashes: set[int] = set()

    def score_example(
        self,
        instruction: str,
        response: str,
        reference_texts: list[str] | None = None,
    ) -> QualitySignals:
        text = (instruction + " " + response).strip()
        refs = reference_texts if reference_texts is not None else self.reference_texts

        # 1. Perplexity
        if self.perplexity_scorer is not None:
            raw_ppl = self.perplexity_scorer.score(response)
        else:
            raw_ppl = 50.0  # neutral default
        ppl_score = _perplexity_to_score(raw_ppl)

        # 2. Length score (based on response)
        length_score = compute_length_score(response)

        # 3. Novelty
        novelty_score = compute_ngram_novelty(text, refs)

        # 4. Dedup
        dedup_score, _ = compute_dedup_score(text, self._seen_hashes)

        # 5. Instruction relevance
        relevance = compute_instruction_relevance(instruction, response)

        # 6. Composite
        w = self.weights
        total_w = (
            w.get("perplexity", 0.3)
            + w.get("length", 0.2)
            + w.get("novelty", 0.2)
            + w.get("dedup", 0.2)
            + w.get("relevance", 0.1)
        )
        if total_w == 0:
            total_w = 1.0

        composite = (
            w.get("perplexity", 0.3) * ppl_score
            + w.get("length", 0.2) * length_score
            + w.get("novelty", 0.2) * novelty_score
            + w.get("dedup", 0.2) * dedup_score
            + w.get("relevance", 0.1) * relevance
        ) / total_w

        composite = float(min(max(composite, 0.0), 1.0))

        return QualitySignals(
            perplexity=raw_ppl,
            length_score=length_score,
            novelty_score=novelty_score,
            dedup_score=dedup_score,
            instruction_relevance=relevance,
            composite=composite,
        )

    def score_dataset(
        self,
        examples: list[dict[str, str]],
        reference_texts: list[str] | None = None,
    ) -> list[ScoredExample]:
        """Score and rank all examples. Best quality = lowest rank."""
        # Reset dedup state for a fresh dataset scoring run
        self._seen_hashes = set()

        scored: list[ScoredExample] = []
        for ex in examples:
            instruction = ex.get("instruction", "")
            response = ex.get("response", "")
            signals = self.score_example(instruction, response, reference_texts)
            text = (instruction + " " + response).strip()
            scored.append(
                ScoredExample(
                    text=text,
                    instruction=instruction,
                    response=response,
                    signals=signals,
                )
            )

        # Sort: higher composite = better quality = lower rank number
        scored.sort(key=lambda s: s.signals.composite, reverse=True)
        for rank_idx, ex in enumerate(scored):
            ex.rank = rank_idx

        return scored

    def filter_top_k(
        self,
        scored: list[ScoredExample],
        k: int,
        strategy: str = "top_k",
        threshold: float = 0.5,
    ) -> list[ScoredExample]:
        if strategy == "top_k":
            return scored[:k]
        elif strategy == "threshold":
            above = [s for s in scored if s.signals.composite >= threshold]
            return above[:k]
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

    def get_quality_report(self, scored: list[ScoredExample]) -> dict[str, float]:
        """Return aggregate stats: mean/std of each signal, fraction passing threshold."""
        if not scored:
            return {}

        def _mean(vals: list[float]) -> float:
            return sum(vals) / len(vals)

        def _std(vals: list[float]) -> float:
            m = _mean(vals)
            variance = sum((v - m) ** 2 for v in vals) / len(vals)
            return math.sqrt(variance)

        signal_fields = [
            ("perplexity", [s.signals.perplexity for s in scored]),
            ("length_score", [s.signals.length_score for s in scored]),
            ("novelty_score", [s.signals.novelty_score for s in scored]),
            ("dedup_score", [s.signals.dedup_score for s in scored]),
            ("instruction_relevance", [s.signals.instruction_relevance for s in scored]),
            ("composite", [s.signals.composite for s in scored]),
        ]

        report: dict[str, float] = {}
        for name, vals in signal_fields:
            report[f"{name}_mean"] = _mean(vals)
            report[f"{name}_std"] = _std(vals)

        # Fraction with composite >= 0.5
        report["fraction_passing"] = sum(1 for s in scored if s.signals.composite >= 0.5) / len(
            scored
        )

        return report
