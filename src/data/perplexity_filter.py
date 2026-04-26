"""Perplexity-based data filtering for LLM pre-training datasets.

Uses a reference model (or proxy) to compute token-level log-probabilities and
sequence perplexity, then keeps only documents within a configured perplexity
band.  All computation is pure PyTorch — no HuggingFace, no scipy, no sklearn.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PerplexityFilterConfig:
    min_perplexity: float = 10.0
    max_perplexity: float = 1000.0
    batch_size: int = 8
    max_seq_len: int = 512


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


def compute_token_log_probs(logits: Tensor, targets: Tensor) -> Tensor:
    """Compute cross-entropy log-probability per token.

    Args:
        logits:  (B, T, V) — raw unnormalised model outputs.
        targets: (B, T)    — integer token ids.

    Returns:
        log_probs: (B, T) — negative cross-entropy per position (i.e. log p).
    """
    B, T, V = logits.shape
    # F.cross_entropy with reduction='none' returns positive CE; negate for log-prob.
    log_probs = -F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        reduction="none",
    )  # (B*T,)
    return log_probs.reshape(B, T)


def compute_sequence_perplexity(log_probs: Tensor, lengths: Tensor) -> Tensor:
    """Compute per-sequence perplexity using a variable-length mask.

    Only positions up to ``lengths[i]`` are averaged for sequence i; padding
    positions are excluded.

    Args:
        log_probs: (B, T) — per-token log-probabilities (should be <= 0).
        lengths:   (B,)   — number of valid (non-padding) tokens per sequence.

    Returns:
        perplexity: (B,) — exp(-mean_log_prob) per sequence.
    """
    B, T = log_probs.shape
    # Build a mask: position j is valid iff j < lengths[i].
    positions = torch.arange(T, device=log_probs.device).unsqueeze(0)  # (1, T)
    mask = positions < lengths.unsqueeze(1).to(log_probs.device)  # (B, T)

    # Masked mean: sum log-probs over valid positions / length.
    masked_sum = (log_probs * mask.float()).sum(dim=1)  # (B,)
    safe_lengths = lengths.float().clamp(min=1.0).to(log_probs.device)
    mean_log_prob = masked_sum / safe_lengths  # (B,)

    perplexity = torch.exp(-mean_log_prob)  # (B,)
    return perplexity


# ---------------------------------------------------------------------------
# PerplexityScorer
# ---------------------------------------------------------------------------


class PerplexityScorer:
    """Score sequences by perplexity using an external model function.

    Args:
        model_fn: Callable that accepts token ids ``(B, T)`` and returns
                  logits ``(B, T, V)``.  Must be compatible with
                  ``torch.no_grad`` context.
        config:   :class:`PerplexityFilterConfig` instance.
    """

    def __init__(
        self,
        model_fn: Callable[[Tensor], Tensor],
        config: PerplexityFilterConfig,
    ) -> None:
        self.model_fn = model_fn
        self.config = config

    # ------------------------------------------------------------------

    def score_batch(self, token_ids: Tensor, lengths: Tensor) -> Tensor:
        """Return perplexity for a batch of pre-padded sequences.

        Args:
            token_ids: (B, T) integer tensor (may contain padding).
            lengths:   (B,)   number of real tokens per sequence.

        Returns:
            perplexity: (B,) float tensor.
        """
        with torch.no_grad():
            logits = self.model_fn(token_ids)  # (B, T, V)

        # Targets are the input ids themselves shifted by 0
        # (language-model convention: predict next token from current logits;
        # here we align logits[:, :-1] → targets[:, 1:] for true LM scoring,
        # but callers may also use the full-length convention; we use the
        # simpler full-sequence alignment so the output shape stays (B, T)).
        log_probs = compute_token_log_probs(logits, token_ids)  # (B, T)
        return compute_sequence_perplexity(log_probs, lengths)  # (B,)

    # ------------------------------------------------------------------

    def score_texts(self, encoded_texts: list[Tensor]) -> list[float]:
        """Pad a list of 1-D token tensors, score in batches, return perplexities.

        Args:
            encoded_texts: list of 1-D integer tensors of varying length.

        Returns:
            List of perplexity floats, one per input sequence.
        """
        if not encoded_texts:
            return []

        cfg = self.config
        results: list[float] = []

        # Process in batches.
        for start in range(0, len(encoded_texts), cfg.batch_size):
            batch_seqs = encoded_texts[start : start + cfg.batch_size]

            # Truncate to max_seq_len.
            batch_seqs = [s[: cfg.max_seq_len] for s in batch_seqs]

            lengths = torch.tensor([len(s) for s in batch_seqs], dtype=torch.long)
            max_len = int(lengths.max().item())

            # Pad with zeros.
            padded = torch.zeros(len(batch_seqs), max_len, dtype=torch.long)
            for i, seq in enumerate(batch_seqs):
                padded[i, : len(seq)] = seq

            perps = self.score_batch(padded, lengths)  # (B,)
            results.extend(perps.tolist())

        return results


# ---------------------------------------------------------------------------
# PerplexityFilter
# ---------------------------------------------------------------------------


class PerplexityFilter:
    """Keep only sequences whose perplexity falls in [min_perplexity, max_perplexity].

    Args:
        scorer: A :class:`PerplexityScorer` instance.
        config: A :class:`PerplexityFilterConfig` instance.
    """

    def __init__(self, scorer: PerplexityScorer, config: PerplexityFilterConfig) -> None:
        self.scorer = scorer
        self.config = config

    # ------------------------------------------------------------------

    def filter_batch(self, encoded_texts: list[Tensor]) -> tuple[list[Tensor], list[float]]:
        """Filter a list of encoded sequences by perplexity.

        Args:
            encoded_texts: list of 1-D integer tensors.

        Returns:
            (kept_texts, kept_scores) — only sequences whose perplexity is
            in [min_perplexity, max_perplexity].
        """
        if not encoded_texts:
            return [], []

        scores = self.scorer.score_texts(encoded_texts)
        kept_texts: list[Tensor] = []
        kept_scores: list[float] = []
        lo, hi = self.config.min_perplexity, self.config.max_perplexity

        for seq, score in zip(encoded_texts, scores):
            if lo <= score <= hi:
                kept_texts.append(seq)
                kept_scores.append(score)

        return kept_texts, kept_scores

    # ------------------------------------------------------------------

    def get_stats(self, scores: list[float]) -> dict[str, float]:
        """Compute summary statistics for a list of perplexity scores.

        Uses a sort-based median (no scipy).

        Args:
            scores: list of perplexity floats.

        Returns:
            Dict with keys: mean, median, min, max, n_filtered, n_kept.
            n_filtered = number of scores outside [min_perplexity, max_perplexity].
            n_kept     = number of scores inside  [min_perplexity, max_perplexity].
        """
        if not scores:
            return {
                "mean": float("nan"),
                "median": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "n_filtered": 0.0,
                "n_kept": 0.0,
            }

        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        mean_val = sum(sorted_scores) / n

        mid = n // 2
        if n % 2 == 1:
            median_val = sorted_scores[mid]
        else:
            median_val = (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0

        lo, hi = self.config.min_perplexity, self.config.max_perplexity
        n_kept = sum(1 for s in scores if lo <= s <= hi)
        n_filtered = n - n_kept

        return {
            "mean": mean_val,
            "median": median_val,
            "min": float(sorted_scores[0]),
            "max": float(sorted_scores[-1]),
            "n_filtered": float(n_filtered),
            "n_kept": float(n_kept),
        }


# ---------------------------------------------------------------------------
# DatasetPerplexityRanker
# ---------------------------------------------------------------------------


class DatasetPerplexityRanker:
    """Rank a dataset of encoded sequences by perplexity (ascending).

    Lower perplexity → model assigns higher probability → typically higher
    quality / more in-distribution text.

    Args:
        scorer: A :class:`PerplexityScorer` instance.
    """

    def __init__(self, scorer: PerplexityScorer) -> None:
        self.scorer = scorer

    # ------------------------------------------------------------------

    def rank(
        self,
        encoded_texts: list[Tensor],
        return_scores: bool = False,
    ):
        """Sort encoded texts by perplexity ascending.

        Args:
            encoded_texts: list of 1-D integer tensors.
            return_scores: if True, return (sorted_texts, sorted_scores).

        Returns:
            sorted_texts if ``return_scores`` is False, else
            (sorted_texts, sorted_scores).
        """
        if not encoded_texts:
            if return_scores:
                return [], []
            return []

        scores = self.scorer.score_texts(encoded_texts)
        paired = sorted(zip(scores, range(len(scores))), key=lambda x: x[0])
        sorted_texts = [encoded_texts[i] for _, i in paired]
        sorted_scores = [s for s, _ in paired]

        if return_scores:
            return sorted_texts, sorted_scores
        return sorted_texts

    # ------------------------------------------------------------------

    def get_percentile_bucket(self, score: float, scores: list[float], n_buckets: int = 10) -> int:
        """Return which decile (0-indexed) the given score falls into.

        Bucket 0  = bottom (lowest / best) perplexity range.
        Bucket n_buckets-1 = top (highest) perplexity range.

        Args:
            score:    the perplexity score to classify.
            scores:   the full distribution of scores.
            n_buckets: number of equal-width buckets (default 10 → deciles).

        Returns:
            int in [0, n_buckets - 1].
        """
        if not scores:
            return 0

        lo = min(scores)
        hi = max(scores)

        if hi == lo:
            return 0

        # Fractional position in [0, 1).
        frac = (score - lo) / (hi - lo)
        bucket = int(frac * n_buckets)
        # Clamp so that the maximum score lands in the last bucket.
        return min(bucket, n_buckets - 1)
