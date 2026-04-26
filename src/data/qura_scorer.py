"""QuRating-style pretraining data quality scorer.

Implements the four quality dimensions from:
  "QuRating: Selecting High-Quality Data for Training Language Models"
  arXiv:2402.09739, Princeton NLP 2024.

Rather than querying an external LLM (as the paper does), this module
implements a *perplexity-based quality proxy* that operates entirely on
pre-computed per-token log-probabilities and token IDs, using pure PyTorch.

Quality dimensions (paper §3):
  1. Writing Quality    — grammar, coherence, style
  2. Educational Value  — would a student learn from this?
  3. Facts / Trivia     — contains factual, verifiable information?
  4. Required Expertise — advanced domain knowledge required?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QuRatingConfig:
    """Weights and thresholds for the QuRating composite score.

    Weights must be positive; they are normalised internally so they need
    not sum to 1.0.
    """

    writing_weight: float = 0.35
    educational_weight: float = 0.30
    facts_weight: float = 0.20
    expertise_weight: float = 0.15

    # Perplexity above this value is treated as the worst possible quality.
    max_perplexity: float = 1000.0

    # Sequences shorter than this (in valid tokens) are considered
    # insufficient and receive a near-zero composite score.
    min_tokens: int = 10


# ---------------------------------------------------------------------------
# Score container
# ---------------------------------------------------------------------------


class QuRatingScores(NamedTuple):
    """Per-document scores along each QuRating quality dimension.

    All individual scores are in [0, 1]; composite is their weighted average.
    """

    writing_quality: float  # 0-1; higher = better writing (lower perplexity)
    educational_value: float  # 0-1; higher = more educational
    facts_score: float  # 0-1; higher = more factual / numeric content
    expertise_score: float  # 0-1; higher = more specialised vocabulary
    composite: float  # weighted average of the four dimensions


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------


class QuRaScorer:
    """QuRating-style scorer operating on tokenised documents.

    Parameters
    ----------
    config:
        Scoring hyper-parameters.  Defaults to ``QuRatingConfig()``.

    Usage
    -----
    Obtain ``token_ids`` and ``log_probs`` (per-token log-probabilities from a
    reference language model), then call :meth:`score_tokens` for a single
    document or :meth:`score_batch` for a padded batch.
    """

    def __init__(self, config: QuRatingConfig | None = None) -> None:
        self.config = config if config is not None else QuRatingConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_tokens(
        self,
        token_ids: Tensor,  # (T,) int token ids
        log_probs: Tensor,  # (T,) per-token log-probabilities from a ref LM
        vocab_size: int = 128000,
    ) -> QuRatingScores:
        """Score a single tokenised document from its LM log-probs and token ids.

        Parameters
        ----------
        token_ids:
            1-D integer tensor of shape ``(T,)``.
        log_probs:
            1-D float tensor of shape ``(T,)`` containing the log-probability
            of each token under a reference language model.
        vocab_size:
            Vocabulary size of the tokeniser (used to derive token-id
            heuristics for educational / facts / expertise signals).

        Returns
        -------
        QuRatingScores
        """
        if token_ids.dim() != 1:
            raise ValueError(f"token_ids must be 1-D, got shape {tuple(token_ids.shape)}")
        if log_probs.dim() != 1:
            raise ValueError(f"log_probs must be 1-D, got shape {tuple(log_probs.shape)}")
        if token_ids.shape[0] != log_probs.shape[0]:
            raise ValueError(
                f"token_ids length {token_ids.shape[0]} != log_probs length {log_probs.shape[0]}"
            )

        cfg = self.config
        T = token_ids.shape[0]

        # Sequences that are too short receive a near-zero composite score.
        if T < cfg.min_tokens:
            return QuRatingScores(
                writing_quality=0.0,
                educational_value=0.0,
                facts_score=0.0,
                expertise_score=0.0,
                composite=0.0,
            )

        # --- 1. Writing Quality (perplexity proxy) -------------------------
        # ppl = exp(-mean(log_probs))
        # writing_score = 1 - min(ppl / max_perplexity, 1.0)
        mean_lp = log_probs.float().mean().item()
        # Clamp to avoid overflow: exp(700) overflows float64
        clamped_neg_mean_lp = min(-mean_lp, math.log(cfg.max_perplexity) + 1.0)
        ppl = math.exp(clamped_neg_mean_lp)
        writing_quality = float(1.0 - min(ppl / cfg.max_perplexity, 1.0))

        # --- 2. Educational Value ------------------------------------------
        # Heuristic: fraction of tokens whose ID falls in a "high-information"
        # range.  We treat the upper half of the vocabulary as containing
        # rarer / more informative tokens (e.g. subword continuations of
        # technical terms, digits embedded in words, capitalised sequences).
        ids_int = token_ids.long()
        high_info_mask = ids_int >= (vocab_size // 2)
        academic_token_fraction = float(high_info_mask.float().mean().item())
        educational_value = float(min(academic_token_fraction * 2.0, 1.0))

        # --- 3. Facts Score ------------------------------------------------
        # Proxy: density of tokens that are likely numeric / digit-like.
        # Token IDs 15–58 span ASCII digits and punctuation in most BPE
        # vocabularies (concrete digit tokens live around IDs 15-27 in
        # tiktoken-style vocabularies, but we use a wider bracket for robustness).
        numeric_mask = (ids_int >= 15) & (ids_int <= 58)
        numeric_density = float(numeric_mask.float().mean().item())
        facts_score = float(min(numeric_density * 5.0, 1.0))

        # --- 4. Expertise Score --------------------------------------------
        # Vocabulary richness: unique tokens / total tokens (type-token ratio).
        unique_count = float(ids_int.unique().shape[0])
        unique_ratio = unique_count / T
        expertise_score = float(min(unique_ratio * 2.0, 1.0))

        # --- Composite -----------------------------------------------------
        composite = self._composite(
            writing_quality, educational_value, facts_score, expertise_score
        )

        return QuRatingScores(
            writing_quality=writing_quality,
            educational_value=educational_value,
            facts_score=facts_score,
            expertise_score=expertise_score,
            composite=composite,
        )

    def score_batch(
        self,
        token_ids: Tensor,  # (B, T)
        log_probs: Tensor,  # (B, T)
        attention_mask: Tensor,  # (B, T) — 1 for valid tokens, 0 for padding
    ) -> list[QuRatingScores]:
        """Score a batch of (possibly padded) documents.

        Parameters
        ----------
        token_ids:
            Integer tensor of shape ``(B, T)``.
        log_probs:
            Float tensor of shape ``(B, T)``.
        attention_mask:
            Binary tensor of shape ``(B, T)``; 1 = valid token, 0 = padding.

        Returns
        -------
        list[QuRatingScores]
            Length-B list, one :class:`QuRatingScores` per document.
        """
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be 2-D (B, T), got shape {tuple(token_ids.shape)}")
        B = token_ids.shape[0]
        results: list[QuRatingScores] = []

        for i in range(B):
            mask_i = attention_mask[i].bool()
            ids_i = token_ids[i][mask_i]
            lp_i = log_probs[i][mask_i]
            results.append(self.score_tokens(ids_i, lp_i))

        return results

    def select_top_k(
        self,
        scores: list[QuRatingScores],
        k: int,
        criterion: str = "composite",
    ) -> list[int]:
        """Return indices of the top-k documents sorted by *criterion*.

        Parameters
        ----------
        scores:
            List of :class:`QuRatingScores`, one per document.
        k:
            Number of documents to select.
        criterion:
            One of ``'composite'``, ``'writing'``, ``'educational'``,
            ``'facts'``, ``'expertise'``.

        Returns
        -------
        list[int]
            Indices into *scores*, in descending order of the chosen criterion.
        """
        _field_map: dict[str, int] = {
            "composite": 4,
            "writing": 0,
            "educational": 1,
            "facts": 2,
            "expertise": 3,
        }
        if criterion not in _field_map:
            raise ValueError(f"Unknown criterion {criterion!r}. Choose from {list(_field_map)!r}.")
        field_idx = _field_map[criterion]
        indexed = [(s[field_idx], i) for i, s in enumerate(scores)]
        indexed.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in indexed[:k]]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _composite(
        self,
        writing_quality: float,
        educational_value: float,
        facts_score: float,
        expertise_score: float,
    ) -> float:
        """Weighted average of the four QuRating dimension scores."""
        cfg = self.config
        total_weight = (
            cfg.writing_weight + cfg.educational_weight + cfg.facts_weight + cfg.expertise_weight
        )
        if total_weight == 0.0:
            return 0.0
        composite = (
            cfg.writing_weight * writing_quality
            + cfg.educational_weight * educational_value
            + cfg.facts_weight * facts_score
            + cfg.expertise_weight * expertise_score
        ) / total_weight
        return float(min(max(composite, 0.0), 1.0))
