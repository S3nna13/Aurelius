"""ColBERT late-interaction scoring (Khattab & Zaharia 2020, arXiv:2004.12832).

Late-interaction keeps per-token embeddings for queries and documents instead
of pooling them into a single vector. The relevance score is computed via
MaxSim: for each query token, take the maximum dot-product against any
document token, then sum across query tokens::

    score(q, d) = Σ_i max_j ⟨q_i, d_j⟩

This module implements the scoring architecture only. The encoder that
produces the per-token embeddings is pluggable; any module that yields a
``[N, D]`` tensor of token embeddings will do.

Pure torch, no foreign imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ColBERTConfig:
    """Configuration for :class:`ColBERTScorer`.

    Attributes:
        embed_dim: Dimensionality ``D`` of each per-token embedding. Used for
            shape validation only; the scorer does not own any parameters.
        normalize: If True, L2-normalize every token embedding before the
            MaxSim reduction. With normalization, dot-products equal cosine
            similarities, bounding MaxSim to ``[-Nq, Nq]``.
    """

    embed_dim: int = 32
    normalize: bool = True


def _l2_normalize(x: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


class ColBERTScorer:
    """MaxSim late-interaction scorer.

    The scorer is a pure function over embeddings; it holds no parameters.
    Gradients flow through ``score_*`` so the scorer can sit inside a training
    loop that fine-tunes the upstream encoder.
    """

    def __init__(self, config: ColBERTConfig) -> None:
        if config.embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {config.embed_dim}")
        self.config = config

    # ------------------------------------------------------------------ utils
    def _check_tokens(self, emb: Tensor, name: str, expect_rank: int) -> None:
        if emb.dim() != expect_rank:
            raise ValueError(f"{name} must have rank {expect_rank}, got shape {tuple(emb.shape)}")
        if emb.shape[-1] != self.config.embed_dim:
            raise ValueError(
                f"{name} last-dim must equal embed_dim={self.config.embed_dim}, got {emb.shape[-1]}"
            )

    def _maybe_normalize(self, emb: Tensor) -> Tensor:
        if self.config.normalize:
            return _l2_normalize(emb, dim=-1)
        return emb

    # ---------------------------------------------------------------- scoring
    def score_pair(self, q_embs: Tensor, d_embs: Tensor) -> float:
        """Score a single (query, doc) pair.

        Args:
            q_embs: ``[Nq, D]`` query token embeddings.
            d_embs: ``[Nd, D]`` document token embeddings.

        Returns:
            Python float: the MaxSim score.
        """
        self._check_tokens(q_embs, "q_embs", expect_rank=2)
        self._check_tokens(d_embs, "d_embs", expect_rank=2)
        if q_embs.shape[0] == 0:
            raise ValueError("q_embs must have at least one token (Nq > 0)")
        if d_embs.shape[0] == 0:
            raise ValueError("d_embs must have at least one token (Nd > 0)")

        q = self._maybe_normalize(q_embs)
        d = self._maybe_normalize(d_embs)
        # [Nq, Nd] dot-products
        sim = q @ d.transpose(0, 1)
        # MaxSim: max over doc tokens, then sum over query tokens
        per_q_max, _ = sim.max(dim=1)
        score = per_q_max.sum()
        return float(score.item())

    def score_batch(self, q_embs: Tensor, d_embs: Tensor) -> Tensor:
        """Score a batch of (query, doc) pairs in parallel.

        Args:
            q_embs: ``[B, Nq, D]``.
            d_embs: ``[B, Nd, D]``.

        Returns:
            Tensor ``[B]`` of MaxSim scores. Gradient-safe.
        """
        self._check_tokens(q_embs, "q_embs", expect_rank=3)
        self._check_tokens(d_embs, "d_embs", expect_rank=3)
        if q_embs.shape[0] != d_embs.shape[0]:
            raise ValueError(
                f"batch mismatch: q_embs B={q_embs.shape[0]} vs d_embs B={d_embs.shape[0]}"
            )
        if q_embs.shape[1] == 0:
            raise ValueError("q_embs must have at least one token (Nq > 0)")
        if d_embs.shape[1] == 0:
            raise ValueError("d_embs must have at least one token (Nd > 0)")

        q = self._maybe_normalize(q_embs)
        d = self._maybe_normalize(d_embs)
        # [B, Nq, Nd]
        sim = torch.matmul(q, d.transpose(1, 2))
        per_q_max, _ = sim.max(dim=2)  # [B, Nq]
        return per_q_max.sum(dim=1)  # [B]

    def score_query_against_corpus(self, q_embs: Tensor, corpus_embs: list[Tensor]) -> list[float]:
        """Score one query against a corpus of docs with variable-length tokens.

        Args:
            q_embs: ``[Nq, D]``.
            corpus_embs: list of ``[Nd_i, D]`` per-doc token tensors. Doc
                token counts may differ, so we do not batch-pad here; we
                loop per doc. For corpora that fit in memory with uniform
                ``Nd``, prefer :meth:`score_batch`.

        Returns:
            List of Python floats, same length as ``corpus_embs``.
        """
        self._check_tokens(q_embs, "q_embs", expect_rank=2)
        if q_embs.shape[0] == 0:
            raise ValueError("q_embs must have at least one token (Nq > 0)")

        q = self._maybe_normalize(q_embs)
        scores: list[float] = []
        for i, d_embs in enumerate(corpus_embs):
            self._check_tokens(d_embs, f"corpus_embs[{i}]", expect_rank=2)
            if d_embs.shape[0] == 0:
                raise ValueError(f"corpus_embs[{i}] must have at least one token (Nd > 0)")
            d = self._maybe_normalize(d_embs)
            sim = q @ d.transpose(0, 1)
            per_q_max, _ = sim.max(dim=1)
            scores.append(float(per_q_max.sum().item()))
        return scores


__all__ = ["ColBERTConfig", "ColBERTScorer"]
