"""Aurelius — SALMON: Self-Alignment with Instructable Reward Models.

Reference: Sun et al. (2023) "SALMON: Self-Alignment with Instructable Reward
Models", arXiv:2310.05910.

Algorithm:
1. A Principle-Conditioned Reward Model (PCRM) scores (principle, prompt,
   response) triples using a pure-PyTorch embedding/cosine proxy reward.
2. SALMONScorer aggregates scores across a list of principles.
3. SALMONFilter selects the best candidate response via Best-of-N scoring.
4. SALMONLoss computes SFT NLL on the winner plus an optional contrastive
   penalty on the losers.

Variable names follow the paper notation where possible:
  pi_i  — principle i
  r     — reward scalar
  y*    — winner (best-scoring) response
  L     — training loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VOCAB_SIZE: int = 512  # proxy vocab for the embedding-based reward
_EMBED_DIM: int = 64  # embedding dimension for the proxy RM


# ---------------------------------------------------------------------------
# Principle-Conditioned Reward Model (PCRM)
# ---------------------------------------------------------------------------


class PrincipleConditionedRM(nn.Module):
    """Proxy principle-conditioned reward model (PCRM).

    Scores a (principle, prompt, response) triple via embedding cosine
    similarity.  No external LM is needed.

    The principle text is hashed to token ids and embedded; prompt and
    response tokens are embedded independently.  The reward is the mean
    cosine similarity between the principle embedding and the response
    token embeddings, clamped to [-1, 1].

    Args:
        vocab_size: Size of the token vocabulary for prompt/response.
        embed_dim:  Dimensionality of all embeddings.
        principle_vocab_size: Hash modulus used to map principle characters
            to token ids.
    """

    def __init__(
        self,
        vocab_size: int = _VOCAB_SIZE,
        embed_dim: int = _EMBED_DIM,
        principle_vocab_size: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.principle_vocab_size = principle_vocab_size

        # Token embedding for prompt + response tokens
        self.token_embed: nn.Embedding = nn.Embedding(vocab_size, embed_dim)

        # Character-level embedding for principle text
        self.principle_embed: nn.Embedding = nn.Embedding(principle_vocab_size, embed_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _principle_to_ids(self, principle: str) -> torch.Tensor:
        """Convert a principle string to integer ids via character hashing."""
        if not principle:
            return torch.zeros(1, dtype=torch.long)
        ids = [ord(c) % self.principle_vocab_size for c in principle]
        return torch.tensor(ids, dtype=torch.long)

    def _mean_embed(self, ids: torch.Tensor, table: nn.Embedding) -> torch.Tensor:
        """Return the mean embedding vector for a sequence of ids.

        Args:
            ids: 1-D LongTensor of length L.  Empty tensors are allowed and
                 will produce a zero vector.

        Returns:
            Tensor of shape (embed_dim,).
        """
        if ids.numel() == 0:
            return torch.zeros(self.embed_dim, device=table.weight.device)
        vecs = table(ids)  # (L, embed_dim)
        return vecs.mean(dim=0)  # (embed_dim,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        principle: str,
        prompt_tokens: torch.Tensor,
        response_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the PCRM scalar reward r(pi_i, x, y).

        Args:
            principle:       Natural-language principle string pi_i.
            prompt_tokens:   1-D LongTensor of prompt token ids (x).
            response_tokens: 1-D LongTensor of response token ids (y).

        Returns:
            Scalar float tensor r ∈ [-1, 1].

        Raises:
            ValueError: If prompt_tokens or response_tokens are not 1-D.
        """
        if prompt_tokens.dim() != 1:
            raise ValueError(f"prompt_tokens must be 1-D, got shape {prompt_tokens.shape}")
        if response_tokens.dim() != 1:
            raise ValueError(f"response_tokens must be 1-D, got shape {response_tokens.shape}")

        pi_ids = self._principle_to_ids(principle).to(self.principle_embed.weight.device)
        pi_emb = self._mean_embed(pi_ids, self.principle_embed)  # (D,)
        resp_emb = self._mean_embed(response_tokens, self.token_embed)  # (D,)

        # Normalised dot product (cosine similarity) as the proxy reward r
        r = F.cosine_similarity(pi_emb.unsqueeze(0), resp_emb.unsqueeze(0))  # (1,)
        return r.squeeze(0)  # scalar


# ---------------------------------------------------------------------------
# SALMON Scorer
# ---------------------------------------------------------------------------


class SALMONScorer(nn.Module):
    """Aggregates PCRM scores across a principle set {pi_1, …, pi_K}.

    Args:
        principles: List of K natural-language principle strings.
        vocab_size: Forwarded to :class:`PrincipleConditionedRM`.
        embed_dim:  Forwarded to :class:`PrincipleConditionedRM`.
    """

    def __init__(
        self,
        principles: list[str],
        vocab_size: int = _VOCAB_SIZE,
        embed_dim: int = _EMBED_DIM,
    ) -> None:
        super().__init__()
        if not principles:
            raise ValueError("principles list must be non-empty")
        self.principles: list[str] = list(principles)
        self.pcrm = PrincipleConditionedRM(vocab_size=vocab_size, embed_dim=embed_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        prompt_tokens: torch.Tensor,
        response_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Score a (prompt, response) pair under every principle.

        Args:
            prompt_tokens:   1-D LongTensor (x).
            response_tokens: 1-D LongTensor (y).

        Returns:
            Tensor of shape (K,) containing r(pi_i, x, y) for each principle.
        """
        scores = [self.pcrm.score(pi, prompt_tokens, response_tokens) for pi in self.principles]
        return torch.stack(scores)  # (K,)

    def aggregate_score(
        self,
        prompt_tokens: torch.Tensor,
        response_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mean SALMON score across all principles.

        Equation from Section 3.2 of the paper:
            score(x, y) = (1/K) * Σ_i  r(pi_i, x, y)

        Returns:
            Scalar float tensor.
        """
        return self.score(prompt_tokens, response_tokens).mean()


# ---------------------------------------------------------------------------
# SALMON Filter — Best-of-N
# ---------------------------------------------------------------------------


class SALMONFilter(nn.Module):
    """Selects the best response from N candidates using SALMON scoring.

    Args:
        scorer: A :class:`SALMONScorer` instance.
    """

    def __init__(self, scorer: SALMONScorer) -> None:
        super().__init__()
        self.scorer = scorer

    def select_best(
        self,
        prompt_tokens: torch.Tensor,
        candidates: list[torch.Tensor],
    ) -> tuple[int, torch.Tensor]:
        """Best-of-N selection.

        Args:
            prompt_tokens: 1-D LongTensor of prompt token ids (x).
            candidates:    List of N 1-D LongTensors {y_1, …, y_N}.

        Returns:
            Tuple (best_idx, scores) where:
                best_idx — integer index of y* = argmax_n score(x, y_n).
                scores   — Tensor of shape (N,) of aggregate scores.

        Raises:
            ValueError: If candidates is empty.
        """
        if not candidates:
            raise ValueError("candidates list must be non-empty")

        agg_scores = torch.stack(
            [self.scorer.aggregate_score(prompt_tokens, y) for y in candidates]
        )  # (N,)
        best_idx = int(agg_scores.argmax().item())
        return best_idx, agg_scores


# ---------------------------------------------------------------------------
# SALMON Loss
# ---------------------------------------------------------------------------


class SALMONLoss(nn.Module):
    """SFT NLL on the winner response with optional contrastive penalty.

    The primary term is the supervised fine-tuning negative log-likelihood
    on the SALMON-selected winner y*:

        L_SFT = -log P(y* | x)

    An optional contrastive term penalises the losers:

        L_contrast = (alpha / M) * Σ_j  max(0, log P(y_j | x) - margin)

    Total loss: L = L_SFT + alpha * L_contrast

    Args:
        alpha:  Weight for the contrastive penalty.  Set to 0 to disable.
        margin: Margin used in the contrastive hinge penalty.
    """

    def __init__(self, alpha: float = 0.1, margin: float = 0.0) -> None:
        super().__init__()
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.alpha = alpha
        self.margin = margin

    def forward(
        self,
        log_probs_winner: torch.Tensor,
        log_probs_losers: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the SALMON training loss.

        Args:
            log_probs_winner: Scalar or 1-D tensor representing
                              log P(y* | x) — the per-token log-probs of the
                              winner, already summed/meaned to a single value.
            log_probs_losers: List of scalars/1-D tensors for each loser.

        Returns:
            Scalar loss tensor with gradient.

        Raises:
            ValueError: If log_probs_winner is empty.
        """
        if log_probs_winner.numel() == 0:
            raise ValueError("log_probs_winner must not be empty")

        # Reduce to scalar if 1-D
        lp_winner = log_probs_winner.mean() if log_probs_winner.dim() > 0 else log_probs_winner

        # L_SFT = -E[log P(y* | x)]
        L_sft = -lp_winner

        if self.alpha == 0.0 or not log_probs_losers:
            return L_sft

        # Contrastive penalty: hinge on each loser exceeding margin
        loser_terms = []
        for lp_loser in log_probs_losers:
            lp_j = lp_loser.mean() if lp_loser.dim() > 0 else lp_loser
            loser_terms.append(F.relu(lp_j - self.margin))

        L_contrast = torch.stack(loser_terms).mean()
        return L_sft + self.alpha * L_contrast
