"""Multi-Token Prediction (MTP) heads — v2.

Reference: Gloeckle et al. 2024 — "Better & Faster Large Language Models via
Multi-Token Prediction" (Meta FAIR).

Key idea: Train K independent output heads, each predicting a future token at
offset k (k = 1, 2, ..., K) from every sequence position.  At inference time
the K heads can be used as a built-in draft model for speculative decoding
without a separate draft network.

Compared to multi_token_prediction.py (v1) this file is a clean, self-contained
re-implementation that adds:
  - MTPConfig dataclass
  - MTPTrunk  (shared MLP applied before all heads when share_trunk=True)
  - MTPHead   (single Linear projection head)
  - MultiTokenPredictor  (orchestrates trunk + heads)
  - MTPLoss   (shifted cross-entropy averaged across K heads)
  - MTPDecoder (argmax draft for speculative decoding)

All classes use pure PyTorch — no flash-attn, no external deps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MTPConfig:
    """Configuration for the Multi-Token Predictor.

    Args:
        d_model:         Hidden size of the backbone transformer.
        vocab_size:      Vocabulary size (number of output classes).
        n_heads:         Number of future-token prediction heads (K).
        share_trunk:     When True a shared MTPTrunk MLP is applied to the
                         backbone hidden states before all K heads.
        trunk_expansion: Width multiplier for the trunk MLP hidden dim.
    """

    d_model: int
    vocab_size: int
    n_heads: int = 4
    share_trunk: bool = True
    trunk_expansion: int = 2


# ---------------------------------------------------------------------------
# MTPHead — one projection head for offset k
# ---------------------------------------------------------------------------


class MTPHead(nn.Module):
    """Single-layer linear projection from hidden states to vocabulary logits.

    Args:
        d_model:    Model hidden dimension.
        vocab_size: Vocabulary size.

    Shape:
        Input:  (B, T, d_model)
        Output: (B, T, vocab_size)
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: Tensor) -> Tensor:  # (B, T, d) -> (B, T, V)
        return self.proj(h)


# ---------------------------------------------------------------------------
# MTPTrunk — shared transformation applied before all K heads
# ---------------------------------------------------------------------------


class MTPTrunk(nn.Module):
    """Two-layer MLP with SiLU activation that transforms hidden states before
    they are passed to the K prediction heads.

    Architecture:
        Linear(d_model, d_model * expansion) -> SiLU -> Linear(d_model * expansion, d_model)

    The output has the **same shape** as the input so it can be used as a
    drop-in transformation before any head.

    Args:
        d_model:   Model hidden dimension.
        expansion: Width multiplier for the inner hidden dimension.

    Shape:
        Input:  (B, T, d_model)
        Output: (B, T, d_model)   [same as input]
    """

    def __init__(self, d_model: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # (B, T, d) -> (B, T, d)
        return self.fc2(F.silu(self.fc1(x)))


# ---------------------------------------------------------------------------
# MultiTokenPredictor — trunk + K heads
# ---------------------------------------------------------------------------


class MultiTokenPredictor(nn.Module):
    """Wraps an optional shared trunk and K independent prediction heads.

    Args:
        config: MTPConfig instance.

    Attributes:
        heads (nn.ModuleList): K MTPHead modules.
        trunk (MTPTrunk | None): Shared trunk when config.share_trunk is True.
    """

    def __init__(self, config: MTPConfig) -> None:
        super().__init__()
        self.config = config

        self.heads = nn.ModuleList(
            [MTPHead(config.d_model, config.vocab_size) for _ in range(config.n_heads)]
        )

        if config.share_trunk:
            self.trunk: MTPTrunk | None = MTPTrunk(config.d_model, config.trunk_expansion)
        else:
            self.trunk = None

    def forward(self, hidden: Tensor) -> list[Tensor]:
        """Run trunk (if present) then each head.

        Args:
            hidden: Backbone hidden states of shape (B, T, d_model).

        Returns:
            List of K tensors, each (B, T, vocab_size), one per head.
        """
        h = self.trunk(hidden) if self.trunk is not None else hidden
        return [head(h) for head in self.heads]


# ---------------------------------------------------------------------------
# MTPLoss — shifted cross-entropy averaged over K heads
# ---------------------------------------------------------------------------


class MTPLoss(nn.Module):
    """Cross-entropy loss averaged across K prediction heads.

    Head k (0-indexed) predicts the token at position t + k + 1, so its
    targets are ``labels[:, k+1:]`` and its logits are
    ``head_logits[k][:, :-k-1, :]``.

    Positions where ``labels == -100`` are masked out (standard HuggingFace
    convention).

    Args:
        n_heads: Number of heads K.

    Returns:
        Tuple of:
        - total_loss (scalar Tensor): Mean loss across all heads.
        - metrics (dict): Per-head losses plus "total_loss".
    """

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads

    def forward(
        self,
        head_logits: list[Tensor],  # list of K (B, T, V)
        labels: Tensor,  # (B, T) LongTensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        metrics: dict[str, Tensor] = {}
        total = torch.zeros((), dtype=head_logits[0].dtype, device=head_logits[0].device)

        for k, logits in enumerate(head_logits):
            shift = k + 1
            # logits: (B, T-shift, V)  targets: (B, T-shift)
            logits_k = logits[:, :-shift, :].contiguous()
            targets_k = labels[:, shift:].contiguous()

            # Flatten for F.cross_entropy
            # (B*(T-shift), V)  and  (B*(T-shift),)
            B, L, V = logits_k.shape
            loss_k = F.cross_entropy(
                logits_k.view(B * L, V),
                targets_k.view(B * L),
                ignore_index=-100,
            )
            metrics[f"head_{k}_loss"] = loss_k
            total = total + loss_k

        total = total / self.n_heads
        metrics["total_loss"] = total
        return total, metrics


# ---------------------------------------------------------------------------
# MTPDecoder — argmax draft tokens for speculative decoding
# ---------------------------------------------------------------------------


class MTPDecoder(nn.Module):
    """Helper that extracts draft token ids from a MultiTokenPredictor.

    Given the hidden state at the **last** position of the current context
    (shape ``(B, 1, d_model)``), each head predicts the next token at its
    respective offset.  Taking the argmax gives K draft token ids.

    Args:
        predictor: A trained MultiTokenPredictor.

    Usage::

        decoder = MTPDecoder(predictor)
        drafts = decoder.draft_tokens(last_hidden)  # (B, n_heads)
    """

    def __init__(self, predictor: MultiTokenPredictor) -> None:
        super().__init__()
        self.predictor = predictor

    @torch.no_grad()
    def draft_tokens(self, hidden: Tensor) -> Tensor:
        """Produce K draft token ids from a single hidden state.

        Args:
            hidden: Tensor of shape (B, 1, d_model) — the backbone hidden
                    state at the last token position.

        Returns:
            LongTensor of shape (B, n_heads) with argmax token ids in
            [0, vocab_size).
        """
        head_logits = self.predictor(hidden)  # list of K (B, 1, V)
        # For each head take the last (only) position and argmax
        drafts = torch.stack(
            [logits[:, -1, :].argmax(dim=-1) for logits in head_logits],
            dim=1,
        )  # (B, n_heads)
        return drafts
