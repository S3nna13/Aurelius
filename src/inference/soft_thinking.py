"""Soft Thinking — differentiable probabilistic token embedding mixing (2025).

During reasoning/thinking, instead of hard-sampling a discrete token, compute a
*soft* next-token embedding as a weighted sum of the top-k token embeddings
weighted by their softmax probabilities. This keeps the computation
differentiable through the entire reasoning chain.

Reference: Soft Thinking (2025) — probabilistic token mixing for LLM reasoning.
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
class SoftThinkingConfig:
    d_model: int = 2048
    vocab_size: int = 128000
    top_k: int = 50            # number of tokens to mix
    temperature: float = 1.0   # applied to logits before softmax
    renormalize: bool = True   # renormalize probabilities after top-k selection


# ---------------------------------------------------------------------------
# Core module
# ---------------------------------------------------------------------------

class SoftThinkingMixer(nn.Module):
    """Computes soft token embeddings from logits + embedding table.

    At each soft thinking step:
    1. Given logits [B, vocab_size], compute probs = softmax(logits / temperature)
    2. Select top-k tokens by probability
    3. Optionally renormalize the top-k weights so they sum to 1
    4. Soft embedding = Σ_i w_i * embed(token_i)  — weighted sum of embeddings

    The soft embedding can be used directly as input to the next transformer
    step, keeping the reasoning chain fully differentiable.
    """

    def __init__(
        self,
        config: SoftThinkingConfig,
        embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        if embedding is None:
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        else:
            self.embedding = embedding

    # ------------------------------------------------------------------
    # Core mixing logic
    # ------------------------------------------------------------------

    def mix(
        self,
        logits: Tensor,
        embedding_weight: Tensor | None = None,
    ) -> Tensor:
        """Compute soft token embeddings from logits.

        Args:
            logits: [B, vocab_size] or [B, T, vocab_size]
            embedding_weight: [vocab_size, d_model] — if None, uses
                self.embedding.weight

        Returns:
            soft_embedding: [B, d_model] or [B, T, d_model]
        """
        cfg = self.config
        emb_weight = embedding_weight if embedding_weight is not None else self.embedding.weight

        is_2d = logits.dim() == 2  # [B, V]

        if is_2d:
            # Expand to [B, 1, V] for unified handling
            logits = logits.unsqueeze(1)  # [B, 1, V]

        # logits: [B, T, V]
        B, T, V = logits.shape
        k = min(cfg.top_k, V)

        # 1. Scale by temperature
        scaled = logits / cfg.temperature  # [B, T, V]

        # 2. Compute softmax probabilities
        probs = F.softmax(scaled, dim=-1)  # [B, T, V]

        # 3. top-k by probability
        topk_weights, topk_indices = torch.topk(probs, k=k, dim=-1)
        # topk_weights: [B, T, k],  topk_indices: [B, T, k]

        # 4. Optionally renormalize
        if cfg.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 5. Gather embedding vectors for top-k indices
        #    emb_weight: [V, d_model]
        #    topk_indices flat: [B*T*k] → gather embeddings → reshape [B, T, k, d_model]
        flat_indices = topk_indices.reshape(-1)          # [B*T*k]
        flat_embeddings = emb_weight[flat_indices]       # [B*T*k, d_model]
        d = emb_weight.shape[-1]
        embeddings = flat_embeddings.reshape(B, T, k, d)  # [B, T, k, d_model]

        # 6. Weighted sum: (weights.unsqueeze(-1) * embeddings).sum(-2)
        soft_emb = (topk_weights.unsqueeze(-1) * embeddings).sum(dim=-2)  # [B, T, d_model]

        if is_2d:
            soft_emb = soft_emb.squeeze(1)  # [B, d_model]

        return soft_emb

    def forward(self, logits: Tensor) -> Tensor:
        """Alias for mix() using self.embedding.weight."""
        return self.mix(logits)

    # ------------------------------------------------------------------
    # Entropy helper
    # ------------------------------------------------------------------

    def entropy(self, logits: Tensor) -> Tensor:
        """Compute Shannon entropy of the full distribution (before top-k).

        Args:
            logits: [B, vocab_size] or [B, T, vocab_size]

        Returns:
            entropy: [B] or [B, T]  — scalar entropy per position
        """
        cfg = self.config
        scaled = logits / cfg.temperature
        probs = F.softmax(scaled, dim=-1)
        # H = -Σ p * log(p),  clamp to avoid log(0)
        log_probs = torch.log(probs.clamp(min=1e-10))
        return -(probs * log_probs).sum(dim=-1)
