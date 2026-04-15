"""Prefix Tuning (Li & Liang, 2021) with optional reparameterization MLP.

Prepends learnable continuous "soft prompt" tokens to the input embeddings,
keeping the base model frozen. Only the prefix parameters are trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PrefixConfig:
    """Configuration for prefix tuning."""

    prefix_length: int = 10
    """Number of learnable prefix tokens to prepend."""

    d_model: int = 512
    """Embedding / hidden dimension of the model."""

    n_layers: int = 12
    """Number of transformer layers (informational; used by callers)."""

    init_from_vocab: bool = False
    """If True, hint to initialise prefix from vocabulary (not used internally
    here; callers may honour it when they have access to an embedding table)."""

    reparam_hidden_dim: Optional[int] = None
    """If set, use a reparameterization MLP with this hidden size instead of a
    direct nn.Parameter.  The MLP maps a (prefix_length,)-shaped index
    embedding through a two-layer network to produce (prefix_length, d_model)
    prefix embeddings."""


# ---------------------------------------------------------------------------
# PrefixEmbedding
# ---------------------------------------------------------------------------

class PrefixEmbedding(nn.Module):
    """Learnable prefix token embeddings.

    Two modes:
    - Direct: ``nn.Parameter`` of shape ``(prefix_length, d_model)``.
    - Reparameterized: 2-layer MLP that maps a fixed index input to the prefix
      embeddings.  The intermediate dimension is ``reparam_hidden_dim``.
    """

    def __init__(self, config: PrefixConfig) -> None:
        super().__init__()
        self.config = config
        P = config.prefix_length
        D = config.d_model

        if config.reparam_hidden_dim is not None:
            H = config.reparam_hidden_dim
            # Fixed input indices (not learnable) — the MLP is what we train.
            self.register_buffer("_input", torch.arange(P, dtype=torch.float32))
            self.mlp = nn.Sequential(
                nn.Linear(1, H),
                nn.Tanh(),
                nn.Linear(H, D),
            )
            self._use_reparam = True
        else:
            self.prefix = nn.Parameter(torch.empty(P, D).normal_(0.0, 0.02))
            self._use_reparam = False

    def forward(self) -> Tensor:
        """Return prefix embeddings of shape ``(prefix_length, d_model)``."""
        if self._use_reparam:
            # _input: (P,) -> (P, 1) -> MLP -> (P, D)
            x = self._input.unsqueeze(-1)   # (P, 1)
            return self.mlp(x)              # (P, D)
        return self.prefix                  # (P, D)


# ---------------------------------------------------------------------------
# prepend_prefix
# ---------------------------------------------------------------------------

def prepend_prefix(prefix: Tensor, input_embeds: Tensor) -> Tensor:
    """Concatenate prefix tokens before input embeddings.

    Args:
        prefix:       ``(P, D)`` prefix embeddings (single, will be broadcast).
        input_embeds: ``(B, T, D)`` token embeddings.

    Returns:
        ``(B, P + T, D)`` combined embeddings.
    """
    B = input_embeds.shape[0]
    # Broadcast prefix across the batch dimension
    prefix_expanded = prefix.unsqueeze(0).expand(B, -1, -1)  # (B, P, D)
    return torch.cat([prefix_expanded, input_embeds], dim=1)  # (B, P+T, D)


# ---------------------------------------------------------------------------
# PrefixTuner
# ---------------------------------------------------------------------------

class PrefixTuner(nn.Module):
    """Wraps a backbone embedding and prepends a learnable prefix.

    The backbone embedding is frozen; only the prefix parameters are trainable.
    """

    def __init__(self, backbone_embed: nn.Embedding, prefix_config: PrefixConfig) -> None:
        super().__init__()
        self.backbone_embed = backbone_embed
        self.prefix_config = prefix_config

        # Freeze backbone embedding
        for param in backbone_embed.parameters():
            param.requires_grad = False

        # Learnable prefix
        self.prefix_embedding = PrefixEmbedding(prefix_config)

    def get_prefix_embeds(self) -> Tensor:
        """Return current prefix embeddings of shape ``(prefix_length, d_model)``."""
        return self.prefix_embedding()

    def forward(self, token_ids: Tensor) -> Tensor:
        """Embed tokens and prepend the prefix.

        Args:
            token_ids: ``(B, T)`` integer token IDs.

        Returns:
            ``(B, prefix_length + T, d_model)`` embeddings.
        """
        token_embeds = self.backbone_embed(token_ids)  # (B, T, D)
        prefix = self.get_prefix_embeds()              # (P, D)
        return prepend_prefix(prefix, token_embeds)    # (B, P+T, D)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return only the prefix parameters (backbone is frozen)."""
        return list(self.prefix_embedding.parameters())


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def freeze_model(model: nn.Module) -> int:
    """Freeze all parameters in *model*.

    Args:
        model: Any ``nn.Module``.

    Returns:
        Number of parameter tensors frozen.
    """
    count = 0
    for param in model.parameters():
        param.requires_grad = False
        count += 1
    return count


def count_trainable_params(model: nn.Module) -> int:
    """Count the number of parameter tensors with ``requires_grad=True``.

    Args:
        model: Any ``nn.Module``.

    Returns:
        Number of trainable parameter tensors.
    """
    return sum(1 for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# PromptTuningConfig / SoftPrompt
# ---------------------------------------------------------------------------

@dataclass
class PromptTuningConfig:
    """Minimal configuration for simple soft-prompt tuning."""

    n_tokens: int = 20
    """Number of learnable soft-prompt tokens."""

    d_model: int = 512
    """Model embedding dimension."""

    init_text: Optional[str] = None
    """Optional text hint for initialisation (not used internally)."""


class SoftPrompt(nn.Module):
    """Simple learnable soft-prompt parameter.

    Stores a single ``nn.Parameter`` of shape ``(n_tokens, d_model)`` and
    returns it from ``forward()``.
    """

    def __init__(self, config: PromptTuningConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.Parameter(
            torch.empty(config.n_tokens, config.d_model).normal_(0.0, 0.02)
        )

    def forward(self) -> Tensor:
        """Return soft-prompt embeddings of shape ``(n_tokens, d_model)``."""
        return self.embeddings
