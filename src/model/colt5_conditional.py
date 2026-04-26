"""CoLT5 Conditional Computation — light/heavy token routing for efficient FFN.

Reference: Ainslie et al. 2023 — "CoLT5: Faster Long-Range Transformers with
Conditional Computation" (https://arxiv.org/abs/2303.09752)

Architecture overview:
    CoLT5 assigns different computation budgets to different tokens. A learned
    router scores each position and routes the top-k highest-scoring tokens
    through an expensive "heavy" FFN path; all tokens pass through a cheap
    "light" FFN path, and the heavy output is *added* on top of the light
    output for selected tokens.

Light / Heavy FFN split:
    light_ffn:  Linear(d_model → d_ff_light) + GELU + Linear(d_ff_light → d_model)
    heavy_ffn:  Linear(d_model → d_ff_heavy) + GELU + Linear(d_ff_heavy → d_model)

Routing:
    router_score = Linear(d_model → 1, bias=False)
    k            = max(1, int(T * heavy_fraction))
    top_k_idx    = topk(router_score, k)           # per-batch, per-sequence
    output       = light_out
    output[top_k] += heavy_out                     # scatter_add blend

CoLT5Block wraps the FFN in a pre-norm residual:
    out = x + CoLT5FFN(RMSNorm(x))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CoLT5Config:
    """Hyperparameters for the CoLT5 conditional computation module."""

    d_model: int = 2048
    d_ff_light: int = 512  # hidden dim for the light (cheap) FFN path
    d_ff_heavy: int = 2048  # hidden dim for the heavy (expensive) FFN path
    heavy_fraction: float = 0.25  # fraction of tokens routed to the heavy path
    n_heavy_heads: int = 8  # heavy attention heads (reserved for future use)
    n_light_heads: int = 2  # light attention heads (reserved for future use)
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# CoLT5FFN — conditional light/heavy FFN with top-k routing
# ---------------------------------------------------------------------------


class CoLT5FFN(nn.Module):
    """Conditional FFN that applies a heavy path only to the top-k tokens.

    All tokens receive the light FFN output; the top-k highest-scoring tokens
    *additionally* receive the heavy FFN output summed on top, implementing
    the blending from Section 3.1 of the CoLT5 paper.

    Args:
        config: ``CoLT5Config`` instance.

    Forward signature::

        (output, routing_scores) = ffn(x)

    where ``output`` has shape ``[B, T, d_model]`` and ``routing_scores`` has
    shape ``[B, T]``.
    """

    def __init__(self, config: CoLT5Config) -> None:
        super().__init__()
        self.config = config

        # Router: scalar score per token
        self.router = nn.Linear(config.d_model, 1, bias=False)

        # Light FFN (cheap — runs for every token)
        self.light_fc1 = nn.Linear(config.d_model, config.d_ff_light, bias=False)
        self.light_fc2 = nn.Linear(config.d_ff_light, config.d_model, bias=False)

        # Heavy FFN (expensive — runs only for top-k tokens)
        self.heavy_fc1 = nn.Linear(config.d_model, config.d_ff_heavy, bias=False)
        self.heavy_fc2 = nn.Linear(config.d_ff_heavy, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    # ------------------------------------------------------------------
    # Light and heavy sub-networks
    # ------------------------------------------------------------------

    def _light_ffn(self, x: Tensor) -> Tensor:
        """Light path: [B, T, d_model] → [B, T, d_model]."""
        return self.dropout(self.light_fc2(F.gelu(self.light_fc1(x))))

    def _heavy_ffn(self, x: Tensor) -> Tensor:
        """Heavy path: [B, k, d_model] → [B, k, d_model]."""
        return self.dropout(self.heavy_fc2(F.gelu(self.heavy_fc1(x))))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply conditional light/heavy FFN.

        Args:
            x: Input tensor of shape ``[B, T, d_model]``.

        Returns:
            Tuple of:
                * ``output``         — ``[B, T, d_model]``
                * ``routing_scores`` — ``[B, T]`` (raw router logits, no softmax)
        """
        B, T, d = x.shape

        # 1. Compute per-token routing scores
        scores = self.router(x).squeeze(-1)  # [B, T]

        # 2. Determine k — number of heavy tokens
        k = max(1, int(T * self.config.heavy_fraction))
        k = min(k, T)  # safety clamp

        # 3. Select top-k token indices (per batch item)
        top_k_idx = scores.topk(k, dim=1).indices  # [B, k]

        # 4. Light path — applied to all tokens
        light_out = self._light_ffn(x)  # [B, T, d_model]

        # 5. Gather the top-k token embeddings for the heavy path
        idx_expanded = top_k_idx.unsqueeze(-1).expand(B, k, d)  # [B, k, d_model]
        heavy_in = x.gather(1, idx_expanded)  # [B, k, d_model]

        # 6. Heavy path — applied only to the top-k tokens
        heavy_out = self._heavy_ffn(heavy_in)  # [B, k, d_model]

        # 6b. Gate heavy output by sigmoid(score) so router receives gradients.
        #     Gather the top-k scores, apply sigmoid, and broadcast over d_model.
        top_k_scores = scores.gather(1, top_k_idx)  # [B, k]
        gate = torch.sigmoid(top_k_scores).unsqueeze(-1)  # [B, k, 1]
        heavy_out = heavy_out * gate  # [B, k, d_model]

        # 7. Blend: add heavy output onto light output at the top-k positions
        output = light_out.scatter_add(1, idx_expanded, heavy_out)  # [B, T, d_model]

        return output, scores

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def routing_stats(self, routing_scores: Tensor) -> dict[str, float]:
        """Compute diagnostic statistics for a batch of routing scores.

        Args:
            routing_scores: Raw router logits, shape ``[B, T]``.

        Returns:
            Dictionary with keys:

            * ``heavy_fraction_actual`` — fraction of tokens selected as heavy
              (equal to the configured fraction by construction, included for
              external monitoring convenience).
            * ``score_entropy`` — mean normalised Shannon entropy of the
              softmax-ed scores across the batch, measuring routing uncertainty
              (high → uniform routing, low → peaked/confident routing).
        """
        B, T = routing_scores.shape
        k = max(1, int(T * self.config.heavy_fraction))
        k = min(k, T)
        heavy_fraction_actual = float(k) / float(T)

        # Normalised entropy: H / log(T) so range is always [0, 1]
        probs = torch.softmax(routing_scores.float(), dim=-1)  # [B, T]
        log_probs = torch.log(probs + 1e-9)
        entropy = -(probs * log_probs).sum(dim=-1)  # [B]
        max_entropy = math.log(T) if T > 1 else 1.0
        score_entropy = float((entropy / max_entropy).mean().item())

        return {
            "heavy_fraction_actual": heavy_fraction_actual,
            "score_entropy": score_entropy,
        }


# ---------------------------------------------------------------------------
# CoLT5Block — pre-norm residual wrapper around CoLT5FFN
# ---------------------------------------------------------------------------


class CoLT5Block(nn.Module):
    """Full CoLT5 block: pre-norm residual + conditional light/heavy FFN.

    Applies RMSNorm before the FFN and adds a residual connection::

        normed  = RMSNorm(x)
        ffn_out, routing_scores = CoLT5FFN(normed)
        output  = x + ffn_out

    Args:
        config: ``CoLT5Config`` instance.

    Forward signature::

        {"output": Tensor[B, T, d_model], "routing_scores": Tensor[B, T]}
    """

    def __init__(self, config: CoLT5Config) -> None:
        super().__init__()
        self.config = config

        # Pre-norm: RMSNorm (eps=1e-6, no bias)
        self.norm = _RMSNorm(config.d_model)

        # Conditional FFN
        self.colt5_ffn = CoLT5FFN(config)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Apply pre-norm residual + CoLT5 conditional FFN.

        Args:
            x: Input tensor of shape ``[B, T, d_model]``.

        Returns:
            Dictionary with keys:

            * ``"output"``         — ``[B, T, d_model]``
            * ``"routing_scores"`` — ``[B, T]``
        """
        ffn_out, routing_scores = self.colt5_ffn(self.norm(x))
        output = x + ffn_out
        return {"output": output, "routing_scores": routing_scores}


# ---------------------------------------------------------------------------
# Minimal RMSNorm (local, avoids cross-module import)
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight.to(x.dtype)


# ---------------------------------------------------------------------------
# Registry — must be imported *after* MODEL_COMPONENT_REGISTRY is created
# ---------------------------------------------------------------------------


def _register() -> None:
    """Register CoLT5 components in MODEL_COMPONENT_REGISTRY."""
    try:
        from src.model import MODEL_COMPONENT_REGISTRY  # noqa: PLC0415

        MODEL_COMPONENT_REGISTRY["colt5_ffn"] = CoLT5FFN
        MODEL_COMPONENT_REGISTRY["colt5_block"] = CoLT5Block
    except ImportError:
        # Graceful fallback when running outside the full package (e.g. tests
        # that import this file directly before the package is installed).
        pass


_register()
