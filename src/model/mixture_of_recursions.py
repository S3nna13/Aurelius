"""Mixture of Recursions (MoR): per-token adaptive recursion depth.

Each token independently decides how many times to apply the same shared
transformer block.  A lightweight router predicts a depth distribution over
{1 … max_depth}; during training a Gumbel-softmax relaxation keeps the
operation differentiable, while at inference the argmax depth is used.

This differs from Mixture-of-Depths (token dropping) and Universal
Transformers (fixed uniform depth) by routing *compute* rather than
*skipping tokens*.

Reference design motivation:
    Mixture of Recursions — adaptive per-token depth via weight-shared blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Small building-block helpers (no external project imports needed)
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no bias)."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class _MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1)  # each (B, T, C)

        # Reshape to (B, n_heads, T, head_dim)
        def _reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = _reshape(q), _reshape(k), _reshape(v)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class _FFN(nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(self, d_model: int, expansion: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * expansion, bias=False)
        self.fc2 = nn.Linear(d_model * expansion, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# RecursionDepthRouter
# ---------------------------------------------------------------------------


class RecursionDepthRouter(nn.Module):
    """Route each token to a recursion depth in {0 … max_depth-1}.

    During training, depths are sampled via Gumbel-softmax (straight-through
    hard samples when ``hard=True`` in training calls).  During evaluation the
    argmax depth is returned directly.

    Args:
        d_model:   Token hidden dimension.
        max_depth: Number of possible depth levels (1-indexed computationally,
                   0-indexed as logit classes).
    """

    def __init__(self, d_model: int, max_depth: int = 4) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.proj = nn.Linear(d_model, max_depth, bias=True)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict per-token recursion depths.

        Args:
            x: Hidden states of shape ``(B, T, d_model)``.

        Returns:
            depths: ``(B, T)`` int64 tensor with depth index in
                    ``[0, max_depth-1]``.
            probs:  ``(B, T, max_depth)`` softmax probability distribution.
        """
        logits = self.proj(x)  # (B, T, max_depth)

        if self.training:
            # Gumbel-softmax: differentiable discrete sampling
            probs = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)

        depths = probs.argmax(dim=-1)  # (B, T) — hard decision in both modes
        return depths, probs

    # ------------------------------------------------------------------
    def expected_depth(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute expected depth as a weighted sum over depth levels.

        Args:
            probs: ``(B, T, max_depth)`` probability distribution.

        Returns:
            ``(B, T)`` tensor of expected depth values in
            ``[0, max_depth-1]``.
        """
        depth_vals = torch.arange(
            self.max_depth, dtype=probs.dtype, device=probs.device
        )  # (max_depth,)
        # Broadcast: (B, T, max_depth) * (max_depth,) → sum over last dim
        return (probs * depth_vals).sum(dim=-1)  # (B, T)


# ---------------------------------------------------------------------------
# RecursiveTransformerBlock
# ---------------------------------------------------------------------------


class RecursiveTransformerBlock(nn.Module):
    """A single shared transformer block (attention + FFN) applied N times.

    The same weights are reused on every recursive application, analogous to
    Universal Transformers, but here the number of applications varies per
    token.

    Args:
        d_model:  Hidden dimension.
        n_heads:  Number of attention heads.
        max_depth: Maximum depth (used for ``apply_with_depths`` soft mode).
    """

    def __init__(self, d_model: int, n_heads: int, max_depth: int = 4) -> None:
        super().__init__()
        self.max_depth = max_depth

        self.norm1 = _RMSNorm(d_model)
        self.attn = _MultiHeadSelfAttention(d_model, n_heads)
        self.norm2 = _RMSNorm(d_model)
        self.ffn = _FFN(d_model)

    # ------------------------------------------------------------------
    def _apply_once(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one residual attention + FFN pass."""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    # ------------------------------------------------------------------
    def apply_n_times(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Apply the shared block exactly ``n`` times to all tokens.

        Args:
            x: ``(B, T, d_model)``
            n: Number of applications (must be >= 1).

        Returns:
            ``(B, T, d_model)`` after n recursive applications.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        for _ in range(n):
            x = self._apply_once(x)
        return x

    # ------------------------------------------------------------------
    def apply_with_depths(
        self,
        x: torch.Tensor,
        depths: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        mode: str = "soft",
    ) -> torch.Tensor:
        """Apply variable depths per token.

        Two modes:
        - ``"soft"``:  Differentiable weighted combination of outputs at each
                        depth level (1 … max_depth applications).  Requires
                        ``probs`` to be provided.
        - ``"hard"``:  Route each token to exactly its assigned depth.  Uses
                        ``depths`` (int) to select which application count to
                        use per token.

        Args:
            x:      ``(B, T, d_model)``
            depths: ``(B, T)`` int64 depth indices in ``[0, max_depth-1]``.
            probs:  ``(B, T, max_depth)`` probability weights (required for
                    soft mode).
            mode:   ``"soft"`` (default) or ``"hard"``.

        Returns:
            ``(B, T, d_model)`` output.
        """
        if mode == "soft":
            if probs is None:
                raise ValueError("probs must be provided for soft mode")
            return self._soft_apply(x, probs)
        return self._hard_apply(x, depths)

    def _soft_apply(
        self, x: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        """Weighted combination across all depth levels.

        Computes hidden states h_d for d = 1 … max_depth, then returns
        sum_d probs[d] * h_d  (per token).
        """
        B, T, C = x.shape
        # Accumulate: list of (B, T, C) tensors, one per depth level
        outputs = []
        cur = x
        for _ in range(self.max_depth):
            cur = self._apply_once(cur)
            outputs.append(cur)  # depth d+1 output

        # Stack: (max_depth, B, T, C) → (B, T, max_depth, C)
        stacked = torch.stack(outputs, dim=0)        # (max_depth, B, T, C)
        stacked = stacked.permute(1, 2, 0, 3)        # (B, T, max_depth, C)

        # probs: (B, T, max_depth, 1) for broadcasting
        weights = probs.unsqueeze(-1)                # (B, T, max_depth, 1)
        out = (stacked * weights).sum(dim=2)          # (B, T, C)
        return out

    def _hard_apply(
        self, x: torch.Tensor, depths: torch.Tensor
    ) -> torch.Tensor:
        """Apply a different number of recursions per token.

        Tokens with the same depth are batched together for efficiency.
        """
        B, T, C = x.shape
        out = torch.zeros_like(x)

        unique_depths = depths.unique().tolist()
        for d in unique_depths:
            d = int(d)
            # depth index d means we apply (d+1) times
            n_apps = max(1, d + 1)
            mask = (depths == d)  # (B, T) bool

            if mask.any():
                # Gather tokens at this depth, apply, scatter back
                # We need to handle each (b, t) independently — flatten to
                # (N, 1, C) where N = number of matching (b,t) pairs
                flat_mask = mask.view(-1)               # (B*T,)
                flat_x = x.view(B * T, C)               # (B*T, C)
                selected = flat_x[flat_mask]             # (N, C)
                selected = selected.unsqueeze(1)          # (N, 1, C)

                applied = self.apply_n_times(selected, n_apps)  # (N, 1, C)
                applied = applied.squeeze(1)              # (N, C)

                flat_out = out.view(B * T, C)
                flat_out[flat_mask] = applied
                out = flat_out.view(B, T, C)

        return out

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        depths: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        mode: str = "soft",
    ) -> torch.Tensor:
        """Forward pass with per-token depth routing.

        Args:
            x:      ``(B, T, d_model)``
            depths: ``(B, T)`` depth indices.
            probs:  ``(B, T, max_depth)`` (required for soft mode).
            mode:   ``"soft"`` or ``"hard"``.

        Returns:
            ``(B, T, d_model)``
        """
        return self.apply_with_depths(x, depths, probs=probs, mode=mode)


# ---------------------------------------------------------------------------
# MixtureOfRecursionsLayer
# ---------------------------------------------------------------------------


class MixtureOfRecursionsLayer(nn.Module):
    """One MoR layer: router + shared recursive block.

    Combines a ``RecursionDepthRouter`` (decides how deep to recurse) with a
    ``RecursiveTransformerBlock`` (executes the recursion).

    Args:
        d_model:   Hidden dimension.
        n_heads:   Number of attention heads.
        max_depth: Maximum recursion depth.
    """

    def __init__(
        self, d_model: int, n_heads: int, max_depth: int = 4
    ) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.router = RecursionDepthRouter(d_model, max_depth)
        self.block = RecursiveTransformerBlock(d_model, n_heads, max_depth)

    # ------------------------------------------------------------------
    def depth_regularizer_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """Encourage tokens to use lower depths (efficiency penalty).

        L_reg = mean(expected_depth) / max_depth

        This normalises the loss to [0, 1] regardless of max_depth.

        Args:
            probs: ``(B, T, max_depth)`` probability distribution.

        Returns:
            Scalar tensor in ``[0, 1]``.
        """
        exp_depth = self.router.expected_depth(probs)   # (B, T)
        return exp_depth.mean() / (self.max_depth - 1 + 1e-8)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply router then recursive block.

        Args:
            x: ``(B, T, d_model)``

        Returns:
            out:        ``(B, T, d_model)`` transformed hidden states.
            depth_probs: ``(B, T, max_depth)`` router probability distribution.
        """
        depths, probs = self.router(x)
        mode = "soft" if self.training else "hard"
        out = self.block(x, depths, probs=probs, mode=mode)
        return out, probs


# ---------------------------------------------------------------------------
# MoRLanguageModel
# ---------------------------------------------------------------------------


class MoRLanguageModel(nn.Module):
    """Full Mixture-of-Recursions language model.

    Stacks multiple ``MixtureOfRecursionsLayer`` modules on top of a token
    embedding and projects to vocabulary logits.

    Args:
        d_model:    Hidden dimension.
        vocab_size: Vocabulary size.
        n_layers:   Number of MoR layers.
        n_heads:    Number of attention heads per layer.
        max_depth:  Maximum recursion depth per layer.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int = 4,
        n_heads: int = 4,
        max_depth: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_depth = max_depth

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                MixtureOfRecursionsLayer(d_model, n_heads, max_depth)
                for _ in range(n_layers)
            ]
        )
        self.norm = _RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run forward pass through all MoR layers.

        Args:
            input_ids: ``(B, T)`` token indices.

        Returns:
            logits:     ``(B, T, vocab_size)`` unnormalised token scores.
            depth_info: List of length ``n_layers``, each entry is
                        ``(B, T, max_depth)`` depth probability tensors.
        """
        x = self.embedding(input_ids)       # (B, T, d_model)
        depth_info: list[torch.Tensor] = []

        for layer in self.layers:
            x, probs = layer(x)
            depth_info.append(probs)

        x = self.norm(x)
        logits = self.lm_head(x)            # (B, T, vocab_size)
        return logits, depth_info

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        depth_reg_weight: float = 0.01,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined LM cross-entropy loss + depth regularisation.

        Next-token prediction loss is computed with a 1-token shift:
        the model predicts token ``t+1`` from token ``t``.

        Args:
            input_ids:       ``(B, T)`` token indices.
            depth_reg_weight: Weight applied to the depth regularisation loss.

        Returns:
            total_loss: Scalar — lm_loss + depth_reg_weight * reg_loss.
            lm_loss:    Scalar cross-entropy loss.
            reg_loss:   Scalar depth regularisation loss.
        """
        logits, depth_info = self(input_ids)   # (B, T, vocab), list of probs

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()    # (B, T-1, vocab)
        shift_labels = input_ids[:, 1:].contiguous()     # (B, T-1)
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )

        # Depth regularisation: average over all layers
        reg_losses: list[torch.Tensor] = []
        for layer_idx, probs in enumerate(depth_info):
            layer_module = self.layers[layer_idx]
            assert isinstance(layer_module, MixtureOfRecursionsLayer)
            reg_losses.append(layer_module.depth_regularizer_loss(probs))

        reg_loss = torch.stack(reg_losses).mean()
        total_loss = lm_loss + depth_reg_weight * reg_loss
        return total_loss, lm_loss, reg_loss


# ---------------------------------------------------------------------------
# RecursionAnalyzer
# ---------------------------------------------------------------------------


class RecursionAnalyzer:
    """Utility class for analysing per-token recursion depth statistics.

    All methods are stateless and operate purely on tensors/lists passed in;
    no model state is stored.
    """

    @staticmethod
    def mean_depth_per_layer(
        depth_infos: list[list[torch.Tensor]],
    ) -> list[float]:
        """Compute mean recursion depth for each layer across a batch.

        Args:
            depth_infos: Outer list is over examples (or steps); inner list
                         is over layers.  Each tensor is
                         ``(B, T, max_depth)`` depth probs for one layer.

        Returns:
            List of length ``n_layers`` with mean expected depth per layer.
        """
        if not depth_infos or not depth_infos[0]:
            return []

        n_layers = len(depth_infos[0])
        layer_means: list[float] = []

        for layer_idx in range(n_layers):
            depths_this_layer: list[torch.Tensor] = []
            for step_info in depth_infos:
                probs = step_info[layer_idx]             # (B, T, max_depth)
                max_depth = probs.shape[-1]
                depth_vals = torch.arange(
                    max_depth, dtype=probs.dtype, device=probs.device
                )
                exp_d = (probs * depth_vals).sum(dim=-1)  # (B, T)
                depths_this_layer.append(exp_d.mean())

            layer_means.append(
                float(torch.stack(depths_this_layer).mean().item())
            )

        return layer_means

    @staticmethod
    def depth_histogram(probs: torch.Tensor) -> dict[int, float]:
        """Compute the fraction of tokens assigned to each depth level.

        The depth assignment is the argmax of ``probs`` per token.

        Args:
            probs: ``(B, T, max_depth)`` probability distribution.

        Returns:
            Dictionary mapping depth index → fraction of tokens (sums to 1).
        """
        B, T, max_depth = probs.shape
        depths = probs.argmax(dim=-1).view(-1)   # (B*T,)
        total = float(depths.numel())
        histogram: dict[int, float] = {}
        for d in range(max_depth):
            count = float((depths == d).sum().item())
            histogram[d] = count / total
        return histogram

    @staticmethod
    def compute_flop_ratio(mean_depth: float, max_depth: int) -> float:
        """Ratio of actual compute used vs maximum possible compute.

        A value of 1.0 means all tokens used ``max_depth`` applications;
        lower values indicate compute savings.

        Args:
            mean_depth: Mean recursion depth (0-indexed) across all tokens.
            max_depth:  Maximum recursion depth (number of depth classes).

        Returns:
            Float in ``(0, 1]``.
        """
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        # mean_depth is 0-indexed, so actual applications = mean_depth + 1
        actual = mean_depth + 1.0
        maximum = float(max_depth)
        ratio = actual / maximum
        # Clamp to (0, 1] — ratio can exceed 1.0 only due to floating-point;
        # clamp defensively.
        return min(max(ratio, 1e-9), 1.0)


# ---------------------------------------------------------------------------
# MoRConfig
# ---------------------------------------------------------------------------


@dataclass
class MoRConfig:
    """Configuration dataclass for Mixture of Recursions models.

    Attributes:
        d_model:          Hidden dimension.
        vocab_size:       Vocabulary size.
        n_layers:         Number of MoR layers.
        n_heads:          Number of attention heads.
        max_depth:        Maximum recursion depth per layer.
        depth_reg_weight: Weight for depth regularisation loss.
    """

    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    n_heads: int = 4
    max_depth: int = 4
    depth_reg_weight: float = 0.01
