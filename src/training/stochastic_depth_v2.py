"""Stochastic Depth (Huang et al. 2016) and Layer Dropping (Fan et al. 2019).

Randomly skip transformer layers during training to improve robustness and
enable inference-time speed/quality trade-offs. Pure PyTorch — no external deps.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# StochasticDepthSchedule
# ---------------------------------------------------------------------------

class StochasticDepthSchedule:
    """Per-layer drop probability schedule.

    Args:
        n_layers: Total number of transformer layers.
        mode: One of "linear", "uniform", or "constant".
              "linear"   — p_l = (l / L) * max_prob  (deeper = more likely to drop)
              "uniform"  — all layers get max_prob / n_layers  (same prob for each)
              "constant" — all layers get max_prob
        max_prob: Maximum drop probability.
    """

    VALID_MODES = ("linear", "uniform", "constant")

    def __init__(self, n_layers: int, mode: str = "linear", max_prob: float = 0.5) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown mode {mode!r}. Expected one of {self.VALID_MODES}."
            )
        if not (0.0 <= max_prob <= 1.0):
            raise ValueError(f"max_prob must be in [0, 1], got {max_prob}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.n_layers = n_layers
        self.mode = mode
        self.max_prob = max_prob

    def prob(self, layer_idx: int) -> float:
        """Drop probability for layer at index layer_idx (0-based).

        Args:
            layer_idx: Layer index in [0, n_layers).

        Returns:
            Drop probability for this layer.
        """
        if self.mode == "linear":
            # Layer 0 has drop prob 0; last layer has drop prob max_prob.
            # Using (l+1)/L so layer 0 still gets a non-zero but small prob,
            # but the spec says p_l = l/L * max_prob (0-indexed, so layer 0 = 0).
            return (layer_idx / self.n_layers) * self.max_prob
        elif self.mode == "uniform":
            # All layers get the same probability.
            return self.max_prob / self.n_layers
        else:  # "constant"
            return self.max_prob

    def survival_probs(self) -> list[float]:
        """Return list of (1 - drop_prob) for all layers.

        Returns:
            List of length n_layers with survival (keep) probabilities.
        """
        return [1.0 - self.prob(i) for i in range(self.n_layers)]


# ---------------------------------------------------------------------------
# StochasticDepth
# ---------------------------------------------------------------------------

class StochasticDepth(nn.Module):
    """Wraps a module to randomly skip it during training.

    During training: with probability drop_prob the module is skipped and x is
    returned unchanged.  During eval the module is always applied.

    If scale_by_keep=True the output is divided by (1 - drop_prob) when the
    module is not skipped, maintaining expected activation magnitudes.

    Args:
        module: The wrapped nn.Module.
        drop_prob: Probability of dropping the module on any given forward call.
        scale_by_keep: If True, scale output by 1/(1-drop_prob) when not dropped.
    """

    def __init__(
        self,
        module: nn.Module,
        drop_prob: float = 0.1,
        scale_by_keep: bool = True,
    ) -> None:
        super().__init__()
        if not (0.0 <= drop_prob <= 1.0):
            raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}")
        self.module = module
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass with stochastic depth.

        Args:
            x: Input tensor (first positional argument passed to module).
            *args: Extra positional args forwarded to wrapped module.
            **kwargs: Extra keyword args forwarded to wrapped module.

        Returns:
            Module output or x unchanged if the layer was dropped.
        """
        if not self.training or self.drop_prob == 0.0:
            return self.module(x, *args, **kwargs)

        if self.drop_prob >= 1.0:
            return x

        # Single stochastic decision for the entire batch.
        keep = torch.rand(1).item() >= self.drop_prob
        if not keep:
            return x

        out = self.module(x, *args, **kwargs)
        if self.scale_by_keep:
            keep_prob = 1.0 - self.drop_prob
            out = out / keep_prob
        return out


# ---------------------------------------------------------------------------
# LayerDropTransformer
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    """Minimal transformer block: self-attention + feed-forward with residuals."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class LayerDropTransformer(nn.Module):
    """Transformer with LayerDrop: each layer sampled independently each step.

    Args:
        d_model: Hidden dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        vocab_size: Vocabulary size.
        drop_schedule: StochasticDepthSchedule controlling per-layer drop probs.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        drop_schedule: StochasticDepthSchedule,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.drop_schedule = drop_schedule
        self._inference_sublayer_count: int | None = None

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            StochasticDepth(
                _TransformerBlock(d_model, n_heads),
                drop_prob=drop_schedule.prob(i),
                scale_by_keep=True,
            )
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def set_inference_sublayer_count(self, k: int) -> None:
        """Use only every (n_layers // k)th layer at inference for speed.

        Args:
            k: Desired number of active layers during inference.
               Pass None to restore full-depth inference.
        """
        if k < 1 or k > self.n_layers:
            raise ValueError(
                f"k must be in [1, {self.n_layers}], got {k}"
            )
        self._inference_sublayer_count = k

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input_ids: (B, T) long tensor of token ids.

        Returns:
            logits: (B, T, vocab_size) float tensor.
        """
        x = self.embed(input_ids)

        if not self.training and self._inference_sublayer_count is not None:
            # Use every stride-th layer for fast inference.
            k = self._inference_sublayer_count
            stride = max(1, self.n_layers // k)
            active_indices = list(range(0, self.n_layers, stride))[:k]
            for i in active_indices:
                # In eval, StochasticDepth always applies the module.
                x = self.layers[i](x)
        else:
            for layer in self.layers:
                x = layer(x)

        x = self.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# DropPathMixer
# ---------------------------------------------------------------------------

class DropPathMixer(nn.Module):
    """Sample drop decisions per example in the batch (sample-wise stochastic depth).

    Applies residual connection with a per-sample binary mask:
        output = x + stochastic_mask * residual

    During training each sample independently decides to keep or drop the residual.
    During eval the full residual is always added (mask = 1).

    Args:
        drop_prob: Probability of dropping the residual for any given sample.
    """

    def __init__(self, drop_prob: float = 0.1) -> None:
        super().__init__()
        if not (0.0 <= drop_prob <= 1.0):
            raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}")
        self.drop_prob = drop_prob

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        """Add residual to x with per-sample stochastic masking.

        Args:
            x: (B, T, D) base activations.
            residual: (B, T, D) residual to add.

        Returns:
            (B, T, D) tensor: x + mask * residual.
        """
        if not self.training or self.drop_prob == 0.0:
            return x + residual

        B = x.shape[0]
        keep_prob = 1.0 - self.drop_prob
        # Bernoulli mask per sample, broadcast over (T, D).
        mask = torch.bernoulli(
            torch.full((B, 1, 1), keep_prob, device=x.device, dtype=x.dtype)
        )
        # Scale so expected value of masked residual equals the original residual.
        mask = mask / keep_prob
        return x + mask * residual


# ---------------------------------------------------------------------------
# StochasticDepthAnalyzer
# ---------------------------------------------------------------------------

class StochasticDepthAnalyzer:
    """Collect and summarize per-step layer drop statistics.

    Args:
        n_layers: Number of transformer layers being tracked.
    """

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        # Each entry is a list[bool] of length n_layers.
        self._history: list[list[bool]] = []

    def record_step(self, layer_masks: list[bool]) -> None:
        """Record which layers were active (True) this step.

        Args:
            layer_masks: Boolean list of length n_layers.
                         True = layer was active (not dropped).
        """
        if len(layer_masks) != self.n_layers:
            raise ValueError(
                f"layer_masks must have length {self.n_layers}, got {len(layer_masks)}"
            )
        self._history.append(list(layer_masks))

    def drop_rates(self) -> list[float]:
        """Empirical drop rate per layer over all recorded steps.

        Returns:
            List of length n_layers with fraction of steps each layer was dropped.
        """
        if not self._history:
            return [0.0] * self.n_layers

        totals = [0.0] * self.n_layers
        n_steps = len(self._history)
        for masks in self._history:
            for i, active in enumerate(masks):
                if not active:
                    totals[i] += 1.0
        return [t / n_steps for t in totals]

    def effective_depth(self) -> float:
        """Mean number of active layers per step.

        Returns:
            Average over all recorded steps.
        """
        if not self._history:
            return float(self.n_layers)

        total_active = sum(sum(masks) for masks in self._history)
        return total_active / len(self._history)

    def reset(self) -> None:
        """Clear all recorded history."""
        self._history = []
