"""Dynamic Depth: adaptive early exit and layer skipping for Aurelius transformer.

Implements per-batch element early exit and per-layer skip routing so that
inference compute is allocated proportionally to input complexity.

Components (distinct from early_exit.py which uses fixed exit classifiers):
  - DynamicDepthConfig: dataclass for thresholds and routing settings
  - ExitRouter: learned sigmoid gate on the last-token hidden state
  - LayerSkipRouter: learned sigmoid gate deciding whether to skip a layer
  - compute_exit_confidence: max-softmax confidence from logit distribution
  - DynamicDepthTransformer: wraps a base AureliusTransformer with dynamic routing
  - AdaptiveLayerSelector: norm-based heuristic for selecting which layers to run
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DynamicDepthConfig:
    """Configuration for dynamic depth routing.

    Attributes:
        exit_threshold:    Confidence (max softmax prob) above which a batch element
                           exits early. Range [0, 1].
        skip_threshold:    Skip-router probability below which a layer is skipped.
                           Range [0, 1]. Lower = more layers skipped.
        min_layers:        Minimum number of layers always executed, regardless of
                           router decisions.
        temperature:       Temperature applied to the base model logits when computing
                           exit confidence.
        use_learned_router: If True, use ExitRouter / LayerSkipRouter. If False, fall
                            back to pure confidence-based gating (no learned routers).
    """

    exit_threshold: float = 0.9
    skip_threshold: float = 0.1
    min_layers: int = 1
    temperature: float = 1.0
    use_learned_router: bool = True


# ---------------------------------------------------------------------------
# ExitRouter
# ---------------------------------------------------------------------------


class ExitRouter(nn.Module):
    """Learned exit gate operating on the last-token hidden state.

    Args:
        d_model: Hidden dimension of the transformer.

    Input:
        x: (B, D) — last-token hidden state for each batch element.

    Output:
        (B, 1) — exit probability in [0, 1].
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute exit probability from last-token hidden state.

        Args:
            x: (B, D) last-token hidden state.

        Returns:
            (B, 1) exit probability.
        """
        return self.sigmoid(self.linear(x))


# ---------------------------------------------------------------------------
# LayerSkipRouter
# ---------------------------------------------------------------------------


class LayerSkipRouter(nn.Module):
    """Learned skip gate for a single transformer layer.

    Args:
        d_model: Hidden dimension of the transformer.

    Input:
        x: (B, D) — last-token hidden state.

    Output:
        (B, 1) — skip probability in [0, 1]. Low value = likely to skip.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute skip probability.

        Args:
            x: (B, D) last-token hidden state.

        Returns:
            (B, 1) skip probability.
        """
        return self.sigmoid(self.linear(x))


# ---------------------------------------------------------------------------
# Confidence utility
# ---------------------------------------------------------------------------


def compute_exit_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-batch-element confidence as max softmax probability.

    Args:
        logits: (B, V) — output logits for the last token position.

    Returns:
        (B,) — maximum softmax probability (confidence) for each element.
    """
    probs = F.softmax(logits, dim=-1)  # (B, V)
    confidence, _ = probs.max(dim=-1)  # (B,)
    return confidence


# ---------------------------------------------------------------------------
# DynamicDepthTransformer
# ---------------------------------------------------------------------------


class DynamicDepthTransformer(nn.Module):
    """Wraps a base AureliusTransformer with dynamic early exit and layer skipping.

    Unlike EarlyExitTransformer (which uses fixed intermediate classifiers),
    this module gates on:
      1. Per-layer skip routers that may bypass individual layers.
      2. Per-layer exit routers that may terminate computation early for each
         batch element independently.

    The exit decision is made on the last-token logits produced by the base
    model's lm_head after the current layer's hidden state is normed.

    Args:
        base_model: An AureliusTransformer instance.
        config:     DynamicDepthConfig controlling routing behaviour.
    """

    def __init__(self, base_model: Any, config: DynamicDepthConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config

        n_layers = base_model.config.n_layers
        d_model = base_model.config.d_model

        self.exit_routers = nn.ModuleList([ExitRouter(d_model) for _ in range(n_layers)])
        self.skip_routers = nn.ModuleList([LayerSkipRouter(d_model) for _ in range(n_layers)])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[None, torch.Tensor, list[int]]:
        """Run transformer layers with dynamic early exit and layer skipping.

        For each layer i:
          - If i < min_layers: always execute.
          - Otherwise:
            1. Consult skip router. If skip_prob < skip_threshold: skip this layer.
            2. After execution, check exit router + confidence. If threshold met,
               mark that batch element as exited.

        Args:
            input_ids: (B, T) token indices.

        Returns:
            Tuple of:
                loss:               Always None (no labels provided).
                logits:             (B, T, V) logits from the exited/final layer.
                exit_layer_indices: list[int] of length B — which layer index each
                                    batch element exited at (0-indexed).
        """
        bm = self.base_model
        B, T = input_ids.shape
        n_layers = bm.config.n_layers
        vocab_size = bm.config.vocab_size
        min_layers = self.config.min_layers
        exit_threshold = self.config.exit_threshold
        skip_threshold = self.config.skip_threshold
        temperature = self.config.temperature

        # Embed tokens
        x = bm.embed(input_ids)  # (B, T, D)
        freqs_cis = bm.freqs_cis[:T]  # (T, head_dim//2) complex

        # Track which batch elements have already exited
        exited = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        exit_layer_indices = [-1] * B  # filled as elements exit

        # Accumulate output logits (updated whenever an element exits)
        output_logits = torch.zeros(B, T, vocab_size, device=input_ids.device)

        for layer_idx, layer in enumerate(bm.layers):
            # --- Skip router (skip only after min_layers) ---
            if layer_idx >= min_layers and self.config.use_learned_router:
                last_hidden = x[:, -1, :]  # (B, D)
                skip_prob = self.skip_routers[layer_idx](last_hidden)  # (B, 1)
                # Elements that have already exited are not affected; for the rest,
                # decide whether to skip. We skip when skip_prob < skip_threshold
                # for ALL active elements (layer-level decision for simplicity —
                # a full per-element skip would require masking the attention graph).
                active_skip_probs = skip_prob[~exited]  # (n_active, 1)
                if active_skip_probs.numel() > 0:
                    mean_skip_prob = active_skip_probs.mean().item()
                    if mean_skip_prob < skip_threshold:
                        # Skip this layer: pass hidden states unchanged
                        continue

            # Run the layer
            x, _kv = layer(x, freqs_cis)  # (B, T, D)

            # --- Exit router (check only after min_layers) ---
            if layer_idx < min_layers - 1:
                continue

            # Compute logits for the current hidden state
            x_normed = bm.norm(x)  # (B, T, D)
            logits = bm.lm_head(x_normed)  # (B, T, V)

            if temperature != 1.0:
                logits = logits / temperature

            # Confidence from last-token logits
            last_logits = logits[:, -1, :]  # (B, V)
            confidence = compute_exit_confidence(last_logits)  # (B,)

            if self.config.use_learned_router:
                last_hidden = x[:, -1, :]
                exit_prob = self.exit_routers[layer_idx](last_hidden).squeeze(-1)  # (B,)
                # Exit if either learned router OR confidence exceeds threshold
                should_exit = (confidence > exit_threshold) | (exit_prob > exit_threshold)
            else:
                should_exit = confidence > exit_threshold

            # Mark newly-exiting batch elements
            newly_exiting = should_exit & ~exited
            if newly_exiting.any():
                for b_idx in range(B):
                    if newly_exiting[b_idx]:
                        output_logits[b_idx] = logits[b_idx]
                        exit_layer_indices[b_idx] = layer_idx
                exited = exited | newly_exiting

            if exited.all():
                break

        # Elements that never exited: use final logits
        never_exited = ~exited
        if never_exited.any():
            x_normed = bm.norm(x)
            final_logits = bm.lm_head(x_normed)
            if temperature != 1.0:
                final_logits = final_logits / temperature
            for b_idx in range(B):
                if never_exited[b_idx]:
                    output_logits[b_idx] = final_logits[b_idx]
                    exit_layer_indices[b_idx] = n_layers - 1

        return None, output_logits, exit_layer_indices

    # ------------------------------------------------------------------
    # Efficiency statistics
    # ------------------------------------------------------------------

    def compute_efficiency_stats(self, exit_layers: list[int]) -> dict:
        """Compute efficiency statistics from exit layer indices.

        Args:
            exit_layers: list[int] of length B, as returned by forward().

        Returns:
            dict with keys:
                "mean_exit_layer":  float — average exit layer across batch.
                "min_exit_layer":   int   — earliest exit layer.
                "max_exit_layer":   int   — latest exit layer.
                "early_exit_rate":  float — fraction of elements that exited
                                           before the final layer.
        """
        n_layers = self.base_model.config.n_layers
        mean_exit = float(sum(exit_layers)) / len(exit_layers)
        min_exit = int(min(exit_layers))
        max_exit = int(max(exit_layers))
        early_exit_rate = sum(1 for e in exit_layers if e < n_layers - 1) / len(exit_layers)
        return {
            "mean_exit_layer": mean_exit,
            "min_exit_layer": min_exit,
            "max_exit_layer": max_exit,
            "early_exit_rate": float(early_exit_rate),
        }


# ---------------------------------------------------------------------------
# AdaptiveLayerSelector
# ---------------------------------------------------------------------------


class AdaptiveLayerSelector:
    """Selects which layers to run based on hidden state complexity (L2 norm).

    Higher hidden-state norm => more complex input => more layers are executed.

    Args:
        n_layers:   Total number of layers in the model.
        min_layers: Minimum layers to always include (default 2).
    """

    def __init__(self, n_layers: int, min_layers: int = 2) -> None:
        self.n_layers = n_layers
        self.min_layers = min_layers

    def select_layers(self, hidden: torch.Tensor) -> list[int]:
        """Select layer indices to execute based on input complexity.

        Complexity is measured as the mean L2 norm of the hidden state across
        batch and sequence dimensions.  The norm is mapped linearly to a layer
        count in [min_layers, n_layers].

        Args:
            hidden: (B, T, D) hidden state tensor.

        Returns:
            Sorted list[int] of layer indices to execute, all in [0, n_layers).
        """
        # Mean L2 norm over batch and sequence
        norm_val = hidden.norm(dim=-1).mean().item()  # scalar

        # Normalise norm to a fraction in [0, 1] using a sigmoid-like clamp.
        # We use tanh so that typical norm values (~1–10) map smoothly.
        fraction = min(1.0, max(0.0, norm_val / (norm_val + 1.0)))  # in (0, 1)

        # Number of layers to run
        n_run = self.min_layers + int(round(fraction * (self.n_layers - self.min_layers)))
        n_run = max(self.min_layers, min(self.n_layers, n_run))

        # Evenly-spaced layer indices across the full stack
        if n_run >= self.n_layers:
            return list(range(self.n_layers))

        step = self.n_layers / n_run
        indices = sorted({int(i * step) for i in range(n_run)})

        # Guarantee min_layers by padding with the last layers if needed
        while len(indices) < self.min_layers:
            candidate = len(indices)
            if candidate not in indices:
                indices.append(candidate)
            indices.sort()

        # Clip any out-of-range indices
        indices = [i for i in indices if i < self.n_layers]
        return indices
