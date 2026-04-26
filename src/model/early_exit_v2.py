"""Patience-based early exit with token-level decisions for Aurelius transformer.

Unlike early_exit.py (batch-level, fixed intermediate classifiers) and
dynamic_depth.py (batch-level skip routing), this module implements:
  - Per-token patience tracking: exit when the top-1 predicted token is
    unchanged for `patience` consecutive layers.
  - Per-token confidence exit: exit when the softmax max-prob exceeds a
    confidence threshold.
  - Decisions are made independently for each token position, not per batch
    element or globally.

Components:
    PatienceConfig       — dataclass controlling patience / confidence thresholds
    TokenExitDecision    — dataclass recording why/when each token exited
    PatienceTracker      — stateful tracker for per-token argmax stability
    TokenLevelEarlyExit  — nn.Module wrapping a base AureliusTransformer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PatienceConfig:
    """Configuration for patience-based token-level early exit.

    Attributes:
        patience:             Exit if the top-1 token prediction is unchanged
                              for this many consecutive layers.
        min_layers:           Minimum number of layers to always run before any
                              exit decision is considered.
        confidence_threshold: Softmax probability above which a token exits
                              immediately (only if exit_on_confidence=True).
        exit_on_confidence:   Whether to enable confidence-based early exit.
        exit_on_patience:     Whether to enable patience-based early exit.
    """

    patience: int = 3
    min_layers: int = 2
    confidence_threshold: float = 0.9
    exit_on_confidence: bool = True
    exit_on_patience: bool = True


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------


@dataclass
class TokenExitDecision:
    """Records the exit decision for a single token position.

    Attributes:
        token_pos:   Position in the sequence (0-indexed).
        exit_layer:  Layer index (0-indexed) at which this token stopped.
        exit_reason: One of "confidence", "patience", or "max_depth".
        confidence:  Softmax max-prob at the exit layer.
    """

    token_pos: int
    exit_layer: int
    exit_reason: str  # "confidence" | "patience" | "max_depth"
    confidence: float


# ---------------------------------------------------------------------------
# Patience tracker
# ---------------------------------------------------------------------------


class PatienceTracker:
    """Tracks per-token consecutive-argmax stability across layers.

    Args:
        patience: Number of consecutive identical argmax values required to
                  declare a token as "stable" (ready to exit).
        seq_len:  Number of token positions (T) to track.
    """

    def __init__(self, patience: int, seq_len: int) -> None:
        self.patience = patience
        self.seq_len = seq_len
        # Number of consecutive layers with the same argmax prediction, shape (T,)
        self._consecutive_same: torch.Tensor = torch.zeros(seq_len, dtype=torch.long)
        # Last seen argmax per token position, shape (T,)
        self._last_argmax: torch.Tensor = torch.full((seq_len,), -1, dtype=torch.long)

    def update(self, logits: torch.Tensor) -> torch.Tensor:
        """Update patience counters and return a boolean mask of exited tokens.

        Args:
            logits: (B, T, V) — logits at the current layer. If B > 1, only the
                    first batch element is used for the token-level argmax tracking
                    (this tracker operates per token position, not per batch).

        Returns:
            Boolean tensor of shape (T,) — True where patience has been exceeded.
        """
        # Use the first batch element for token-level tracking (T, V)
        tok_logits = logits[0]  # (T, V)
        current_argmax = tok_logits.argmax(dim=-1)  # (T,)

        # Compare to last recorded argmax
        same = current_argmax == self._last_argmax  # (T,) bool

        # Increment where same, reset to 1 where different
        # (1 because we just saw one occurrence of this argmax)
        self._consecutive_same = torch.where(
            same,
            self._consecutive_same + 1,
            torch.ones_like(self._consecutive_same),
        )
        self._last_argmax = current_argmax

        # Patience exceeded when counter >= patience
        return self._consecutive_same >= self.patience

    def reset(self) -> None:
        """Reset all counters to zero (and last argmax to -1)."""
        self._consecutive_same.zero_()
        self._last_argmax.fill_(-1)


# ---------------------------------------------------------------------------
# Token-level early exit module
# ---------------------------------------------------------------------------


class TokenLevelEarlyExit(nn.Module):
    """Wraps an AureliusTransformer with token-level patience-based early exit.

    This module differs from EarlyExitTransformer (batch-level, fixed classifiers)
    and DynamicDepthTransformer (batch-level skip/exit routing) by making
    independent early exit decisions for each token position.

    A lightweight Linear(d_model, 1)+Sigmoid classifier is attached to every
    layer to estimate token-level confidence independently of the vocabulary
    projection.

    Args:
        base_model: An AureliusTransformer (or compatible) instance.
        config:     PatienceConfig controlling patience and confidence settings.
    """

    def __init__(self, base_model: Any, config: PatienceConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config

        n_layers = base_model.config.n_layers
        d_model = base_model.config.d_model

        # One lightweight confidence classifier per layer
        self.exit_classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 1),
                    nn.Sigmoid(),
                )
                for _ in range(n_layers)
            ]
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[None, torch.Tensor, list[TokenExitDecision]]:
        """Run layers one at a time, tracking per-token exit conditions.

        For each layer (after min_layers have been processed):
          1. Compute full (B, T, V) logits via the base model's lm_head.
          2. Check confidence threshold: tokens whose softmax max-prob exceeds
             confidence_threshold exit immediately (reason="confidence").
          3. Update the patience tracker with current argmax predictions.
          4. Tokens whose patience counter has been reached exit
             (reason="patience").
          5. Any tokens still active after all layers exit with reason="max_depth".

        Args:
            input_ids: (B, T) integer token tensor.

        Returns:
            Tuple of:
                None              — no loss (inference only).
                final_logits      — (B, T, V) logits at the layer each token exited.
                decisions         — list[TokenExitDecision], one per token position T.
        """
        bm = self.base_model
        B, T = input_ids.shape
        n_layers = bm.config.n_layers
        vocab_size = bm.config.vocab_size
        device = input_ids.device

        # Embed tokens and get RoPE freqs
        x = bm.embed(input_ids)  # (B, T, D)
        freqs_cis = bm.freqs_cis[:T]  # (T, ...)

        # Per-token exit tracking
        token_exited = torch.zeros(T, dtype=torch.bool, device=device)
        exit_layer = torch.full((T,), n_layers - 1, dtype=torch.long, device=device)
        exit_reason: list[str] = ["max_depth"] * T
        exit_confidence = torch.zeros(T, device=device)

        # Accumulated output logits — updated as tokens exit
        final_logits = torch.zeros(B, T, vocab_size, device=device)

        # Patience tracker (CPU tensors — indices only)
        tracker = PatienceTracker(
            patience=self.config.patience,
            seq_len=T,
        )

        for layer_idx, layer in enumerate(bm.layers):
            # Run the transformer layer
            x, _kv = layer(x, freqs_cis)  # (B, T, D)

            # Skip exit checks until min_layers have been processed
            if layer_idx < self.config.min_layers - 1:
                continue

            # Compute full logits for exit decisions
            x_normed = bm.norm(x)  # (B, T, D)
            logits = bm.lm_head(x_normed)  # (B, T, V)

            # Per-token confidence: use first batch element for token-level decisions
            probs = logits[0].softmax(dim=-1)  # (T, V)
            conf = probs.max(dim=-1).values  # (T,)

            # --- Confidence-based exit ---
            if self.config.exit_on_confidence:
                conf_mask = (conf >= self.config.confidence_threshold) & ~token_exited
                if conf_mask.any():
                    idxs = conf_mask.nonzero(as_tuple=True)[0]
                    for t in idxs.tolist():
                        exit_layer[t] = layer_idx
                        exit_reason[t] = "confidence"
                        exit_confidence[t] = conf[t].item()
                        final_logits[:, t, :] = logits[:, t, :]
                    token_exited |= conf_mask

            # --- Patience-based exit ---
            if self.config.exit_on_patience and not token_exited.all():
                patience_mask = tracker.update(logits)  # (T,) bool (CPU)
                patience_mask_dev = patience_mask.to(device)
                new_patience = patience_mask_dev & ~token_exited
                if new_patience.any():
                    idxs = new_patience.nonzero(as_tuple=True)[0]
                    for t in idxs.tolist():
                        if not token_exited[t]:
                            exit_layer[t] = layer_idx
                            exit_reason[t] = "patience"
                            exit_confidence[t] = conf[t].item()
                            final_logits[:, t, :] = logits[:, t, :]
                    token_exited |= new_patience
            elif self.config.exit_on_patience:
                # Still call update to keep tracker state consistent
                tracker.update(logits)

            # If all tokens have exited we can stop
            if token_exited.all():
                break

        # Remaining tokens that never exited — use final layer logits
        remaining = ~token_exited
        if remaining.any():
            x_normed = bm.norm(x)
            last_logits = bm.lm_head(x_normed)
            probs_last = last_logits[0].softmax(dim=-1)
            conf_last = probs_last.max(dim=-1).values
            idxs = remaining.nonzero(as_tuple=True)[0]
            for t in idxs.tolist():
                exit_confidence[t] = conf_last[t].item()
                final_logits[:, t, :] = last_logits[:, t, :]
            # exit_layer and exit_reason already defaulted to n_layers-1 / "max_depth"

        # Build decision list
        decisions: list[TokenExitDecision] = [
            TokenExitDecision(
                token_pos=t,
                exit_layer=int(exit_layer[t].item()),
                exit_reason=exit_reason[t],
                confidence=float(exit_confidence[t].item()),
            )
            for t in range(T)
        ]

        return None, final_logits, decisions

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_exit_stats(self, decisions: list[TokenExitDecision]) -> dict:
        """Compute summary statistics from a list of token exit decisions.

        Args:
            decisions: list[TokenExitDecision] as returned by forward().

        Returns:
            dict with keys:
                "mean_exit_layer":    float — average exit layer across tokens.
                "exit_reason_counts": dict[str, int] — counts per exit reason.
                "efficiency":         float in [0, 1] — 1 - mean_exit_layer / n_layers.
        """
        n_layers = self.base_model.config.n_layers

        if not decisions:
            return {
                "mean_exit_layer": 0.0,
                "exit_reason_counts": {},
                "efficiency": 0.0,
            }

        mean_exit_layer = sum(d.exit_layer for d in decisions) / len(decisions)

        reason_counts: dict[str, int] = {}
        for d in decisions:
            reason_counts[d.exit_reason] = reason_counts.get(d.exit_reason, 0) + 1

        efficiency = max(0.0, min(1.0, 1.0 - mean_exit_layer / n_layers))

        return {
            "mean_exit_layer": mean_exit_layer,
            "exit_reason_counts": reason_counts,
            "efficiency": efficiency,
        }
