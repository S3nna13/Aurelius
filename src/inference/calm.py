"""CALM: Confident Adaptive Language Modeling (Schuster et al., NeurIPS 2022).

arXiv:2207.07061

Enables adaptive computation by exiting at intermediate transformer layers when
the model is "confident enough" about its prediction. Provides risk-controlled
early exit with statistical coverage guarantees.

Variable notation matches the paper:
  L  — total number of layers
  l  — current layer index (0-indexed internally, 1-indexed in paper)
  λ  — confidence threshold
  ε  — maximum acceptable error rate (1 - target_coverage)
  c_l — confidence score at layer l
  h_l — hidden state at layer l
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# CALMConfidenceScorer
# ---------------------------------------------------------------------------

class CALMConfidenceScorer:
    """Computes scalar confidence c_l ∈ [0, 1] from layer-l logits.

    Two methods (Section 3 of the paper):
      - 'softmax_max':    c_l = max(softmax(logits))
      - 'softmax_entropy': c_l = 1 - H(softmax(logits)) / log(V)
        where H is Shannon entropy and V is vocabulary size.
    """

    VALID_METHODS = {"softmax_max", "softmax_entropy"}

    def __init__(self, method: str = "softmax_max") -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown confidence method '{method}'. "
                f"Choose from {self.VALID_METHODS}."
            )
        self.method = method

    def score(self, logits: Tensor) -> float:
        """Compute confidence from logits of shape (..., V).

        Operates on the last token of the last batch dimension for a single
        token step. Accepts (V,), (T, V), or (B, T, V); always returns a
        single Python float representing the mean confidence over the batch.

        Args:
            logits: Raw (un-normalised) logit tensor, last dim = vocab size V.

        Returns:
            Scalar float confidence in [0, 1].
        """
        # Flatten to (N, V) where N = product of leading dims
        V = logits.shape[-1]
        flat = logits.reshape(-1, V)  # (N, V)

        probs = F.softmax(flat, dim=-1)  # (N, V)

        if self.method == "softmax_max":
            # c_l = max_v p_v
            c = probs.max(dim=-1).values  # (N,)
        else:  # softmax_entropy
            # H(p) = -sum p log p   (nats or bits — we use nats)
            log_p = torch.log(probs.clamp(min=1e-10))
            H = -(probs * log_p).sum(dim=-1)  # (N,)
            max_H = math.log(V) if V > 1 else 1.0
            # c_l = 1 - H / log(V)  ∈ [0, 1]
            c = 1.0 - H / max_H  # (N,)

        return float(c.mean().item())


# ---------------------------------------------------------------------------
# CALMEarlyExitDecoder
# ---------------------------------------------------------------------------

class CALMEarlyExitDecoder(nn.Module):
    """Decides at which layer to exit for a single forward pass.

    Per-token exit procedure (Section 3):
      For l in {0, ..., L-1}:
        1. Compute logits_l = lm_head(h_l)
        2. Compute c_l = scorer.score(logits_l)
        3. If c_l >= λ  OR  l == L-1: EXIT → return (logits_l, l, c_l)

    Args:
        n_layers: Total number of transformer layers L.
        threshold: Confidence threshold λ ∈ [0, 1].
        confidence_method: 'softmax_max' or 'softmax_entropy'.
    """

    def __init__(
        self,
        n_layers: int,
        threshold: float = 0.9,
        confidence_method: str = "softmax_max",
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_layers = n_layers
        self.threshold = threshold
        self.scorer = CALMConfidenceScorer(method=confidence_method)

    def forward(
        self,
        layer_outputs: List[Tensor],
        lm_head: nn.Linear,
    ) -> Tuple[Tensor, int, float]:
        """Run adaptive exit over layer hidden states.

        Args:
            layer_outputs: List of L tensors, each (B, T, d_model).
            lm_head: Linear projection from d_model → vocab_size V.

        Returns:
            (final_logits, exit_layer, confidence)
              final_logits : (B, T, V)
              exit_layer   : int in [0, n_layers - 1]
              confidence   : float in [0, 1]
        """
        if len(layer_outputs) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} layer outputs, "
                f"got {len(layer_outputs)}."
            )

        λ = self.threshold

        for l, h_l in enumerate(layer_outputs):
            # h_l : (B, T, d_model)
            logits_l = lm_head(h_l)  # (B, T, V)
            c_l = self.scorer.score(logits_l)

            if c_l >= λ or l == self.n_layers - 1:
                return logits_l, l, c_l

        # Should be unreachable, but be explicit
        raise RuntimeError("CALM forward loop exited without returning.")  # pragma: no cover


# ---------------------------------------------------------------------------
# CALMCalibrator
# ---------------------------------------------------------------------------

class CALMCalibrator:
    """Finds optimal threshold λ via binary search on a calibration set.

    Calibration procedure (Section 4.2, Algorithm 1):
      Given a calibration set of (layer_outputs, lm_head) pairs and the
      corresponding full-model predictions, find the smallest λ such that:
        P(early_exit_prediction != full_model_prediction) <= ε
      where ε = 1 - target_coverage.

    The full-model prediction is taken to be the argmax of the LAST layer's
    logits (i.e., no early exit).
    """

    def __init__(
        self,
        confidence_method: str = "softmax_max",
        n_search_steps: int = 20,
    ) -> None:
        self.confidence_method = confidence_method
        self.n_search_steps = n_search_steps

    def calibrate(
        self,
        layer_outputs_list: List[List[Tensor]],
        lm_head: nn.Linear,
        target_coverage: float = 0.95,
    ) -> float:
        """Find the smallest λ that achieves the target coverage.

        Args:
            layer_outputs_list: List of N calibration examples; each element
                is a list of L tensors of shape (B, T, d_model).
            lm_head: Shared LM head (d_model → V).
            target_coverage: Desired P(early == full) >= target_coverage,
                equivalently ε = 1 - target_coverage.

        Returns:
            Optimal threshold λ* ∈ [0, 1].
        """
        if not (0.0 < target_coverage <= 1.0):
            raise ValueError(
                f"target_coverage must be in (0, 1], got {target_coverage}."
            )
        ε = 1.0 - target_coverage

        # Pre-compute full-model (last-layer) predictions for each example
        full_preds: List[Tensor] = []
        with torch.no_grad():
            for layer_outputs in layer_outputs_list:
                h_last = layer_outputs[-1]           # (B, T, d_model)
                logits_full = lm_head(h_last)        # (B, T, V)
                full_preds.append(logits_full.argmax(dim=-1))  # (B, T)

        def error_rate_at(λ: float) -> float:
            """Fraction of examples where early exit disagrees with full model."""
            n_layers = len(layer_outputs_list[0])
            decoder = CALMEarlyExitDecoder(
                n_layers=n_layers,
                threshold=λ,
                confidence_method=self.confidence_method,
            )
            errors = 0
            total = 0
            with torch.no_grad():
                for layer_outputs, full_pred in zip(layer_outputs_list, full_preds):
                    logits_exit, _, _ = decoder.forward(layer_outputs, lm_head)
                    exit_pred = logits_exit.argmax(dim=-1)  # (B, T)
                    errors += int((exit_pred != full_pred).sum().item())
                    total += full_pred.numel()
            return errors / total if total > 0 else 0.0

        # Binary search: find smallest λ s.t. error_rate(λ) <= ε
        # Higher λ → later exit → lower error rate
        lo, hi = 0.0, 1.0
        best_λ = 1.0  # conservative default: always go to last layer

        for _ in range(self.n_search_steps):
            mid = (lo + hi) / 2.0
            if error_rate_at(mid) <= ε:
                best_λ = mid
                hi = mid  # try lower λ (earlier exit)
            else:
                lo = mid   # need higher λ to reduce errors

        return float(best_λ)


# ---------------------------------------------------------------------------
# CALMDecoder
# ---------------------------------------------------------------------------

class CALMDecoder(nn.Module):
    """High-level decoder wrapping CALMEarlyExitDecoder for multi-token decoding.

    Handles a batch of token positions, each yielding its own exit layer and
    confidence score. Exposes an efficiency metric via average_exit_layer().

    Args:
        n_layers: Total transformer layers L.
        lm_head: LM projection head (d_model → V).
        threshold: Confidence threshold λ.
        confidence_method: Scorer method for confidence computation.
    """

    def __init__(
        self,
        n_layers: int,
        lm_head: nn.Linear,
        threshold: float = 0.9,
        confidence_method: str = "softmax_max",
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.lm_head = lm_head
        self.threshold = threshold
        self._exit_decoder = CALMEarlyExitDecoder(
            n_layers=n_layers,
            threshold=threshold,
            confidence_method=confidence_method,
        )

    def decode_step(
        self,
        layer_outputs: List[Tensor],
    ) -> Tuple[Tensor, List[int], List[float]]:
        """Run CALM early exit for one decoding step.

        Unlike per-position iteration, this treats the full sequence (B, T, d)
        as a single unit, returning one exit_layer and confidence for the step
        (matching the paper's per-token-step framing).

        Args:
            layer_outputs: List of L tensors of shape (B, T, d_model).

        Returns:
            (token_ids, exit_layers, confidences)
              token_ids   : (B, T) int64 — argmax predictions
              exit_layers : list of length 1 (the single exit layer for the step)
              confidences : list of length 1 (the confidence at exit)
        """
        final_logits, exit_layer, confidence = self._exit_decoder.forward(
            layer_outputs, self.lm_head
        )
        token_ids = final_logits.argmax(dim=-1)  # (B, T)
        return token_ids, [exit_layer], [confidence]

    @staticmethod
    def average_exit_layer(exit_layers: List[int]) -> float:
        """Compute mean exit layer across decoding steps (efficiency metric).

        Lower value → more computation saved on average.

        Args:
            exit_layers: List of exit layer indices (0-indexed).

        Returns:
            Mean exit layer as a float.
        """
        if not exit_layers:
            raise ValueError("exit_layers must be non-empty.")
        return float(sum(exit_layers) / len(exit_layers))
