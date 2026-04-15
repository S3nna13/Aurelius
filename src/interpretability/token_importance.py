"""Token importance scoring and saliency maps for transformer interpretability.

Methods:
  - Gradient norm: L2 norm of loss gradient w.r.t. input embeddings
  - Attention rollout: recursive attention matrix multiplication across layers
  - Integrated gradients: path-integrated attribution from baseline to input
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ImportanceConfig:
    method: str = "gradient_norm"   # "gradient_norm", "attention_rollout", "integrated_gradients"
    normalize: bool = True
    smooth_window: int = 1


# ---------------------------------------------------------------------------
# Gradient norm importance
# ---------------------------------------------------------------------------

def gradient_norm_importance(
    model: nn.Module,
    embeddings: Tensor,
    labels: Tensor,
    loss_fn: Callable,
) -> Tensor:
    """Compute L2 norm of loss gradient w.r.t. embeddings.

    Args:
        model: callable model that takes embeddings → logits.
        embeddings: (B, T, d_emb) with requires_grad=True.
        labels: (B, T) token ids.
        loss_fn: callable(logits, labels) → scalar loss.

    Returns:
        (B, T) non-negative importance scores.
    """
    if not embeddings.requires_grad:
        embeddings = embeddings.detach().requires_grad_(True)

    logits = model(embeddings)
    loss = loss_fn(logits, labels)
    loss.backward(retain_graph=True)

    grad = embeddings.grad  # (B, T, d_emb)
    if grad is None:
        return torch.zeros(embeddings.shape[:2])

    importance = grad.norm(dim=-1)  # (B, T)
    return importance.detach()


# ---------------------------------------------------------------------------
# Attention rollout
# ---------------------------------------------------------------------------

def attention_rollout(
    attention_weights: List[Tensor],
    add_residual: bool = True,
) -> Tensor:
    """Compute attention rollout by multiplying attention matrices across layers.

    Args:
        attention_weights: list of (B, H, T, T) attention weight tensors per layer.
        add_residual: if True, add 0.5 * identity residual at each step.

    Returns:
        (B, T, T) aggregated attention.
    """
    if not attention_weights:
        raise ValueError("attention_weights must not be empty")

    B, H, T, _ = attention_weights[0].shape

    # Average over heads
    rollout = attention_weights[0].mean(dim=1)  # (B, T, T)

    if add_residual:
        eye = torch.eye(T, device=rollout.device, dtype=rollout.dtype).unsqueeze(0)
        rollout = 0.5 * rollout + 0.5 * eye
        # Renormalize rows
        rollout = rollout / rollout.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    for attn in attention_weights[1:]:
        layer_attn = attn.mean(dim=1)  # (B, T, T)
        if add_residual:
            layer_attn = 0.5 * layer_attn + 0.5 * eye
            layer_attn = layer_attn / layer_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = torch.bmm(layer_attn, rollout)

    return rollout


# ---------------------------------------------------------------------------
# Integrated gradients
# ---------------------------------------------------------------------------

def integrated_gradients(
    model: nn.Module,
    embeddings: Tensor,
    baseline: Tensor,
    labels: Tensor,
    loss_fn: Callable,
    n_steps: int = 20,
) -> Tensor:
    """Compute integrated gradients attribution.

    Interpolates from baseline to input in n_steps, averages gradients,
    then multiplies by (input - baseline).

    Args:
        model: callable(embeddings) → logits.
        embeddings: (B, T, d_emb) actual input.
        baseline: (B, T, d_emb) neutral baseline (usually zeros).
        labels: (B, T).
        loss_fn: callable(logits, labels) → scalar.
        n_steps: number of interpolation steps.

    Returns:
        (B, T) importance as L2 norm of (B, T, d_emb) attributions.
    """
    grad_sum = torch.zeros_like(embeddings)

    for step in range(n_steps):
        alpha = step / max(n_steps - 1, 1)
        interp = (baseline + alpha * (embeddings - baseline)).detach().requires_grad_(True)
        logits = model(interp)
        loss = loss_fn(logits, labels)
        loss.backward()
        if interp.grad is not None:
            grad_sum = grad_sum + interp.grad.detach()

    avg_grad = grad_sum / n_steps
    attributions = avg_grad * (embeddings - baseline).detach()  # (B, T, d_emb)
    return attributions.norm(dim=-1)  # (B, T)


# ---------------------------------------------------------------------------
# Smoothing and normalization
# ---------------------------------------------------------------------------

def smooth_importance(importance: Tensor, window: int) -> Tensor:
    """Apply 1D average smoothing over T dimension.

    Args:
        importance: (B, T).
        window: smoothing window size (1 = no-op).

    Returns:
        (B, T) smoothed tensor.
    """
    if window <= 1:
        return importance

    B, T = importance.shape
    # Reflect pad to handle edges
    pad = window // 2
    padded = F.pad(importance.unsqueeze(1), (pad, pad), mode="reflect")  # (B, 1, T+2pad)
    kernel = torch.ones(1, 1, window, device=importance.device, dtype=importance.dtype) / window
    smoothed = F.conv1d(padded, kernel)  # (B, 1, T)
    return smoothed.squeeze(1)


def normalize_importance(importance: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize each sequence to sum to 1.

    Args:
        importance: (B, T).
        eps: numerical stability.

    Returns:
        (B, T) normalized.
    """
    row_sums = importance.sum(dim=-1, keepdim=True).clamp(min=eps)
    return importance / row_sums


# ---------------------------------------------------------------------------
# TokenImportanceScorer
# ---------------------------------------------------------------------------

class TokenImportanceScorer:
    """High-level interface for token importance scoring."""

    def __init__(self, config: ImportanceConfig) -> None:
        self.config = config

    def score(
        self,
        model: nn.Module,
        embeddings: Tensor,
        labels: Tensor,
        loss_fn: Callable,
    ) -> Tensor:
        """Compute (B, T) importance scores using the configured method."""
        if self.config.method == "gradient_norm":
            imp = gradient_norm_importance(model, embeddings, labels, loss_fn)
        else:
            # Default fallback: gradient norm
            imp = gradient_norm_importance(model, embeddings, labels, loss_fn)

        if self.config.smooth_window > 1:
            imp = smooth_importance(imp, self.config.smooth_window)
        if self.config.normalize:
            imp = normalize_importance(imp)
        return imp

    def get_top_tokens(self, importance: Tensor, k: int) -> Tensor:
        """Return (B, k) indices of most important tokens per sequence.

        Args:
            importance: (B, T).
            k: number of top tokens.

        Returns:
            (B, k) int64 indices.
        """
        _, indices = importance.topk(k, dim=-1)
        return indices
