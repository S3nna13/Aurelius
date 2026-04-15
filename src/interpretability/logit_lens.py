"""
Logit Lens interpretability tool for the Aurelius LLM project.

Projects intermediate hidden states through the final unembedding matrix
to inspect which tokens the model predicts at each layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LogitLensConfig:
    """Configuration for the LogitLens analysis."""
    vocab_size: int = 50257
    d_model: int = 512
    n_layers: int = 12
    apply_ln: bool = True


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------

def _layer_norm(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float = 1e-5,
) -> Tensor:
    """Apply LayerNorm manually (pure PyTorch, no nn.LayerNorm dependency)."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / (var + eps).sqrt()
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    return x_norm


def project_hidden_state(
    hidden: Tensor,
    unembed: Tensor,
    ln_weight: Optional[Tensor] = None,
    ln_bias: Optional[Tensor] = None,
) -> Tensor:
    """Project a hidden state tensor through the unembedding matrix.

    Args:
        hidden:    (B, T, d_model) hidden states.
        unembed:   (vocab_size, d_model) unembedding / lm_head weight matrix.
        ln_weight: Optional (d_model,) LayerNorm scale applied before projection.
        ln_bias:   Optional (d_model,) LayerNorm bias applied before projection.

    Returns:
        (B, T, vocab_size) raw logits.
    """
    if ln_weight is not None or ln_bias is not None:
        hidden = _layer_norm(hidden, ln_weight, ln_bias)
    # (B, T, d_model) @ (d_model, vocab_size) -> (B, T, vocab_size)
    logits = hidden @ unembed.T
    return logits


def get_top_tokens(logits: Tensor, k: int = 5) -> Tensor:
    """Return the top-k token indices at each (batch, sequence) position.

    Args:
        logits: (B, T, vocab_size) logits.
        k:      Number of top tokens to return.

    Returns:
        (B, T, k) integer tensor of top token indices (highest logit first).
    """
    # torch.topk returns (values, indices)
    _, indices = torch.topk(logits, k=k, dim=-1)
    return indices


def compute_kl_divergence(p_logits: Tensor, q_logits: Tensor) -> Tensor:
    """Compute KL divergence KL(softmax(p) || softmax(q)) per (B, T) position.

    Args:
        p_logits: (B, T, vocab_size) logits for distribution P.
        q_logits: (B, T, vocab_size) logits for distribution Q.

    Returns:
        (B, T) tensor of KL divergence values (in nats).
    """
    log_p = F.log_softmax(p_logits, dim=-1)  # (B, T, V)
    log_q = F.log_softmax(q_logits, dim=-1)  # (B, T, V)
    p = log_p.exp()                           # (B, T, V)
    # KL(P || Q) = Σ p * (log_p - log_q)
    kl = (p * (log_p - log_q)).sum(dim=-1)   # (B, T)
    return kl


# ---------------------------------------------------------------------------
# LogitLens class
# ---------------------------------------------------------------------------

class LogitLens:
    """Logit Lens: project hidden states at every layer to vocabulary logits."""

    def __init__(
        self,
        config: LogitLensConfig,
        unembed_weight: Tensor,
        ln_weight: Optional[Tensor] = None,
        ln_bias: Optional[Tensor] = None,
    ) -> None:
        """
        Args:
            config:         LogitLensConfig with model hyper-parameters.
            unembed_weight: (vocab_size, d_model) unembedding matrix.
            ln_weight:      Optional final LayerNorm scale (d_model,).
            ln_bias:        Optional final LayerNorm bias (d_model,).
        """
        self.config = config
        self.unembed_weight = unembed_weight
        self.ln_weight = ln_weight if config.apply_ln else None
        self.ln_bias = ln_bias if config.apply_ln else None

    def analyze_layer(self, hidden: Tensor) -> Tensor:
        """Project a single layer's hidden state to logits.

        Args:
            hidden: (B, T, d_model) hidden states for one layer.

        Returns:
            (B, T, vocab_size) logits.
        """
        return project_hidden_state(
            hidden, self.unembed_weight, self.ln_weight, self.ln_bias
        )

    def analyze_all_layers(self, hiddens: List[Tensor]) -> Tensor:
        """Project every layer's hidden states to logits.

        Args:
            hiddens: List of n_layers tensors, each (B, T, d_model).

        Returns:
            (n_layers, B, T, vocab_size) stacked logits.
        """
        layer_logits = [self.analyze_layer(h) for h in hiddens]
        return torch.stack(layer_logits, dim=0)  # (n_layers, B, T, V)

    def get_prediction_depth(
        self,
        hiddens: List[Tensor],
        target_ids: Tensor,
        k: int = 1,
    ) -> Tensor:
        """Find the first layer at which each target token appears in the top-k.

        Args:
            hiddens:    List of n_layers tensors, each (B, T, d_model).
            target_ids: (B, T) integer tensor of target token ids.
            k:          How many top predictions to consider per position.

        Returns:
            (B, T) integer tensor. Entry is the zero-based layer index of the
            first layer where the target token is in the top-k predictions.
            -1 if the target is never in top-k across all layers.
        """
        all_logits = self.analyze_all_layers(hiddens)  # (L, B, T, V)
        n_layers, B, T, _ = all_logits.shape

        top_indices = get_top_tokens(all_logits.view(n_layers * B, T, -1), k=k)
        # Reshape back: (L, B, T, k)
        top_indices = top_indices.view(n_layers, B, T, k)

        # target_ids: (B, T) -> broadcast to (L, B, T, 1) for comparison
        target = target_ids.unsqueeze(0).unsqueeze(-1)  # (1, B, T, 1)
        matches = (top_indices == target).any(dim=-1)   # (L, B, T) bool

        # For each (b, t), find the first layer (dim=0) where match is True
        # torch.argmax on bool gives first True; but if no True exists, result is 0
        # We detect the "never matched" case separately.
        any_match = matches.any(dim=0)  # (B, T)
        depth = matches.long().argmax(dim=0)  # (B, T)  — first True layer index
        depth[~any_match] = -1
        return depth


# ---------------------------------------------------------------------------
# LayerwiseEntropyTracker
# ---------------------------------------------------------------------------

class LayerwiseEntropyTracker:
    """Track per-layer entropy of the logit-lens distributions."""

    def __init__(self) -> None:
        pass

    def compute_entropy(self, logits: Tensor) -> Tensor:
        """Compute Shannon entropy H = -Σ p * log(p) in nats.

        Args:
            logits: (B, T, V) raw logits.

        Returns:
            (B, T) entropy tensor.
        """
        log_p = F.log_softmax(logits, dim=-1)  # (B, T, V)
        p = log_p.exp()                         # (B, T, V)
        entropy = -(p * log_p).sum(dim=-1)      # (B, T)
        return entropy

    def track(self, lens: LogitLens, hiddens: List[Tensor]) -> Tensor:
        """Compute entropy at every layer.

        Args:
            lens:    A LogitLens instance.
            hiddens: List of n_layers tensors, each (B, T, d_model).

        Returns:
            (n_layers, B, T) entropy tensor.
        """
        all_logits = lens.analyze_all_layers(hiddens)  # (L, B, T, V)
        n_layers, B, T, V = all_logits.shape

        # Compute entropy for each layer
        entropies = []
        for l in range(n_layers):
            entropies.append(self.compute_entropy(all_logits[l]))  # (B, T)
        return torch.stack(entropies, dim=0)  # (L, B, T)
