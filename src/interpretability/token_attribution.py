"""Token Attribution — input saliency for Aurelius transformer.

Methods:
  1. gradient          — |∂logit/∂embed(token_i)| at each position
  2. integrated_gradients — path-integrated gradient from zero baseline
  3. attention_rollout — layer-by-layer attention matrix product
  4. erasure           — mask each token and measure logit change
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AttributionConfig:
    method: str = "gradient"      # 'gradient', 'integrated_gradients', 'attention_rollout', 'erasure'
    n_steps: int = 20             # for integrated gradients
    normalize: bool = True
    baseline_type: str = "zero"   # 'zero' (only supported baseline for now)


# ---------------------------------------------------------------------------
# Helper: forward pass returning logits from raw embeddings
# ---------------------------------------------------------------------------

def _forward_from_embeddings(
    model: nn.Module,
    embeddings: Tensor,
    freqs_cis: Tensor,
) -> Tensor:
    """Run transformer layers and lm_head starting from pre-computed embeddings.

    Bypasses model.embed so that gradients flow through 'embeddings' directly.

    Returns:
        logits: (B, S, vocab_size)
    """
    x = embeddings
    mask = None
    present_key_values: list = []
    for layer in model.layers:
        x, kv = layer(x, freqs_cis, mask, None)
        present_key_values.append(kv)
    x = model.norm(x)
    logits = model.lm_head(x)
    return logits


# ---------------------------------------------------------------------------
# Attention weight extraction via hooks
# ---------------------------------------------------------------------------

def _extract_attention_weights(model: nn.Module, input_ids: Tensor) -> List[Tensor]:
    """Run a forward pass and collect per-layer attention weight matrices.

    Registers temporary forward hooks on each GroupedQueryAttention module
    to intercept the softmax attention weights before the output projection.

    Returns:
        List of (1, n_heads, S, S) tensors, one per layer.
    """
    attn_weights: List[Tensor] = []
    hooks = []

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            # inputs[0] is the normed hidden state x: (B, S, d_model)
            x = inputs[0]
            B, S, _ = x.shape

            # Re-compute Q, K manually so we can get softmax weights
            # (Flash Attention doesn't expose them directly)
            n_heads = module.n_heads
            n_kv_heads = module.n_kv_heads
            head_dim = module.head_dim
            n_rep = module.n_rep

            # Project
            q = module.q_proj(x).view(B, S, n_heads, head_dim)
            k = module.k_proj(x).view(B, S, n_kv_heads, head_dim)

            # Apply RoPE — freqs_cis slice for S positions
            from src.model.attention import apply_rope
            freqs_cis_s = model.freqs_cis[:S]
            q = apply_rope(q, freqs_cis_s)
            k = apply_rope(k, freqs_cis_s)

            # Expand KV heads for GQA
            if n_rep > 1:
                k = k.unsqueeze(3).expand(B, S, n_kv_heads, n_rep, head_dim)
                k = k.reshape(B, S, n_heads, head_dim)

            # (B, n_heads, S, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

            scale = head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S, S)

            # Causal mask
            causal_mask = torch.triu(
                torch.full((S, S), float("-inf"), device=scores.device), diagonal=1
            )
            scores = scores + causal_mask

            weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)  # (B, H, S, S)
            attn_weights.append(weights.detach())

        return hook

    for i, layer in enumerate(model.layers):
        h = layer.attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return attn_weights


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TokenAttribution:
    """Input token attribution methods for the Aurelius transformer."""

    def __init__(self, model: nn.Module, method: str = "gradient") -> None:
        self.model = model
        self.method = method

    # ------------------------------------------------------------------
    # Gradient attribution
    # ------------------------------------------------------------------

    def gradient_attribution(
        self,
        input_ids: Tensor,
        target_pos: int,
        target_token: int,
    ) -> Tensor:
        """Gradient of logit[target_pos, target_token] w.r.t. token embeddings.

        Args:
            input_ids: (1, S) or (S,) token ids.
            target_pos: sequence position of the target logit.
            target_token: vocabulary index of the target token.

        Returns:
            (S,) attribution scores (L2 norm of gradient at each position).
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        B, S = input_ids.shape
        assert B == 1, "gradient_attribution expects a single sequence (B=1)"

        # Embed and attach gradient
        embeddings = self.model.embed(input_ids)  # (1, S, d_model)
        embeddings = embeddings.detach().requires_grad_(True)

        freqs_cis = self.model.freqs_cis[:S]

        logits = _forward_from_embeddings(self.model, embeddings, freqs_cis)
        # logits: (1, S, vocab_size)

        target_logit = logits[0, target_pos, target_token]
        target_logit.backward()

        grad = embeddings.grad  # (1, S, d_model)
        if grad is None:
            return torch.zeros(S)

        scores = grad[0].norm(dim=-1)  # (S,)
        return scores.detach()

    # ------------------------------------------------------------------
    # Integrated gradients
    # ------------------------------------------------------------------

    def integrated_gradients(
        self,
        input_ids: Tensor,
        target_pos: int,
        target_token: int,
        n_steps: int = 20,
    ) -> Tensor:
        """Integrated gradients: average gradient along interpolation from baseline.

        Args:
            input_ids: (1, S) token ids.
            target_pos: position of target logit.
            target_token: vocabulary index of target.
            n_steps: number of integration steps.

        Returns:
            (S,) attribution scores.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        B, S = input_ids.shape
        assert B == 1

        actual_emb = self.model.embed(input_ids).detach()  # (1, S, d_model)
        baseline = torch.zeros_like(actual_emb)            # zero baseline

        freqs_cis = self.model.freqs_cis[:S]
        grad_sum = torch.zeros_like(actual_emb)

        for step in range(n_steps):
            alpha = step / max(n_steps - 1, 1)
            interp = (baseline + alpha * (actual_emb - baseline)).detach().requires_grad_(True)

            logits = _forward_from_embeddings(self.model, interp, freqs_cis)
            target_logit = logits[0, target_pos, target_token]
            target_logit.backward()

            if interp.grad is not None:
                grad_sum = grad_sum + interp.grad.detach()

        avg_grad = grad_sum / n_steps
        attributions = avg_grad * (actual_emb - baseline)  # (1, S, d_model)
        scores = attributions[0].norm(dim=-1)               # (S,)
        return scores.detach()

    # ------------------------------------------------------------------
    # Attention rollout
    # ------------------------------------------------------------------

    def attention_rollout(self, input_ids: Tensor) -> Tensor:
        """Propagate attention matrices layer-by-layer with residual blending.

        A_rollout = A_L @ ... @ A_2 @ A_1
        Each layer: A_i = 0.5 * A_i + 0.5 * I, then row-normalize.

        Args:
            input_ids: (1, S) or (S,) token ids.

        Returns:
            (S, S) rollout matrix (row = output position, col = input position).
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        B, S = input_ids.shape
        assert B == 1

        # Collect per-layer attention weights: list of (1, H, S, S)
        attn_weights = _extract_attention_weights(self.model, input_ids)

        eye = torch.eye(S, device=input_ids.device)

        # Start with first layer, averaged over heads: (S, S)
        rollout = attn_weights[0][0].mean(dim=0)  # (S, S)
        rollout = 0.5 * rollout + 0.5 * eye
        rollout = rollout / rollout.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        for attn in attn_weights[1:]:
            layer_mat = attn[0].mean(dim=0)  # (S, S)
            layer_mat = 0.5 * layer_mat + 0.5 * eye
            layer_mat = layer_mat / layer_mat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            rollout = torch.matmul(layer_mat, rollout)

        return rollout.detach()

    # ------------------------------------------------------------------
    # Erasure (occlusion) attribution
    # ------------------------------------------------------------------

    def erasure_attribution(
        self,
        input_ids: Tensor,
        target_pos: int,
        target_token: int,
    ) -> Tensor:
        """Mask each token and measure drop in target logit.

        For each position i, replace token i with 0 (pad) and measure
        |logit_original - logit_masked|.

        Args:
            input_ids: (1, S) or (S,) token ids.
            target_pos: position of target logit.
            target_token: vocabulary index.

        Returns:
            (S,) attribution scores (non-negative).
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        B, S = input_ids.shape
        assert B == 1

        with torch.no_grad():
            _, logits_orig, _ = self.model(input_ids)
            baseline_logit = logits_orig[0, target_pos, target_token].item()

        scores = torch.zeros(S)
        for i in range(S):
            masked = input_ids.clone()
            masked[0, i] = 0  # replace with pad token id 0
            with torch.no_grad():
                _, logits_masked, _ = self.model(masked)
            masked_logit = logits_masked[0, target_pos, target_token].item()
            scores[i] = abs(baseline_logit - masked_logit)

        return scores

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------

    def normalize_attributions(self, scores: Tensor) -> Tensor:
        """Normalize attribution scores to [0, 1].

        Args:
            scores: (S,) or (S, S) attribution tensor.

        Returns:
            Tensor of same shape with values in [0, 1].
        """
        s_min = scores.min()
        s_max = scores.max()
        if (s_max - s_min).abs() < 1e-8:
            return torch.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)


# ---------------------------------------------------------------------------
# top_k_tokens helper
# ---------------------------------------------------------------------------

def top_k_tokens(
    attributions: Tensor,
    k: int,
    input_ids: Tensor,
) -> List[Tuple[int, int, float]]:
    """Return the top-k attributed tokens.

    Args:
        attributions: (S,) attribution scores.
        k: number of top tokens to return.
        input_ids: (S,) or (1, S) token ids.

    Returns:
        List of (token_id, position, score) tuples sorted by descending score.
    """
    if input_ids.dim() == 2:
        input_ids = input_ids[0]

    S = attributions.shape[0]
    k = min(k, S)

    top_scores, top_positions = attributions.topk(k)
    result: List[Tuple[int, int, float]] = []
    for pos, score in zip(top_positions.tolist(), top_scores.tolist()):
        token_id = int(input_ids[pos].item())
        result.append((token_id, int(pos), float(score)))
    return result
