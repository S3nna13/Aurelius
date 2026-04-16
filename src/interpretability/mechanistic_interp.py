"""Mechanistic Interpretability toolkit for Aurelius.

Based on Elhage et al. 2022 "Toy Models of Superposition" and Anthropic
mechanistic interpretability research.

Pure PyTorch only -- no HuggingFace, no sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MechInterpConfig:
    n_singular_vectors: int = 8
    superposition_threshold: float = 0.1
    layer_idx: int = -1


def analyze_weight_matrix(
    W: Tensor,
    n_singular_vectors: int = 8,
) -> dict:
    """SVD analysis of a weight matrix.

    Returns:
        {
          'singular_values': (min(m,n),),
          'effective_rank': float,
          'condition_number': float,
          'left_singular_vectors': (m, k),
          'right_singular_vectors': (k, n),
        }
    """
    W_float = W.float()
    U, S, Vh = torch.linalg.svd(W_float, full_matrices=False)

    k = min(n_singular_vectors, S.shape[0])

    sum_s = S.sum()
    sum_s2 = (S ** 2).sum()
    effective_rank = float((sum_s ** 2) / (sum_s2 + 1e-12))

    condition_number = float(S[0] / (S[-1] + 1e-12))

    return {
        "singular_values": S,
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "left_singular_vectors": U[:, :k],
        "right_singular_vectors": Vh[:k, :],
    }


def detect_superposition(
    W: Tensor,
    threshold: float = 0.1,
) -> dict:
    """Detect superposition: when d_features > d_model, features share dimensions.

    Metric: fraction of singular values below threshold * s_max.

    Returns:
        {
          'superposition_score': float,
          'has_superposition': bool,
          'n_active_dimensions': int,
          'compression_ratio': float,
        }
    """
    W_float = W.float()
    _, S, _ = torch.linalg.svd(W_float, full_matrices=False)

    s_max = float(S[0])
    cutoff = threshold * s_max

    n_total = S.shape[0]
    n_active = int((S > cutoff).sum().item())
    n_small = n_total - n_active

    superposition_score = float(n_small) / float(n_total)
    has_superposition = superposition_score > 0.0
    compression_ratio = float(W.shape[0]) / max(n_active, 1)

    return {
        "superposition_score": superposition_score,
        "has_superposition": has_superposition,
        "n_active_dimensions": n_active,
        "compression_ratio": compression_ratio,
    }


def _recompute_attention_weights(
    attn_module: nn.Module,
    x: Tensor,
    freqs_cis: Tensor,
) -> Tensor:
    """Manually recompute (n_heads, S, S) softmax attention weights."""
    from src.model.attention import apply_rope

    B, S, _ = x.shape
    n_heads = attn_module.n_heads
    n_kv_heads = attn_module.n_kv_heads
    head_dim = attn_module.head_dim
    n_rep = n_heads // n_kv_heads

    q = attn_module.q_proj(x).view(B, S, n_heads, head_dim)
    k = attn_module.k_proj(x).view(B, S, n_kv_heads, head_dim)

    q = apply_rope(q, freqs_cis)
    k = apply_rope(k, freqs_cis)

    if n_rep > 1:
        k = k.unsqueeze(3).expand(B, S, n_kv_heads, n_rep, head_dim)
        k = k.reshape(B, S, n_heads, head_dim)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    scale = head_dim ** -0.5
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale

    causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
    attn_logits = attn_logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn_weights = torch.softmax(attn_logits, dim=-1)
    return attn_weights[0]


def compute_attention_pattern_similarity(
    model: nn.Module,
    input_ids: Tensor,
    layer_idx: int = 0,
) -> Tensor:
    """Extract attention weights from a specific layer.

    Returns (n_heads, T, T) attention pattern matrix.
    """
    model.eval()
    with torch.no_grad():
        B, T = input_ids.shape
        x = model.embed(input_ids)
        freqs_cis = model.freqs_cis[:T]

        for i, layer in enumerate(model.layers):
            normed = layer.attn_norm(x)
            if i == layer_idx:
                attn_weights = _recompute_attention_weights(layer.attn, normed, freqs_cis)
                break
            attn_out, _ = layer.attn(normed, freqs_cis)
            x = x + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))

    return attn_weights


def logit_lens_analysis(
    model: nn.Module,
    input_ids: Tensor,
    target_token: int,
) -> Tensor:
    """Apply logit lens at each layer.

    Returns (n_layers,) tensor of logit for target_token at each layer.
    """
    model.eval()
    with torch.no_grad():
        B, T = input_ids.shape
        x = model.embed(input_ids)
        freqs_cis = model.freqs_cis[:T]

        logits_per_layer = []
        for layer in model.layers:
            attn_out, _ = layer.attn(layer.attn_norm(x), freqs_cis)
            x = x + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))

            normed = model.norm(x)
            logits = model.lm_head(normed)
            layer_logit = logits[0, -1, target_token]
            logits_per_layer.append(layer_logit)

    return torch.stack(logits_per_layer)


def identify_attention_heads_by_type(
    model: nn.Module,
    input_ids: Tensor,
) -> dict:
    """Identify attention head types by their attention patterns.

    Returns dict with keys:
    - 'induction': heads attending to [prev token of repeated token]
    - 'previous_token': heads mostly attending to position t-1
    - 'current_token': heads mostly attending to current position (diagonal)

    Each value is list of (layer_idx, head_idx) tuples.
    """
    model.eval()
    _, T = input_ids.shape

    result = {
        "induction": [],
        "previous_token": [],
        "current_token": [],
    }

    with torch.no_grad():
        x = model.embed(input_ids)
        freqs_cis = model.freqs_cis[:T]

        for layer_idx, layer in enumerate(model.layers):
            normed = layer.attn_norm(x)
            attn_weights = _recompute_attention_weights(layer.attn, normed, freqs_cis)
            n_heads = attn_weights.shape[0]

            for head_idx in range(n_heads):
                pat = attn_weights[head_idx]

                if T > 0:
                    diag_score = pat.diagonal().mean().item()
                    if diag_score > 0.5:
                        result["current_token"].append((layer_idx, head_idx))
                        continue

                if T > 1:
                    prev_score = pat[1:, :-1].diagonal().mean().item()
                    if prev_score > 0.4:
                        result["previous_token"].append((layer_idx, head_idx))
                        continue

                if T >= 4:
                    half = T // 2
                    induction_score = 0.0
                    count = 0
                    for t in range(half, T):
                        mirror = t - half
                        if mirror >= 0:
                            induction_score += pat[t, mirror].item()
                            count += 1
                    if count > 0 and induction_score / count > 0.3:
                        result["induction"].append((layer_idx, head_idx))

            attn_out, _ = layer.attn(layer.attn_norm(x), freqs_cis)
            x = x + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))

    return result


def compute_residual_stream_contributions(
    model: nn.Module,
    input_ids: Tensor,
) -> dict:
    """Decompose residual stream into contributions per layer.

    Returns:
        {
          'embedding': (T, d_model),
          'layer_0': (T, d_model),
          'layer_1': ...,
        }
    """
    model.eval()
    with torch.no_grad():
        B, T = input_ids.shape
        x = model.embed(input_ids)
        freqs_cis = model.freqs_cis[:T]

        contributions = {
            "embedding": x[0].clone(),
        }

        for i, layer in enumerate(model.layers):
            attn_out, _ = layer.attn(layer.attn_norm(x), freqs_cis)
            x = x + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))
            contributions[f"layer_{i}"] = x[0].clone()

    return contributions
