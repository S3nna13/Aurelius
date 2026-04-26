"""Tests for src/model/token_pruning.py."""

from __future__ import annotations

import math

import pytest
import torch

from src.model.token_pruning import (
    AdaptiveTokenPruner,
    TokenPruningConfig,
    TokenPruningLayer,
    evaluate_pruning_efficiency,
    score_tokens_by_attention,
    score_tokens_by_gradient,
    select_important_tokens,
)

# ---------------------------------------------------------------------------
# Shared test dimensions
# ---------------------------------------------------------------------------
B = 2
T = 16
D = 32
N_HEADS = 4
KEEP_RATIO = 0.5
MIN_TOKENS = 2

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------
def test_token_pruning_config_defaults():
    cfg = TokenPruningConfig()
    assert cfg.keep_ratio == 0.5
    assert cfg.scoring_method == "attention"
    assert cfg.min_tokens == 1
    assert cfg.protect_positions == []


# ---------------------------------------------------------------------------
# 2. score_tokens_by_attention — output shape (B, T)
# ---------------------------------------------------------------------------
def test_score_tokens_by_attention_shape():
    torch.manual_seed(0)
    attn = torch.rand(B, N_HEADS, T, T)
    scores = score_tokens_by_attention(attn)
    assert scores.shape == (B, T), f"Expected ({B}, {T}), got {scores.shape}"


# ---------------------------------------------------------------------------
# 3. score_tokens_by_attention — all scores >= 0
# ---------------------------------------------------------------------------
def test_score_tokens_by_attention_positive():
    torch.manual_seed(0)
    attn = torch.rand(B, N_HEADS, T, T).abs()  # ensure non-negative weights
    scores = score_tokens_by_attention(attn)
    assert (scores >= 0).all(), "All attention-based scores should be non-negative"


# ---------------------------------------------------------------------------
# 4. score_tokens_by_gradient — output shape (B, T)
# ---------------------------------------------------------------------------
def test_score_tokens_by_gradient_shape():
    torch.manual_seed(0)
    hidden = torch.randn(B, T, D)
    grad = torch.randn(B, T, D)
    scores = score_tokens_by_gradient(hidden, grad)
    assert scores.shape == (B, T), f"Expected ({B}, {T}), got {scores.shape}"


# ---------------------------------------------------------------------------
# 5. select_important_tokens — approximately keep_ratio fraction kept
# ---------------------------------------------------------------------------
def test_select_important_tokens_ratio():
    torch.manual_seed(0)
    cfg = TokenPruningConfig(keep_ratio=KEEP_RATIO, min_tokens=MIN_TOKENS)
    scores = torch.randn(B, T)
    mask = select_important_tokens(scores, cfg)

    expected_k = max(MIN_TOKENS, math.ceil(KEEP_RATIO * T))
    for b in range(B):
        kept = mask[b].sum().item()
        # At least expected_k (protected positions could add more, but here none)
        assert kept >= expected_k, f"Batch {b}: kept {kept} < expected {expected_k}"


# ---------------------------------------------------------------------------
# 6. select_important_tokens — protected positions always kept
# ---------------------------------------------------------------------------
def test_select_important_tokens_protect():
    torch.manual_seed(0)
    protect = [0, 3, 7]
    cfg = TokenPruningConfig(keep_ratio=0.1, min_tokens=1, protect_positions=protect)
    # Give protected positions very low scores so they would normally be pruned
    scores = torch.full((B, T), 1.0)
    for p in protect:
        scores[:, p] = -999.0

    mask = select_important_tokens(scores, cfg)
    for p in protect:
        assert mask[:, p].all(), f"Position {p} should always be kept"


# ---------------------------------------------------------------------------
# 7. select_important_tokens — at least min_tokens kept
# ---------------------------------------------------------------------------
def test_select_important_tokens_min_tokens():
    torch.manual_seed(0)
    min_tok = 5
    cfg = TokenPruningConfig(keep_ratio=0.0, min_tokens=min_tok)
    scores = torch.randn(B, T)
    mask = select_important_tokens(scores, cfg)

    for b in range(B):
        kept = mask[b].sum().item()
        assert kept >= min_tok, f"Batch {b}: kept {kept} < min_tokens {min_tok}"


# ---------------------------------------------------------------------------
# 8. TokenPruningLayer — output same shape as input
# ---------------------------------------------------------------------------
def test_token_pruning_layer_shape():
    torch.manual_seed(0)
    cfg = TokenPruningConfig(keep_ratio=KEEP_RATIO, min_tokens=MIN_TOKENS)
    layer = TokenPruningLayer(cfg)
    hidden = torch.randn(B, T, D)
    attn = torch.rand(B, N_HEADS, T, T)

    pruned, mask = layer(hidden, attn)
    assert pruned.shape == (B, T, D), f"Pruned shape mismatch: {pruned.shape}"
    assert mask.shape == (B, T), f"Mask shape mismatch: {mask.shape}"


# ---------------------------------------------------------------------------
# 9. TokenPruningLayer — correct number of True values in mask
# ---------------------------------------------------------------------------
def test_token_pruning_layer_mask_sum():
    torch.manual_seed(0)
    cfg = TokenPruningConfig(keep_ratio=KEEP_RATIO, min_tokens=MIN_TOKENS)
    layer = TokenPruningLayer(cfg)
    hidden = torch.randn(B, T, D)
    attn = torch.rand(B, N_HEADS, T, T)

    _, mask = layer(hidden, attn)
    expected_k = max(MIN_TOKENS, math.ceil(KEEP_RATIO * T))
    for b in range(B):
        kept = mask[b].sum().item()
        assert kept >= expected_k, (
            f"Batch {b}: mask has {kept} True values, expected >= {expected_k}"
        )


# ---------------------------------------------------------------------------
# 10. AdaptiveTokenPruner — all outputs have correct shapes
# ---------------------------------------------------------------------------
def test_adaptive_token_pruner_shapes():
    torch.manual_seed(0)
    cfg = TokenPruningConfig(keep_ratio=KEEP_RATIO, min_tokens=MIN_TOKENS)
    pruner = AdaptiveTokenPruner(d_model=D, config=cfg)
    hidden = torch.randn(B, T, D)

    pruned, mask, scores = pruner(hidden)
    assert pruned.shape == (B, T, D), f"pruned_hidden shape: {pruned.shape}"
    assert mask.shape == (B, T), f"keep_mask shape: {mask.shape}"
    assert scores.shape == (B, T), f"scores shape: {scores.shape}"


# ---------------------------------------------------------------------------
# 11. evaluate_pruning_efficiency — dict has expected keys
# ---------------------------------------------------------------------------
def test_evaluate_pruning_efficiency_keys():
    result = evaluate_pruning_efficiency(original_seq_len=T, keep_ratio=KEEP_RATIO)
    expected_keys = {"kept_tokens", "pruned_tokens", "theoretical_speedup"}
    assert set(result.keys()) == expected_keys, (
        f"Keys mismatch: got {set(result.keys())}, expected {expected_keys}"
    )
    # Sanity-check values
    assert result["kept_tokens"] + result["pruned_tokens"] == T
    assert result["theoretical_speedup"] == pytest.approx(1.0 / KEEP_RATIO**2)
