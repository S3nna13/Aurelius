"""Tests for src/model/token_merging.py — ToMe-inspired token merging module."""

from __future__ import annotations

import torch
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.model.token_merging import (
    ToMeConfig,
    compute_token_similarity,
    find_merge_pairs,
    merge_tokens,
    unmerge_tokens,
    ToMeLayer,
    apply_tome,
    estimate_speedup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tiny_cfg(**kwargs) -> AureliusConfig:
    """Minimal AureliusConfig for fast tests."""
    defaults = dict(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return AureliusConfig(**defaults)


def tiny_model() -> AureliusTransformer:
    return AureliusTransformer(tiny_cfg())


B, T, D = 1, 8, 64


# ---------------------------------------------------------------------------
# 1. ToMeConfig defaults
# ---------------------------------------------------------------------------

def test_tome_config_defaults():
    cfg = ToMeConfig()
    assert cfg.r == 8
    assert cfg.merge_mode == "mean"
    assert cfg.similarity_metric == "cosine"
    assert cfg.layer_start == 0


# ---------------------------------------------------------------------------
# 2. compute_token_similarity — cosine shape (B, T, T)
# ---------------------------------------------------------------------------

def test_compute_similarity_cosine_shape():
    x = torch.randn(B, T, D)
    sim = compute_token_similarity(x, metric="cosine")
    assert sim.shape == (B, T, T), f"Expected ({B}, {T}, {T}), got {sim.shape}"


# ---------------------------------------------------------------------------
# 3. compute_token_similarity — cosine diagonal approximately 1.0
# ---------------------------------------------------------------------------

def test_compute_similarity_cosine_diagonal():
    torch.manual_seed(0)
    x = torch.randn(B, T, D)
    sim = compute_token_similarity(x, metric="cosine")
    diag = sim[0].diagonal()
    assert torch.allclose(diag, torch.ones(T), atol=1e-5), (
        f"Cosine diagonal should be ~1.0, got: {diag}"
    )


# ---------------------------------------------------------------------------
# 4. compute_token_similarity — dot product shape (B, T, T)
# ---------------------------------------------------------------------------

def test_compute_similarity_dot_shape():
    x = torch.randn(B, T, D)
    sim = compute_token_similarity(x, metric="dot")
    assert sim.shape == (B, T, T), f"Expected ({B}, {T}, {T}), got {sim.shape}"


# ---------------------------------------------------------------------------
# 5. compute_token_similarity — l2 shape (B, T, T)
# ---------------------------------------------------------------------------

def test_compute_similarity_l2_shape():
    x = torch.randn(B, T, D)
    sim = compute_token_similarity(x, metric="l2")
    assert sim.shape == (B, T, T), f"Expected ({B}, {T}, {T}), got {sim.shape}"


# ---------------------------------------------------------------------------
# 6. find_merge_pairs — returns at most r pairs
# ---------------------------------------------------------------------------

def test_find_merge_pairs_count():
    torch.manual_seed(0)
    x = torch.randn(1, T, D)
    sim = compute_token_similarity(x, metric="cosine")
    r = 3
    pairs = find_merge_pairs(sim, r=r)
    assert len(pairs) <= r, f"Expected at most {r} pairs, got {len(pairs)}"


# ---------------------------------------------------------------------------
# 7. find_merge_pairs — pairs are adjacent (|i-j| == 1)
# ---------------------------------------------------------------------------

def test_find_merge_pairs_adjacent():
    torch.manual_seed(1)
    x = torch.randn(1, T, D)
    sim = compute_token_similarity(x, metric="cosine")
    pairs = find_merge_pairs(sim, r=3)
    for i, j in pairs:
        assert abs(i - j) == 1, f"Pair ({i}, {j}) is not adjacent"


# ---------------------------------------------------------------------------
# 8. merge_tokens — reduces sequence length by len(pairs)
# ---------------------------------------------------------------------------

def test_merge_tokens_reduces_length():
    torch.manual_seed(2)
    x = torch.randn(B, T, D)
    sim = compute_token_similarity(x, metric="cosine")
    pairs = find_merge_pairs(sim, r=2)
    merged = merge_tokens(x, pairs, mode="mean")
    expected_len = T - len(pairs)
    assert merged.shape == (B, expected_len, D), (
        f"Expected ({B}, {expected_len}, {D}), got {merged.shape}"
    )


# ---------------------------------------------------------------------------
# 9. merge_tokens — mode="mean" gives correct average
# ---------------------------------------------------------------------------

def test_merge_tokens_mean_value():
    # Construct x so pair (0, 1) has a known merge result
    x = torch.zeros(1, 4, 4)
    x[0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x[0, 1] = torch.tensor([3.0, 4.0, 5.0, 6.0])

    pairs = [(0, 1)]
    merged = merge_tokens(x, pairs, mode="mean")

    expected = torch.tensor([2.0, 3.0, 4.0, 5.0])
    assert torch.allclose(merged[0, 0], expected, atol=1e-6), (
        f"Mean merge incorrect: {merged[0, 0]} != {expected}"
    )
    # Sequence length reduced by 1
    assert merged.shape[1] == 3


# ---------------------------------------------------------------------------
# 10. unmerge_tokens — restores original sequence length
# ---------------------------------------------------------------------------

def test_unmerge_tokens_restores_length():
    torch.manual_seed(3)
    x = torch.randn(B, T, D)
    sim = compute_token_similarity(x, metric="cosine")
    pairs = find_merge_pairs(sim, r=2)
    merged = merge_tokens(x, pairs, mode="mean")
    restored = unmerge_tokens(merged, pairs, original_len=T)
    assert restored.shape == (B, T, D), (
        f"Expected ({B}, {T}, {D}), got {restored.shape}"
    )


# ---------------------------------------------------------------------------
# 11. ToMeLayer — forward returns correct output shape
# ---------------------------------------------------------------------------

def test_tome_layer_forward_shape():
    torch.manual_seed(4)
    model = tiny_model()
    base_layer = model.layers[0]
    cfg = ToMeConfig(r=2, layer_start=0)
    tome_layer = ToMeLayer(base_layer, config=cfg, layer_idx=0)

    from src.model.attention import precompute_rope_frequencies
    freqs_cis = precompute_rope_frequencies(32, T)

    x = torch.randn(B, T, D)
    out = tome_layer(x, freqs_cis=freqs_cis)

    # TransformerBlock returns tuple; extract hidden
    hidden = out[0] if isinstance(out, tuple) else out
    assert hidden.shape == (B, T, D), (
        f"ToMeLayer output shape {hidden.shape} != expected ({B}, {T}, {D})"
    )


# ---------------------------------------------------------------------------
# 12. apply_tome — wraps model layers with ToMeLayer
# ---------------------------------------------------------------------------

def test_apply_tome_wraps_layers():
    model = tiny_model()
    cfg = ToMeConfig(r=2)
    apply_tome(model, cfg)
    for i, layer in enumerate(model.layers):
        assert isinstance(layer, ToMeLayer), (
            f"Layer {i} is {type(layer)}, expected ToMeLayer"
        )


# ---------------------------------------------------------------------------
# 13. apply_tome — model still produces output of same logit shape
# ---------------------------------------------------------------------------

def test_apply_tome_model_output_shape():
    torch.manual_seed(5)
    cfg_model = tiny_cfg()
    model = AureliusTransformer(cfg_model)
    tome_cfg = ToMeConfig(r=1, layer_start=0)
    apply_tome(model, tome_cfg)

    input_ids = torch.randint(0, cfg_model.vocab_size, (B, T))
    with torch.no_grad():
        loss, logits, pkv = model(input_ids)

    assert logits.shape == (B, T, cfg_model.vocab_size), (
        f"Logits shape {logits.shape} != ({B}, {T}, {cfg_model.vocab_size})"
    )


# ---------------------------------------------------------------------------
# 14. estimate_speedup — returns float >= 1.0
# ---------------------------------------------------------------------------

def test_estimate_speedup_ge_one():
    result = estimate_speedup(original_seq_len=64, r=4, n_layers=4)
    assert isinstance(result, float), "estimate_speedup should return a float"
    assert result >= 1.0, f"Speedup {result} should be >= 1.0"


# ---------------------------------------------------------------------------
# 15. estimate_speedup — increases with more layers
# ---------------------------------------------------------------------------

def test_estimate_speedup_increases_with_layers():
    base = estimate_speedup(original_seq_len=64, r=4, n_layers=2)
    more = estimate_speedup(original_seq_len=64, r=4, n_layers=6)
    assert more >= base, (
        f"Speedup with more layers ({more}) should be >= fewer layers ({base})"
    )


# ---------------------------------------------------------------------------
# 16. ToMeLayer — skips merging when layer_idx < layer_start
# ---------------------------------------------------------------------------

def test_tome_layer_skips_below_layer_start():
    """When layer_idx < layer_start, output should equal base_layer output."""
    torch.manual_seed(6)
    model = tiny_model()
    base_layer = model.layers[0]
    cfg = ToMeConfig(r=2, layer_start=5)  # layer 0 is below start
    tome_layer = ToMeLayer(base_layer, config=cfg, layer_idx=0)

    from src.model.attention import precompute_rope_frequencies
    freqs_cis = precompute_rope_frequencies(32, T)
    x = torch.randn(B, T, D)

    out_tome = tome_layer(x, freqs_cis=freqs_cis)
    out_base = base_layer(x, freqs_cis=freqs_cis)

    hidden_tome = out_tome[0] if isinstance(out_tome, tuple) else out_tome
    hidden_base = out_base[0] if isinstance(out_base, tuple) else out_base
    assert torch.allclose(hidden_tome, hidden_base, atol=1e-6), (
        "ToMeLayer should skip merging when layer_idx < layer_start"
    )


# ---------------------------------------------------------------------------
# 17. find_merge_pairs — returns fewer pairs than r when T < 2r
# ---------------------------------------------------------------------------

def test_find_merge_pairs_small_sequence():
    """With T=4 and r=10, we get at most (4-1)//2 = 1 pair."""
    torch.manual_seed(7)
    x = torch.randn(1, 4, D)
    sim = compute_token_similarity(x, metric="cosine")
    pairs = find_merge_pairs(sim, r=10)
    assert len(pairs) <= (4 - 1) // 2, (
        f"Expected at most 1 pair for T=4, got {len(pairs)}"
    )
