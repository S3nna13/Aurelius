"""Tests for src/interpretability/mechanistic_interp.py

Tiny config: n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
             head_dim=16, d_ff=128, vocab_size=256, max_seq_len=64
"""

import torch

from src.interpretability.mechanistic_interp import (
    analyze_weight_matrix,
    compute_attention_pattern_similarity,
    compute_residual_stream_contributions,
    detect_superposition,
    identify_attention_heads_by_type,
    logit_lens_analysis,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared tiny fixtures
# ---------------------------------------------------------------------------

torch.manual_seed(42)

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

SEQ_LEN = 8


def _make_model() -> AureliusTransformer:
    torch.manual_seed(0)
    return AureliusTransformer(TINY_CFG).eval()


def _make_input() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 256, (1, SEQ_LEN))


# ---------------------------------------------------------------------------
# 1. analyze_weight_matrix — returns dict with all required keys
# ---------------------------------------------------------------------------


def test_analyze_weight_matrix_keys():
    torch.manual_seed(7)
    W = torch.randn(32, 16)
    result = analyze_weight_matrix(W)
    required = {
        "singular_values",
        "effective_rank",
        "condition_number",
        "left_singular_vectors",
        "right_singular_vectors",
    }
    assert required.issubset(result.keys())


# ---------------------------------------------------------------------------
# 2. singular_values shape is (min(m,n),)
# ---------------------------------------------------------------------------


def test_analyze_weight_matrix_singular_values_shape():
    torch.manual_seed(7)
    W = torch.randn(32, 16)
    result = analyze_weight_matrix(W)
    s = result["singular_values"]
    assert s.shape == (min(32, 16),)


def test_analyze_weight_matrix_singular_values_shape_tall():
    torch.manual_seed(8)
    W = torch.randn(8, 20)
    result = analyze_weight_matrix(W)
    s = result["singular_values"]
    assert s.shape == (min(8, 20),)


# ---------------------------------------------------------------------------
# 3. effective_rank is a positive float <= min(m,n)
# ---------------------------------------------------------------------------


def test_analyze_weight_matrix_effective_rank():
    torch.manual_seed(9)
    W = torch.randn(32, 16)
    result = analyze_weight_matrix(W)
    er = result["effective_rank"]
    assert isinstance(er, float)
    assert er > 0.0
    assert er <= min(W.shape)


# ---------------------------------------------------------------------------
# 4. detect_superposition — returns dict with all required keys
# ---------------------------------------------------------------------------


def test_detect_superposition_keys():
    torch.manual_seed(10)
    W = torch.randn(64, 32)
    result = detect_superposition(W)
    required = {
        "superposition_score",
        "has_superposition",
        "n_active_dimensions",
        "compression_ratio",
    }
    assert required.issubset(result.keys())


# ---------------------------------------------------------------------------
# 5. detect_superposition on fat random matrix — just check it runs
# ---------------------------------------------------------------------------


def test_detect_superposition_runs():
    torch.manual_seed(11)
    W = torch.randn(64, 128)
    result = detect_superposition(W)
    assert isinstance(result["has_superposition"], bool)


# ---------------------------------------------------------------------------
# 6. detect_superposition compression_ratio > 0
# ---------------------------------------------------------------------------


def test_detect_superposition_compression_ratio_positive():
    torch.manual_seed(12)
    W = torch.randn(32, 64)
    result = detect_superposition(W)
    assert result["compression_ratio"] > 0.0


# ---------------------------------------------------------------------------
# 7. compute_attention_pattern_similarity returns (n_heads, T, T)
# ---------------------------------------------------------------------------


def test_attention_pattern_similarity_shape():
    model = _make_model()
    input_ids = _make_input()
    T = input_ids.shape[1]
    n_heads = TINY_CFG.n_heads
    result = compute_attention_pattern_similarity(model, input_ids, layer_idx=0)
    assert result.shape == (n_heads, T, T)


# ---------------------------------------------------------------------------
# 8. attention patterns sum to ~1 along last dim (softmax rows)
# ---------------------------------------------------------------------------


def test_attention_pattern_row_sum():
    model = _make_model()
    input_ids = _make_input()
    result = compute_attention_pattern_similarity(model, input_ids, layer_idx=0)
    row_sums = result.sum(dim=-1)  # (n_heads, T)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


# ---------------------------------------------------------------------------
# 9. logit_lens_analysis returns (n_layers,) tensor
# ---------------------------------------------------------------------------


def test_logit_lens_shape():
    torch.manual_seed(42)
    model = _make_model()
    input_ids = _make_input()
    result = logit_lens_analysis(model, input_ids, target_token=5)
    assert result.shape == (TINY_CFG.n_layers,)


# ---------------------------------------------------------------------------
# 10. logit_lens values are finite
# ---------------------------------------------------------------------------


def test_logit_lens_finite():
    torch.manual_seed(42)
    model = _make_model()
    input_ids = _make_input()
    result = logit_lens_analysis(model, input_ids, target_token=5)
    assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# 11. identify_attention_heads_by_type returns dict with correct keys
# ---------------------------------------------------------------------------


def test_identify_heads_by_type_keys():
    model = _make_model()
    input_ids = _make_input()
    result = identify_attention_heads_by_type(model, input_ids)
    assert "induction" in result
    assert "previous_token" in result
    assert "current_token" in result


def test_identify_heads_by_type_values_are_lists():
    model = _make_model()
    input_ids = _make_input()
    result = identify_attention_heads_by_type(model, input_ids)
    for key in ("induction", "previous_token", "current_token"):
        assert isinstance(result[key], list)


# ---------------------------------------------------------------------------
# 12. compute_residual_stream_contributions returns dict with layer keys
# ---------------------------------------------------------------------------


def test_residual_stream_contributions_keys():
    model = _make_model()
    input_ids = _make_input()
    result = compute_residual_stream_contributions(model, input_ids)
    assert "embedding" in result
    for i in range(TINY_CFG.n_layers):
        assert f"layer_{i}" in result


def test_residual_stream_contributions_shapes():
    model = _make_model()
    input_ids = _make_input()
    T = input_ids.shape[1]
    result = compute_residual_stream_contributions(model, input_ids)
    assert result["embedding"].shape == (T, TINY_CFG.d_model)
    assert result["layer_0"].shape == (T, TINY_CFG.d_model)
