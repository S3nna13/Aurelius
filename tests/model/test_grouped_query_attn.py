"""Tests for src/model/grouped_query_attn.py.

Uses tiny dimensions so that tests run quickly on CPU.
"""

from __future__ import annotations

import pytest
import torch

from src.model.grouped_query_attn import (
    GQABlock,
    GQAConfig,
    GroupedQueryAttention,
    MultiQueryAttention,
    count_kv_cache_params,
    repeat_kv,
)

# ---------------------------------------------------------------------------
# Shared tiny constants
# ---------------------------------------------------------------------------
D_MODEL = 16
N_HEADS = 4
N_KV_HEADS = 2
D_HEAD = 4
BATCH = 2
SEQ = 6


@pytest.fixture
def cfg() -> GQAConfig:
    return GQAConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        d_head=D_HEAD,
        causal=True,
        dropout_p=0.0,
    )


@pytest.fixture
def x(cfg: GQAConfig) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ, cfg.d_model)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    """d_head should default to d_model // n_heads when not specified."""
    cfg = GQAConfig(d_model=64, n_heads=8, n_kv_heads=2)
    assert cfg.d_head == 64 // 8


# ---------------------------------------------------------------------------
# 2. Invalid n_kv_heads raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_n_kv_heads_raises() -> None:
    """n_heads not divisible by n_kv_heads must raise ValueError."""
    with pytest.raises(ValueError, match="divisible"):
        GQAConfig(d_model=16, n_heads=4, n_kv_heads=3)


# ---------------------------------------------------------------------------
# 3. repeat_kv output shape
# ---------------------------------------------------------------------------


def test_repeat_kv_output_shape() -> None:
    """repeat_kv should expand (B, T, n_kv_heads, d_head) by n_rep."""
    n_rep = N_HEADS // N_KV_HEADS  # 2
    kv = torch.randn(BATCH, SEQ, N_KV_HEADS, D_HEAD)
    out = repeat_kv(kv, n_rep)
    assert out.shape == (BATCH, SEQ, N_KV_HEADS * n_rep, D_HEAD)


# ---------------------------------------------------------------------------
# 4. repeat_kv expand correctness — first and expanded heads are equal
# ---------------------------------------------------------------------------


def test_repeat_kv_correctness() -> None:
    """Each expanded head must equal its source KV head."""
    n_rep = 3
    n_kv = 2
    kv = torch.randn(BATCH, SEQ, n_kv, D_HEAD)
    out = repeat_kv(kv, n_rep)
    # out shape: (B, T, n_kv * n_rep, d_head)
    for kv_idx in range(n_kv):
        for rep in range(n_rep):
            expanded_head = out[:, :, kv_idx * n_rep + rep, :]
            original_head = kv[:, :, kv_idx, :]
            assert torch.allclose(expanded_head, original_head), (
                f"Expanded head {kv_idx * n_rep + rep} does not match KV head {kv_idx}"
            )


# ---------------------------------------------------------------------------
# 5. GQA output shape
# ---------------------------------------------------------------------------


def test_gqa_output_shape(cfg: GQAConfig, x: torch.Tensor) -> None:
    """GroupedQueryAttention must return (B, T, d_model)."""
    model = GroupedQueryAttention(cfg)
    out = model(x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# 6. MQA output shape
# ---------------------------------------------------------------------------


def test_mqa_output_shape(x: torch.Tensor) -> None:
    """MultiQueryAttention must return (B, T, d_model)."""
    model = MultiQueryAttention(d_model=D_MODEL, n_heads=N_HEADS, d_head=D_HEAD)
    out = model(x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# 7. Gradient flows through GQA
# ---------------------------------------------------------------------------


def test_gqa_gradient_flow(cfg: GQAConfig, x: torch.Tensor) -> None:
    """Loss.backward() must produce non-None gradients for all parameters."""
    x = x.requires_grad_(True)
    model = GroupedQueryAttention(cfg)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# 8. GQABlock output shape
# ---------------------------------------------------------------------------


def test_gqa_block_output_shape(cfg: GQAConfig, x: torch.Tensor) -> None:
    """GQABlock must return (B, T, d_model)."""
    block = GQABlock(cfg)
    out = block(x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# 9. GQABlock residual — output != input (residual changes values)
# ---------------------------------------------------------------------------


def test_gqa_block_residual(cfg: GQAConfig, x: torch.Tensor) -> None:
    """GQABlock output should not be identical to input (residual is non-trivial)."""
    block = GQABlock(cfg)
    out = block(x)
    assert not torch.allclose(out, x), "GQABlock output equals input — residual has no effect"


# ---------------------------------------------------------------------------
# 10. count_kv_cache_params keys present
# ---------------------------------------------------------------------------


def test_count_kv_cache_params_keys(cfg: GQAConfig) -> None:
    """count_kv_cache_params must return all required keys."""
    stats = count_kv_cache_params(cfg)
    assert "n_kv_params" in stats
    assert "n_q_params" in stats
    assert "kv_reduction_ratio_vs_mha" in stats


# ---------------------------------------------------------------------------
# 11. kv_reduction_ratio correct (n_heads // n_kv_heads)
# ---------------------------------------------------------------------------


def test_kv_reduction_ratio(cfg: GQAConfig) -> None:
    """kv_reduction_ratio_vs_mha must equal n_heads // n_kv_heads."""
    stats = count_kv_cache_params(cfg)
    expected = N_HEADS // N_KV_HEADS
    assert stats["kv_reduction_ratio_vs_mha"] == expected


# ---------------------------------------------------------------------------
# 12. MQA has exactly 1 KV head
# ---------------------------------------------------------------------------


def test_mqa_has_one_kv_head() -> None:
    """MultiQueryAttention's inner config must have n_kv_heads == 1."""
    model = MultiQueryAttention(d_model=D_MODEL, n_heads=N_HEADS, d_head=D_HEAD)
    assert model.attn.config.n_kv_heads == 1


# ---------------------------------------------------------------------------
# 13. GQA with n_kv_heads == n_heads has same KV param count as MHA
# ---------------------------------------------------------------------------


def test_gqa_equals_mha_when_kv_heads_match() -> None:
    """When n_kv_heads == n_heads the KV param count equals full MHA."""
    cfg_mha = GQAConfig(d_model=D_MODEL, n_heads=N_HEADS, n_kv_heads=N_HEADS, d_head=D_HEAD)
    stats = count_kv_cache_params(cfg_mha)
    # In full MHA every head is its own KV head, ratio should be 1
    assert stats["kv_reduction_ratio_vs_mha"] == 1
    # n_kv_params == n_q_params (same head count)
    assert stats["n_kv_params"] == N_HEADS * D_HEAD * 2
    assert stats["n_q_params"] == N_HEADS * D_HEAD


# ---------------------------------------------------------------------------
# 14. Single-token sequence (T=1) works without error
# ---------------------------------------------------------------------------


def test_single_token_sequence(cfg: GQAConfig) -> None:
    """GQA must handle a single-token sequence (T=1) correctly."""
    torch.manual_seed(42)
    x_single = torch.randn(BATCH, 1, D_MODEL)
    model = GroupedQueryAttention(cfg)
    out = model(x_single)
    assert out.shape == (BATCH, 1, D_MODEL)
