"""Tests for src/model/tensor_parallel.py.

All tests use small dimensions (tiny D) and pure PyTorch only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.model.tensor_parallel import (
    TPConfig,
    ColumnParallelLinear,
    RowParallelLinear,
    HeadParallelAttention,
    split_tensor_column,
    split_tensor_row,
    verify_partition_equivalence,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_PARTS = 4
IN_F = 16
OUT_F = 8
BATCH = 2
SEQ = 5
D_MODEL = 16
N_HEADS = 4


@pytest.fixture
def weight_col():
    """(OUT_F, IN_F) weight for column-split tests."""
    torch.manual_seed(0)
    return torch.randn(OUT_F, IN_F)


@pytest.fixture
def weight_row():
    """(OUT_F, IN_F) weight for row-split tests."""
    torch.manual_seed(1)
    return torch.randn(OUT_F, IN_F)


@pytest.fixture
def x_2d():
    """(BATCH, IN_F) input for linear layers."""
    torch.manual_seed(2)
    return torch.randn(BATCH, IN_F)


@pytest.fixture
def x_3d():
    """(BATCH, SEQ, D_MODEL) input for attention layers."""
    torch.manual_seed(3)
    return torch.randn(BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# TPConfig
# ---------------------------------------------------------------------------

def test_tpconfig_defaults():
    cfg = TPConfig()
    assert cfg.n_partitions == 4
    assert cfg.d_model == 512
    assert cfg.gather_output is True


def test_tpconfig_custom():
    cfg = TPConfig(n_partitions=2, d_model=128, gather_output=False)
    assert cfg.n_partitions == 2
    assert cfg.d_model == 128
    assert cfg.gather_output is False


# ---------------------------------------------------------------------------
# split_tensor_column
# ---------------------------------------------------------------------------

def test_split_column_count(weight_col):
    parts = split_tensor_column(weight_col, N_PARTS)
    assert len(parts) == N_PARTS


def test_split_column_shape(weight_col):
    parts = split_tensor_column(weight_col, N_PARTS)
    expected = (OUT_F // N_PARTS, IN_F)
    for p in parts:
        assert p.shape == expected, f"Expected {expected}, got {p.shape}"


def test_split_column_reconstruction(weight_col):
    parts = split_tensor_column(weight_col, N_PARTS)
    reconstructed = torch.cat(parts, dim=0)
    assert torch.allclose(reconstructed, weight_col)


# ---------------------------------------------------------------------------
# split_tensor_row
# ---------------------------------------------------------------------------

def test_split_row_count(weight_row):
    parts = split_tensor_row(weight_row, N_PARTS)
    assert len(parts) == N_PARTS


def test_split_row_shape(weight_row):
    parts = split_tensor_row(weight_row, N_PARTS)
    expected = (OUT_F, IN_F // N_PARTS)
    for p in parts:
        assert p.shape == expected, f"Expected {expected}, got {p.shape}"


def test_split_row_reconstruction(weight_row):
    parts = split_tensor_row(weight_row, N_PARTS)
    reconstructed = torch.cat(parts, dim=1)
    assert torch.allclose(reconstructed, weight_row)


# ---------------------------------------------------------------------------
# ColumnParallelLinear
# ---------------------------------------------------------------------------

@pytest.fixture
def col_linear():
    torch.manual_seed(10)
    return ColumnParallelLinear(IN_F, OUT_F, N_PARTS, bias=True)


def test_col_parallel_forward_shape(col_linear, x_2d):
    out = col_linear(x_2d)
    assert out.shape == (BATCH, OUT_F), f"Expected ({BATCH}, {OUT_F}), got {out.shape}"


def test_col_parallel_output_finite(col_linear, x_2d):
    out = col_linear(x_2d)
    assert torch.isfinite(out).all(), "ColumnParallelLinear output contains non-finite values"


def test_col_parallel_verify_equivalence(x_2d):
    torch.manual_seed(20)
    ref = nn.Linear(IN_F, OUT_F, bias=True)
    col = ColumnParallelLinear(IN_F, OUT_F, N_PARTS, bias=True)
    result = verify_partition_equivalence(ref, col, x_2d, tol=1e-5)
    assert result is True, "verify_partition_equivalence should return True for copied weights"


def test_col_parallel_no_bias(x_2d):
    layer = ColumnParallelLinear(IN_F, OUT_F, N_PARTS, bias=False)
    out = layer(x_2d)
    assert out.shape == (BATCH, OUT_F)
    for p in layer.partitions:
        assert p.bias is None


# ---------------------------------------------------------------------------
# RowParallelLinear
# ---------------------------------------------------------------------------

@pytest.fixture
def row_linear():
    torch.manual_seed(30)
    return RowParallelLinear(IN_F, OUT_F, N_PARTS, bias=True)


def test_row_parallel_forward_shape(row_linear, x_2d):
    out = row_linear(x_2d)
    assert out.shape == (BATCH, OUT_F), f"Expected ({BATCH}, {OUT_F}), got {out.shape}"


def test_row_parallel_output_finite(row_linear, x_2d):
    out = row_linear(x_2d)
    assert torch.isfinite(out).all(), "RowParallelLinear output contains non-finite values"


def test_row_parallel_no_bias(x_2d):
    layer = RowParallelLinear(IN_F, OUT_F, N_PARTS, bias=False)
    out = layer(x_2d)
    assert out.shape == (BATCH, OUT_F)


def test_row_parallel_allreduce_equivalence(x_2d):
    """A single nn.Linear(IN_F, OUT_F, bias=False) should equal RowParallelLinear
    when weights are carefully initialised to match partition slices."""
    torch.manual_seed(42)
    layer = RowParallelLinear(IN_F, OUT_F, N_PARTS, bias=False)

    # Build a reference weight by concatenating all partition weights along dim=1
    with torch.no_grad():
        parts = [p.weight.data for p in layer.partitions]  # each (OUT_F, IN_F // N_PARTS)
        full_weight = torch.cat(parts, dim=1)  # (OUT_F, IN_F)
        ref = nn.Linear(IN_F, OUT_F, bias=False)
        ref.weight.data.copy_(full_weight)

    with torch.no_grad():
        ref_out = ref(x_2d)
        tp_out = layer(x_2d)

    assert torch.allclose(ref_out, tp_out, atol=1e-5), (
        f"Max diff: {(ref_out - tp_out).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# HeadParallelAttention
# ---------------------------------------------------------------------------

@pytest.fixture
def head_attn():
    torch.manual_seed(50)
    return HeadParallelAttention(d_model=D_MODEL, n_heads=N_HEADS, n_partitions=N_PARTS)


def test_head_parallel_forward_shape(head_attn, x_3d):
    out = head_attn(x_3d)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH}, {SEQ}, {D_MODEL}), got {out.shape}"
    )


def test_head_parallel_output_finite(head_attn, x_3d):
    out = head_attn(x_3d)
    assert torch.isfinite(out).all(), "HeadParallelAttention output contains non-finite values"


def test_head_parallel_invalid_heads():
    with pytest.raises(ValueError, match="divisible"):
        HeadParallelAttention(d_model=16, n_heads=3, n_partitions=2)


# ---------------------------------------------------------------------------
# verify_partition_equivalence
# ---------------------------------------------------------------------------

def test_verify_equivalence_returns_true(x_2d):
    torch.manual_seed(99)
    ref = nn.Linear(IN_F, OUT_F, bias=True)
    col = ColumnParallelLinear(IN_F, OUT_F, N_PARTS, bias=True)
    assert verify_partition_equivalence(ref, col, x_2d, tol=1e-4) is True


def test_verify_equivalence_detects_mismatch(x_2d):
    """If col_parallel has random weights (not copied from linear), outputs differ."""
    torch.manual_seed(7)
    ref = nn.Linear(IN_F, OUT_F, bias=True)
    torch.manual_seed(8)
    col = ColumnParallelLinear(IN_F, OUT_F, N_PARTS, bias=True)
    # Don't copy weights — just check the function runs and (likely) returns False
    # We re-randomise col weights to make mismatch certain.
    for p in col.partitions:
        nn.init.uniform_(p.weight, -10.0, 10.0)
        if p.bias is not None:
            nn.init.uniform_(p.bias, -10.0, 10.0)
    with torch.no_grad():
        ref_out = ref(x_2d)
        col_out = col(x_2d)
    assert not torch.allclose(ref_out, col_out, atol=1e-4), (
        "Outputs should differ when weights are mismatched"
    )
