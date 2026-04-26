"""Tests for NAS primitives: mixed operations, weight sharing, and DARTS-style search."""

from __future__ import annotations

import torch

from src.model.nas_primitives import (
    ConvOp,
    DARTSCell,
    IdentityOp,
    LinearOp,
    MixedOperation,
    NASArchitectureOptimizer,
    NASConfig,
    ZeroOp,
    discretize_architecture,
)

torch.manual_seed(0)

B, T, D = 2, 8, 64


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _sample_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, D)


def _default_ops(d_model: int = D) -> list:
    return [IdentityOp(), ZeroOp(), LinearOp(d_model), ConvOp(d_model)]


# ---------------------------------------------------------------------------
# 1. NASConfig defaults
# ---------------------------------------------------------------------------


def test_nas_config_defaults():
    cfg = NASConfig()
    assert cfg.n_ops == 4
    assert cfg.d_model == 64
    assert cfg.temperature == 1.0
    assert cfg.straight_through is True


# ---------------------------------------------------------------------------
# 2. IdentityOp shape
# ---------------------------------------------------------------------------


def test_identity_op_shape():
    op = IdentityOp()
    x = _sample_input()
    out = op(x)
    assert out.shape == x.shape
    assert torch.equal(out, x), "IdentityOp must return the input unchanged"


# ---------------------------------------------------------------------------
# 3. ZeroOp zeros
# ---------------------------------------------------------------------------


def test_zero_op_zeros():
    op = ZeroOp()
    x = _sample_input()
    out = op(x)
    assert out.shape == x.shape
    assert out.sum().item() == 0.0, "ZeroOp must return all zeros"


# ---------------------------------------------------------------------------
# 4. LinearOp shape
# ---------------------------------------------------------------------------


def test_linear_op_shape():
    op = LinearOp(D)
    x = _sample_input()
    out = op(x)
    assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# 5. ConvOp shape
# ---------------------------------------------------------------------------


def test_conv_op_shape():
    op = ConvOp(D)
    x = _sample_input()
    out = op(x)
    assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# 6. MixedOperation output shape
# ---------------------------------------------------------------------------


def test_mixed_operation_output_shape():
    cfg = NASConfig(d_model=D)
    mixed = MixedOperation(_default_ops(), cfg)
    x = _sample_input()
    out = mixed(x)
    assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# 7. MixedOperation softmax weights sum to 1
# ---------------------------------------------------------------------------


def test_mixed_operation_weights_sum_one():
    cfg = NASConfig(d_model=D)
    mixed = MixedOperation(_default_ops(), cfg)
    import torch.nn.functional as F

    weights = F.softmax(mixed.arch_weights / cfg.temperature, dim=0)
    assert abs(weights.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 8. MixedOperation gumbel_forward shape
# ---------------------------------------------------------------------------


def test_mixed_operation_gumbel_shape():
    cfg = NASConfig(d_model=D)
    mixed = MixedOperation(_default_ops(), cfg)
    x = _sample_input()
    out = mixed.gumbel_forward(x)
    assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# 9. DARTSCell output shape
# ---------------------------------------------------------------------------


def test_darts_cell_output_shape():
    cfg = NASConfig(d_model=D)
    cell = DARTSCell(n_nodes=3, d_model=D, config=cfg)
    x = _sample_input()
    out = cell(x)
    assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# 10. DARTSCell get_architecture_weights returns dict
# ---------------------------------------------------------------------------


def test_darts_cell_arch_weights_dict():
    cfg = NASConfig(d_model=D)
    cell = DARTSCell(n_nodes=3, d_model=D, config=cfg)
    weights = cell.get_architecture_weights()
    assert isinstance(weights, dict)
    assert len(weights) > 0
    for key, val in weights.items():
        assert isinstance(key, str)
        assert isinstance(val, torch.Tensor)


# ---------------------------------------------------------------------------
# 11. discretize_architecture keys
# ---------------------------------------------------------------------------


def test_discretize_architecture_keys():
    cfg = NASConfig(d_model=D)
    cell = DARTSCell(n_nodes=3, d_model=D, config=cfg)
    arch = discretize_architecture(cell, top_k=1)
    assert isinstance(arch, dict)
    assert len(arch) > 0
    for key, val in arch.items():
        assert isinstance(key, str)
        assert isinstance(val, str)
    # All edge names should appear in architecture weights dict
    arch_weight_keys = set(cell.get_architecture_weights().keys())
    for key in arch.keys():
        assert key in arch_weight_keys


# ---------------------------------------------------------------------------
# 12. NASArchitectureOptimizer param groups non-empty
# ---------------------------------------------------------------------------


def test_nas_optimizer_param_groups():
    cfg = NASConfig(d_model=D)
    cell = DARTSCell(n_nodes=3, d_model=D, config=cfg)
    optimizer = NASArchitectureOptimizer(cell, arch_lr=3e-4, weight_lr=3e-4)
    assert len(optimizer.arch_params()) > 0, "arch_params must be non-empty"
    assert len(optimizer.weight_params()) > 0, "weight_params must be non-empty"
    # Ensure no overlap between arch and weight params
    arch_ids = {id(p) for p in optimizer.arch_params()}
    weight_ids = {id(p) for p in optimizer.weight_params()}
    assert arch_ids.isdisjoint(weight_ids), "arch and weight params must not overlap"
