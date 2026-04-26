"""Tests for process_supervision.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alignment.process_supervision import (
    ProcessSupervision,
    process_supervision_loss,
    reference_process_supervision_loss,
)
from src.model.config import AureliusConfig

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


class MockModel(nn.Module):
    """Produces padded step representations ``z`` for the verifier."""

    def __init__(self, cfg: AureliusConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        z = self.embed(input_ids)
        return self.proj(z)


def _make_inputs(batch: int = 2, steps: int = 4):
    input_ids = (
        torch.arange(batch * steps, dtype=torch.long).reshape(batch, steps) % TINY_CFG.vocab_size
    )
    y = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=torch.float32)[:batch, :steps]
    m = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)[:batch, :steps]
    return input_ids, y, m


def test_forward_shape_and_dtype_tiny_config():
    torch.manual_seed(0)
    backbone = MockModel(TINY_CFG)
    verifier = ProcessSupervision(TINY_CFG.d_model)
    input_ids, y, m = _make_inputs()

    z = backbone(input_ids)
    out = verifier(z, y=y, m=m)

    assert out.v.shape == (2, 4)
    assert out.p.shape == (2, 4)
    assert out.v.dtype == z.dtype
    assert out.p.dtype == z.dtype
    assert out.loss is not None
    assert out.loss.ndim == 0


def test_loss_matches_manual_masked_bce():
    v = torch.tensor([[0.2, -1.0, 0.7], [1.5, -0.3, 0.1]], dtype=torch.float32)
    y = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    m = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.bool)

    loss = process_supervision_loss(v, y, m)
    expected = F.binary_cross_entropy_with_logits(v[m], y[m], reduction="mean")
    assert torch.allclose(loss, expected, atol=1e-6)


def test_reference_formulation_equivalence_atol_1e_5():
    torch.manual_seed(1)
    v = torch.randn(3, 5)
    y = torch.randint(0, 2, (3, 5), dtype=torch.long).float()
    m = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 0, 1, 1, 1]],
        dtype=torch.bool,
    )

    loss = process_supervision_loss(v, y, m)
    ref = reference_process_supervision_loss(v, y, m)
    assert torch.allclose(loss, ref, atol=1e-5)


def test_backward_produces_finite_grads_on_all_trainable_params():
    torch.manual_seed(2)
    backbone = MockModel(TINY_CFG)
    verifier = ProcessSupervision(TINY_CFG.d_model)
    input_ids, y, m = _make_inputs()

    z = backbone(input_ids)
    loss = verifier(z, y=y, m=m).loss
    assert loss is not None
    loss.backward()

    params = list(backbone.parameters()) + list(verifier.parameters())
    assert params
    for param in params:
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_forward_is_deterministic_under_manual_seed():
    torch.manual_seed(7)
    model_a = ProcessSupervision(TINY_CFG.d_model)
    z = torch.randn(2, 4, TINY_CFG.d_model)
    out_a = model_a(z)

    torch.manual_seed(7)
    model_b = ProcessSupervision(TINY_CFG.d_model)
    z_b = torch.randn(2, 4, TINY_CFG.d_model)
    out_b = model_b(z_b)

    assert torch.equal(z, z_b)
    assert torch.allclose(out_a.v, out_b.v)
    assert torch.allclose(out_a.p, out_b.p)


def test_batch_one_seq_len_one_edge_case():
    torch.manual_seed(3)
    verifier = ProcessSupervision(TINY_CFG.d_model)
    z = torch.randn(1, 1, TINY_CFG.d_model)
    y = torch.tensor([[1.0]])
    m = torch.tensor([[True]])

    out = verifier(z, y=y, m=m)
    assert out.v.shape == (1, 1)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_masked_positions_do_not_affect_loss():
    torch.manual_seed(4)
    verifier = ProcessSupervision(TINY_CFG.d_model)
    z = torch.randn(1, 3, TINY_CFG.d_model)
    y = torch.tensor([[1.0, 0.0, 1.0]])
    m = torch.tensor([[True, False, True]])

    out = verifier(z, y=y, m=m)

    v = verifier(z).v
    expected = F.binary_cross_entropy_with_logits(v[:, [0, 2]], y[:, [0, 2]], reduction="mean")
    assert out.loss is not None
    assert torch.allclose(out.loss, expected, atol=1e-6)


def test_padded_batch_matches_truncated_unpadded_batch():
    torch.manual_seed(5)
    verifier = ProcessSupervision(TINY_CFG.d_model)
    z = torch.randn(2, 4, TINY_CFG.d_model)
    y = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    m = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

    padded_loss = verifier(z, y=y, m=m).loss
    z_changed = z.clone()
    z_changed[1, 2:] = 1000.0 * torch.randn_like(z_changed[1, 2:])
    y_changed = y.clone()
    y_changed[1, 2:] = 1.0 - y_changed[1, 2:]
    trim_loss = verifier(z_changed, y=y_changed, m=m).loss

    assert padded_loss is not None and trim_loss is not None
    assert torch.allclose(padded_loss, trim_loss, atol=1e-6)


def test_all_masked_returns_zero_connected_loss():
    torch.manual_seed(6)
    verifier = ProcessSupervision(TINY_CFG.d_model)
    z = torch.randn(2, 3, TINY_CFG.d_model, requires_grad=True)
    y = torch.zeros(2, 3)
    m = torch.zeros(2, 3, dtype=torch.bool)

    out = verifier(z, y=y, m=m)
    assert out.loss is not None
    assert out.loss.item() == 0.0
    out.loss.backward()
    assert z.grad is not None
    assert torch.all(z.grad == 0)


def test_extreme_inputs_remain_finite():
    verifier = ProcessSupervision(TINY_CFG.d_model)
    z = torch.full((2, 3, TINY_CFG.d_model), 1.0e6)
    y = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    m = torch.ones(2, 3, dtype=torch.bool)

    out = verifier(z, y=y, m=m)
    assert torch.isfinite(out.v).all()
    assert torch.isfinite(out.p).all()
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_probabilities_stay_in_unit_interval():
    torch.manual_seed(8)
    verifier = ProcessSupervision(TINY_CFG.d_model)
    z = torch.randn(3, 2, TINY_CFG.d_model)

    p = verifier(z).p
    assert torch.all(p >= 0.0)
    assert torch.all(p <= 1.0)


def test_default_mask_matches_explicit_all_true_mask():
    torch.manual_seed(9)
    verifier = ProcessSupervision(TINY_CFG.d_model)
    z = torch.randn(2, 3, TINY_CFG.d_model)
    y = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    m = torch.ones(2, 3, dtype=torch.bool)

    implicit = verifier(z, y=y).loss
    explicit = verifier(z, y=y, m=m).loss
    assert implicit is not None and explicit is not None
    assert torch.allclose(implicit, explicit, atol=1e-6)


def test_invalid_shapes_raise_value_error():
    verifier = ProcessSupervision(TINY_CFG.d_model)
    z = torch.randn(2, 3, TINY_CFG.d_model)
    y = torch.ones(2, 4)
    m = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError):
        verifier(z, y=y, m=m)
