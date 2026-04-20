"""Tests for sink_logit_bias."""

from __future__ import annotations

import pytest
import torch

from src.inference.sink_logit_bias import SinkLogitBiasApplier, apply_sink_token_logit_bias


def test_bias_changes_sink_ids_on_last_positions():
    torch.manual_seed(0)
    logits = torch.zeros(1, 4, 8)
    out = apply_sink_token_logit_bias(
        logits,
        [1, 3],
        last_n_positions=2,
        bonus=2.0,
    )
    assert torch.allclose(out[0, :2, :], logits[0, :2, :])
    assert out[0, 2, 1] == 2.0 and out[0, 2, 3] == 2.0
    assert out[0, 3, 1] == 2.0


def test_gradient_flows():
    logits = torch.randn(2, 5, 16, requires_grad=True)
    out = apply_sink_token_logit_bias(
        logits,
        [0],
        last_n_positions=1,
        bonus=0.5,
    )
    out.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_oob_token_raises():
    logits = torch.randn(1, 2, 4)
    with pytest.raises(ValueError):
        apply_sink_token_logit_bias(logits, [99], last_n_positions=1, bonus=1.0)


def test_last_n_exceeds_seq_raises():
    logits = torch.randn(1, 2, 4)
    with pytest.raises(ValueError):
        apply_sink_token_logit_bias(logits, [0], last_n_positions=5, bonus=1.0)


def test_empty_sink_ids_raises():
    logits = torch.randn(1, 2, 4)
    with pytest.raises(ValueError):
        apply_sink_token_logit_bias(logits, [], last_n_positions=1, bonus=1.0)


def test_zero_last_n_returns_clone_equal():
    logits = torch.randn(1, 3, 5)
    out = apply_sink_token_logit_bias(logits, [1], last_n_positions=0, bonus=9.0)
    assert torch.equal(out, logits)


def test_bad_logits_rank():
    with pytest.raises(ValueError):
        apply_sink_token_logit_bias(torch.randn(4, 8), [0], last_n_positions=1, bonus=1.0)


def test_applier_callable():
    ap = SinkLogitBiasApplier([2], last_n_positions=1, bonus=3.0)
    logits = torch.zeros(1, 1, 5)
    out = ap(logits)
    assert out[0, 0, 2] == 3.0


def test_applier_empty_ids_raises():
    with pytest.raises(ValueError):
        SinkLogitBiasApplier(())


def test_determinism():
    torch.manual_seed(1)
    a = torch.randn(1, 6, 10)
    torch.manual_seed(1)
    b = torch.randn(1, 6, 10)
    o1 = apply_sink_token_logit_bias(b, [2, 4], last_n_positions=3, bonus=0.25)
    torch.manual_seed(1)
    b2 = torch.randn(1, 6, 10)
    o2 = apply_sink_token_logit_bias(b2, [2, 4], last_n_positions=3, bonus=0.25)
    assert torch.allclose(o1, o2)
