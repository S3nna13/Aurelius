"""Tests for loss landscape sharpness metrics (SAM + Hutchinson Hessian trace)."""
import torch
import pytest
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.sharpness import (
    SharpnessConfig,
    SharpnessResult,
    sam_perturbation,
    restore_perturbation,
    sharpness_aware_loss,
    hutchinson_hessian_trace,
    measure_sharpness,
)


def _make_model():
    torch.manual_seed(42)
    cfg = AureliusConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


def _make_batch(batch_size=2, seq_len=16, vocab_size=256):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, labels


def test_sam_perturbation_changes_params():
    """After sam_perturbation, at least one parameter should differ from original."""
    model = _make_model()
    input_ids, labels = _make_batch()

    # capture original param values
    original_params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    loss, _, _ = model(input_ids, labels=labels)
    sam_perturbation(model, loss)

    changed = False
    for n, p in model.named_parameters():
        if p.requires_grad and n in original_params:
            if not torch.allclose(p.data, original_params[n]):
                changed = True
                break
    assert changed, "Parameters should change after SAM perturbation"


def test_sam_perturbation_returns_grad_norm():
    """sam_perturbation result dict must contain 'grad_norm' > 0."""
    model = _make_model()
    input_ids, labels = _make_batch()

    loss, _, _ = model(input_ids, labels=labels)
    result = sam_perturbation(model, loss)

    assert "grad_norm" in result
    assert result["grad_norm"] > 0


def test_restore_perturbation_recovers_weights():
    """After restore_perturbation, params must match original values."""
    model = _make_model()
    input_ids, labels = _make_batch()

    original_params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    loss, _, _ = model(input_ids, labels=labels)
    result = sam_perturbation(model, loss)
    restore_perturbation(model, result["perturbation"])

    for n, p in model.named_parameters():
        if p.requires_grad and n in original_params:
            assert torch.allclose(p.data, original_params[n], atol=1e-6), (
                f"Parameter {n} not restored after perturbation"
            )


def test_sharpness_aware_loss_returns_two_losses():
    """sharpness_aware_loss must return a tuple of two Tensors."""
    model = _make_model()
    input_ids, labels = _make_batch()

    result = sharpness_aware_loss(model, input_ids, labels)

    assert isinstance(result, tuple)
    assert len(result) == 2
    orig_loss, perturbed_loss = result
    assert isinstance(orig_loss, torch.Tensor)
    assert isinstance(perturbed_loss, torch.Tensor)
    assert orig_loss.ndim == 0
    assert perturbed_loss.ndim == 0


def test_perturbed_loss_differs_from_original():
    """Perturbed loss should differ from original loss for a non-trivial model."""
    torch.manual_seed(0)
    model = _make_model()
    input_ids, labels = _make_batch()

    orig_loss, perturbed_loss = sharpness_aware_loss(model, input_ids, labels)

    assert not torch.isclose(orig_loss, perturbed_loss, atol=1e-6), (
        "Perturbed loss should differ from original loss"
    )


def test_hutchinson_hessian_trace_positive():
    """Hutchinson trace estimator should return a positive float for typical loss."""
    torch.manual_seed(7)
    model = _make_model()
    input_ids, labels = _make_batch()

    def loss_fn():
        loss, _, _ = model(input_ids, labels=labels)
        return loss

    trace = hutchinson_hessian_trace(model, loss_fn, n_samples=5)

    assert isinstance(trace, float)
    assert trace > 0, f"Expected positive Hessian trace, got {trace}"


def test_hutchinson_hessian_trace_n_samples():
    """hutchinson_hessian_trace runs without error for n_samples=3."""
    model = _make_model()
    input_ids, labels = _make_batch()

    def loss_fn():
        loss, _, _ = model(input_ids, labels=labels)
        return loss

    trace = hutchinson_hessian_trace(model, loss_fn, n_samples=3)
    assert isinstance(trace, float)
    assert torch.isfinite(torch.tensor(trace))


def test_sharpness_result_fields():
    """SharpnessResult must have all required fields."""
    result = SharpnessResult(
        original_loss=1.0,
        perturbed_loss=1.1,
        sharpness=0.1,
        hessian_trace=None,
        flatness_score=0.9,
    )
    assert hasattr(result, "original_loss")
    assert hasattr(result, "perturbed_loss")
    assert hasattr(result, "sharpness")
    assert hasattr(result, "hessian_trace")
    assert hasattr(result, "flatness_score")


def test_flatness_score_range():
    """flatness_score must be in (0, 1]."""
    model = _make_model()
    input_ids, labels = _make_batch()
    cfg = SharpnessConfig()

    result = measure_sharpness(model, input_ids, labels, cfg, compute_hessian=False)

    assert 0 < result.flatness_score <= 1.0, (
        f"flatness_score {result.flatness_score} not in (0, 1]"
    )


def test_measure_sharpness_no_hessian():
    """When compute_hessian=False, hessian_trace must be None."""
    model = _make_model()
    input_ids, labels = _make_batch()
    cfg = SharpnessConfig()

    result = measure_sharpness(model, input_ids, labels, cfg, compute_hessian=False)

    assert result.hessian_trace is None


def test_measure_sharpness_with_hessian():
    """When compute_hessian=True, hessian_trace must be a float."""
    model = _make_model()
    input_ids, labels = _make_batch()
    cfg = SharpnessConfig(n_hutchinson_samples=3)

    result = measure_sharpness(model, input_ids, labels, cfg, compute_hessian=True)

    assert isinstance(result.hessian_trace, float)
    assert torch.isfinite(torch.tensor(result.hessian_trace))
