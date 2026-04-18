"""Tests for per_sample_grad_clip.py."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.security.per_sample_grad_clip import GradSampleHook, PerSampleClipper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


@pytest.fixture()
def batch() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(4, 8)


@pytest.fixture()
def clipper(model: nn.Sequential) -> PerSampleClipper:
    return PerSampleClipper(model, max_grad_norm=1.0)


def _forward_and_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out.sum()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_per_sample_clipper_instantiates(model: nn.Sequential) -> None:
    """PerSampleClipper can be instantiated with a model and max_grad_norm."""
    clipper = PerSampleClipper(model, max_grad_norm=1.0)
    assert clipper is not None


def test_grad_sample_hooks_registered_on_linear_layers(model: nn.Sequential) -> None:
    """GradSampleHooks are attached to every nn.Linear layer in the model."""
    clipper = PerSampleClipper(model, max_grad_norm=1.0)
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    assert len(clipper._hooks) == linear_count
    assert len(clipper._linear_layers) == linear_count


def test_step_runs_without_error(clipper: PerSampleClipper, batch: torch.Tensor, model: nn.Sequential) -> None:
    """step() completes without raising an exception."""
    loss = _forward_and_loss(model, batch)
    clipper.step(loss)


def test_step_returns_dict(clipper: PerSampleClipper, batch: torch.Tensor, model: nn.Sequential) -> None:
    """step() returns a dict."""
    loss = _forward_and_loss(model, batch)
    result = clipper.step(loss)
    assert isinstance(result, dict)


def test_step_dict_has_entry_per_linear_layer(clipper: PerSampleClipper, batch: torch.Tensor, model: nn.Sequential) -> None:
    """step() dict contains one entry for each nn.Linear layer."""
    loss = _forward_and_loss(model, batch)
    result = clipper.step(loss)
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    assert len(result) == linear_count


def test_step_sets_grad_on_first_linear(model: nn.Sequential, batch: torch.Tensor) -> None:
    """After step(), the first linear layer's weight.grad is set (not None)."""
    clipper = PerSampleClipper(model, max_grad_norm=1.0)
    loss = _forward_and_loss(model, batch)
    clipper.step(loss)
    first_linear = model[0]
    assert first_linear.weight.grad is not None


def test_grad_shape_matches_weight_shape(model: nn.Sequential, batch: torch.Tensor) -> None:
    """Grad tensor shape matches the corresponding weight tensor shape."""
    clipper = PerSampleClipper(model, max_grad_norm=1.0)
    loss = _forward_and_loss(model, batch)
    clipper.step(loss)
    for _, layer in clipper._linear_layers:
        assert layer.weight.grad.shape == layer.weight.shape


def test_grad_is_finite(model: nn.Sequential, batch: torch.Tensor) -> None:
    """All gradient values are finite (no NaN or Inf) after step()."""
    clipper = PerSampleClipper(model, max_grad_norm=1.0)
    loss = _forward_and_loss(model, batch)
    clipper.step(loss)
    for _, layer in clipper._linear_layers:
        assert torch.isfinite(layer.weight.grad).all()


def test_tiny_clip_norm_produces_small_gradients(model: nn.Sequential, batch: torch.Tensor) -> None:
    """With a tiny max_grad_norm, the resulting summed gradients are small."""
    max_norm = 0.01
    clipper = PerSampleClipper(model, max_grad_norm=max_norm)
    loss = _forward_and_loss(model, batch)
    clipper.step(loss)
    B = batch.shape[0]
    for _, layer in clipper._linear_layers:
        # Each sample contributes at most max_norm; summed over B samples
        grad_norm = layer.weight.grad.norm().item()
        assert grad_norm <= max_norm * B + 1e-5


def test_large_clip_norm_approximately_unclipped(model: nn.Sequential, batch: torch.Tensor) -> None:
    """With a huge max_grad_norm, clipping has no effect and grads are normal."""
    torch.manual_seed(0)
    model_ref = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    torch.manual_seed(0)
    model_clip = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    # Copy weights so both models are identical
    for p_ref, p_clip in zip(model_ref.parameters(), model_clip.parameters()):
        p_clip.data.copy_(p_ref.data)

    # Standard backward on reference model
    x = batch.clone()
    out_ref = model_ref(x)
    out_ref.sum().backward()

    # Clipped backward with huge norm on clip model
    clipper = PerSampleClipper(model_clip, max_grad_norm=1e6)
    loss_clip = _forward_and_loss(model_clip, x)
    clipper.step(loss_clip)

    for (_, layer_clip), layer_ref in zip(clipper._linear_layers, [model_ref[0], model_ref[2]]):
        # Summed per-sample grads should closely match standard grad (divided by nothing)
        assert torch.allclose(layer_clip.weight.grad, layer_ref.weight.grad, atol=1e-4)


def test_remove_hooks_runs_without_error(clipper: PerSampleClipper) -> None:
    """remove_hooks() completes without raising an exception."""
    clipper.remove_hooks()


def test_pre_clip_norm_mean_non_negative(model: nn.Sequential, batch: torch.Tensor) -> None:
    """step() dict values (pre-clip norm means) are all non-negative."""
    clipper = PerSampleClipper(model, max_grad_norm=1.0)
    loss = _forward_and_loss(model, batch)
    result = clipper.step(loss)
    for name, norm_mean in result.items():
        assert norm_mean >= 0.0, f"Layer {name} has negative pre-clip norm mean: {norm_mean}"


def test_works_with_batch_size_1(model: nn.Sequential) -> None:
    """PerSampleClipper works correctly when batch size is 1."""
    torch.manual_seed(7)
    x = torch.randn(1, 8)
    clipper = PerSampleClipper(model, max_grad_norm=1.0)
    loss = _forward_and_loss(model, x)
    result = clipper.step(loss)
    assert isinstance(result, dict)
    for _, layer in clipper._linear_layers:
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
