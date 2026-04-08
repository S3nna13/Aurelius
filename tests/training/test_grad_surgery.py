"""Tests for PCGrad and CAGrad gradient surgery."""
import torch
import pytest

from src.training.grad_surgery import (
    GradSurgeryConfig,
    MultiTaskGradManager,
    cagrad,
    pcgrad,
    project_gradient,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=32,
    )
    return AureliusTransformer(cfg)


@pytest.fixture
def n_params(small_model):
    return sum(p.numel() for p in small_model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# project_gradient tests
# ---------------------------------------------------------------------------

def test_project_gradient_conflicting():
    """Conflicting gradients: projected g_i has smaller component along g_j."""
    torch.manual_seed(0)
    g_i = torch.tensor([1.0, 0.0])
    g_j = torch.tensor([-1.0, 0.0])   # exactly opposite -> conflicting

    result = project_gradient(g_i, g_j)

    # After projection, component along g_j direction should be reduced / zero
    g_j_unit = g_j / g_j.norm()
    orig_component = torch.dot(g_i, g_j_unit).item()
    proj_component = torch.dot(result, g_j_unit).item()

    assert proj_component > orig_component, (
        "Projected gradient should have a smaller (less negative) component along g_j"
    )


def test_project_gradient_aligned():
    """Aligned gradients: result should equal g_i (unchanged)."""
    g_i = torch.tensor([1.0, 2.0, 3.0])
    g_j = torch.tensor([0.5, 1.0, 1.5])   # same direction as g_i

    result = project_gradient(g_i, g_j)

    assert torch.allclose(result, g_i), "Aligned gradients should not be modified"


# ---------------------------------------------------------------------------
# pcgrad tests
# ---------------------------------------------------------------------------

def test_pcgrad_returns_tensor():
    """pcgrad should return a 1-D tensor of same size as inputs."""
    torch.manual_seed(0)
    D = 50
    grads = [torch.randn(D) for _ in range(3)]

    result = pcgrad(grads)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (D,)


def test_pcgrad_two_tasks_conflicting():
    """With partially conflicting gradients, PCGrad result differs from simple average.

    g1 = [1, 0] and g2 = [-1, 1] are conflicting (cos < 0).
    Simple average = [0, 0.5].
    PCGrad projects each gradient, so result != simple average.
    """
    # g1 · g2 = -1 < 0, so they are conflicting
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([-1.0, 1.0])

    result = pcgrad([g1, g2])
    simple_avg = (g1 + g2) / 2

    # PCGrad should produce a different result from simple averaging
    assert not torch.allclose(result, simple_avg), (
        "PCGrad should handle conflicting gradients differently from simple averaging"
    )


# ---------------------------------------------------------------------------
# cagrad tests
# ---------------------------------------------------------------------------

def test_cagrad_returns_tensor():
    """cagrad should return a 1-D tensor of same size as inputs."""
    torch.manual_seed(0)
    D = 50
    grads = [torch.randn(D) for _ in range(3)]

    result = cagrad(grads, c=0.5)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (D,)


def test_cagrad_c_zero_is_average():
    """With c=0, cagrad should equal the simple average of task gradients."""
    torch.manual_seed(2)
    D = 20
    grads = [torch.randn(D) for _ in range(4)]

    result = cagrad(grads, c=0.0)
    expected = torch.stack(grads).mean(dim=0)

    assert torch.allclose(result, expected, atol=1e-6), (
        "cagrad with c=0 must equal the simple average"
    )


# ---------------------------------------------------------------------------
# MultiTaskGradManager tests
# ---------------------------------------------------------------------------

def _make_dummy_losses(model, n_tasks=3, seq_len=8, vocab_size=256):
    """Create synthetic scalar losses from the model for testing."""
    losses = []
    for _ in range(n_tasks):
        input_ids = torch.randint(0, vocab_size, (2, seq_len))
        labels = torch.randint(0, vocab_size, (2, seq_len))
        loss, _, _ = model(input_ids=input_ids, labels=labels)
        losses.append(loss)
    return losses


def test_grad_manager_compute_gradient(small_model, n_params):
    """compute_merged_gradient should return a flat gradient vector of correct size."""
    torch.manual_seed(3)
    manager = MultiTaskGradManager(small_model, GradSurgeryConfig(method="pcgrad"))
    losses = _make_dummy_losses(small_model)

    merged = manager.compute_merged_gradient(losses)

    assert isinstance(merged, torch.Tensor)
    assert merged.shape == (n_params,)


def test_grad_manager_apply_gradient(small_model, n_params):
    """After apply_gradient, model parameters should have .grad set."""
    torch.manual_seed(4)
    manager = MultiTaskGradManager(small_model, GradSurgeryConfig(method="pcgrad"))
    losses = _make_dummy_losses(small_model)

    merged = manager.compute_merged_gradient(losses)
    manager.apply_gradient(merged)

    grad_params = [p for p in small_model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grad_params) > 0, "At least some parameters should have gradients after apply"


def test_grad_manager_pcgrad_method(small_model):
    """PCGrad method should run end-to-end without errors."""
    torch.manual_seed(5)
    cfg = GradSurgeryConfig(method="pcgrad", normalize=True)
    manager = MultiTaskGradManager(small_model, cfg)
    losses = _make_dummy_losses(small_model, n_tasks=2)

    merged = manager.compute_merged_gradient(losses)
    manager.apply_gradient(merged)

    # Verify gradients are finite
    for p in small_model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Gradients must be finite after PCGrad"


def test_grad_manager_cagrad_method(small_model):
    """CAGrad method should run end-to-end without errors."""
    torch.manual_seed(6)
    cfg = GradSurgeryConfig(method="cagrad", cagrad_c=0.5, normalize=True)
    manager = MultiTaskGradManager(small_model, cfg)
    losses = _make_dummy_losses(small_model, n_tasks=2)

    merged = manager.compute_merged_gradient(losses)
    manager.apply_gradient(merged)

    # Verify gradients are finite
    for p in small_model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Gradients must be finite after CAGrad"
