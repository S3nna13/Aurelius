"""Tests for loss-landscape helpers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.loss_landscape import (
    LandscapeConfig,
    SAMOptimizer,
    compute_gradient_norm,
    compute_hessian_trace,
    compute_weight_norm,
    measure_sharpness,
    sam_first_step,
)

# ---------------------------------------------------------------------------
# Small model fixture shared by all tests
# ---------------------------------------------------------------------------

SMALL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


def make_model() -> AureliusTransformer:
    torch.manual_seed(0)
    return AureliusTransformer(SMALL_CFG)


def make_input(batch: int = 2, seq: int = 16) -> torch.Tensor:
    return torch.randint(0, SMALL_CFG.vocab_size, (batch, seq))


def ce_loss_fn(model: nn.Module) -> torch.Tensor:
    """Simple cross-entropy loss for use with measure_sharpness."""
    input_ids = make_input()
    with torch.no_grad():
        _, logits, _ = model(input_ids)
    # logits: (B, S, V) — use next-token prediction
    shift_logits = logits[:, :-1, :].reshape(-1, SMALL_CFG.vocab_size)
    shift_labels = input_ids[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels)


def compute_loss_with_grad(model: nn.Module) -> torch.Tensor:
    """Compute a loss that allows grad computation (no torch.no_grad)."""
    input_ids = make_input()
    _, logits, _ = model(input_ids)
    shift_logits = logits[:, :-1, :].reshape(-1, SMALL_CFG.vocab_size)
    shift_labels = input_ids[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_landscape_config_defaults():
    cfg = LandscapeConfig()
    assert cfg.rho == 0.05
    assert cfg.sharpness_n_perturbations == 10
    assert cfg.sharpness_epsilon == 0.01
    assert cfg.adaptive_sam is False


def test_compute_gradient_norm_returns_float():
    model = make_model()
    loss = compute_loss_with_grad(model)
    loss.backward()
    result = compute_gradient_norm(model)
    assert isinstance(result, float)


def test_compute_gradient_norm_zero_when_no_grads():
    model = make_model()
    # No backward called — no gradients
    result = compute_gradient_norm(model)
    assert result == 0.0


def test_compute_weight_norm_returns_positive_float():
    model = make_model()
    result = compute_weight_norm(model)
    assert isinstance(result, float)
    assert result > 0.0


def test_measure_sharpness_returns_float():
    model = make_model()
    result = measure_sharpness(model, ce_loss_fn, n_perturbations=3, epsilon=0.01)
    assert isinstance(result, float)


def test_measure_sharpness_higher_epsilon_higher_sharpness():
    """Higher epsilon should yield higher (or equal) sharpness on average."""
    torch.manual_seed(42)
    model = make_model()
    sharpness_small = measure_sharpness(model, ce_loss_fn, n_perturbations=20, epsilon=0.001)
    sharpness_large = measure_sharpness(model, ce_loss_fn, n_perturbations=20, epsilon=0.1)
    # With larger perturbations, the loss increase should be larger
    assert sharpness_large >= sharpness_small - 0.1, (
        f"Expected larger epsilon to give higher sharpness: "
        f"small={sharpness_small:.4f}, large={sharpness_large:.4f}"
    )


def test_sam_first_step_returns_original_params_dict():
    model = make_model()
    loss = compute_loss_with_grad(model)
    original_params = sam_first_step(model, loss, rho=0.05)
    assert isinstance(original_params, dict)
    assert len(original_params) == len(dict(model.named_parameters()))
    # All values should be Tensors
    for name, tensor in original_params.items():
        assert isinstance(tensor, torch.Tensor)


def test_sam_first_step_perturbs_model_parameters():
    model = make_model()
    # Save params before perturbation
    params_before = {name: p.data.clone() for name, p in model.named_parameters()}
    loss = compute_loss_with_grad(model)
    sam_first_step(model, loss, rho=0.05)
    # At least some parameters should have changed
    any_changed = any(
        not torch.allclose(p.data, params_before[name]) for name, p in model.named_parameters()
    )
    assert any_changed, "sam_first_step should perturb model parameters"


def test_sam_optimizer_first_and_second_step_run():
    model = make_model()
    base_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    cfg = LandscapeConfig(rho=0.05)
    sam = SAMOptimizer(base_opt, model, cfg)

    # First step
    sam.zero_grad()
    loss1 = compute_loss_with_grad(model)
    sam.first_step(loss1)

    # Second step (loss at perturbed point)
    loss2 = compute_loss_with_grad(model)
    sam.second_step(loss2)  # should not raise


def test_compute_hessian_trace_returns_float():
    model = make_model()
    loss = compute_loss_with_grad(model)
    result = compute_hessian_trace(model, loss, n_samples=2)
    assert isinstance(result, float)


def test_compute_gradient_norm_after_backward_is_positive():
    model = make_model()
    loss = compute_loss_with_grad(model)
    loss.backward()
    result = compute_gradient_norm(model)
    assert result > 0.0, "Gradient norm should be positive after backward"


def test_sam_optimizer_zero_grad_clears_gradients():
    model = make_model()
    base_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    cfg = LandscapeConfig()
    sam = SAMOptimizer(base_opt, model, cfg)

    # Create some gradients
    loss = compute_loss_with_grad(model)
    loss.backward()

    # Verify gradients exist
    has_grads_before = any(p.grad is not None for p in model.parameters())
    assert has_grads_before, "Should have gradients before zero_grad"

    sam.zero_grad()

    # After zero_grad, gradients should be zeroed
    all_zero = all(p.grad is None or p.grad.abs().max().item() == 0.0 for p in model.parameters())
    assert all_zero, "Gradients should be cleared after zero_grad"
