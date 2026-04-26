"""Tests for WARM: Weight Averaged Reward Models (Ramé et al., arXiv:2401.12187).

16 tests covering WARMEnsemble, WARMInterpolation, WARMRewardModel, WARMTrainer.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.alignment.warm import (
    WARMEnsemble,
    WARMInterpolation,
    WARMRewardModel,
    WARMTrainer,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_linear(in_f: int = 8, out_f: int = 4, seed: int = 0) -> nn.Module:
    """Create a tiny nn.Linear with a fixed seed."""
    torch.manual_seed(seed)
    return nn.Linear(in_f, out_f)


def _make_reward_model(d_model: int = 8, seed: int = 0) -> WARMRewardModel:
    """Build a WARMRewardModel with an nn.Identity backbone."""
    torch.manual_seed(seed)
    backbone = nn.Identity()
    rm = WARMRewardModel(backbone=backbone, d_model=d_model)
    # Re-seed the reward head after construction for reproducibility.
    torch.manual_seed(seed)
    nn.init.normal_(rm.reward_head.weight)
    return rm


def _make_linear_reward_model(d_model: int = 8, seed: int = 0) -> WARMRewardModel:
    """Build a WARMRewardModel with a real Linear backbone for gradient tests."""
    torch.manual_seed(seed)
    backbone = nn.Linear(d_model, d_model)
    rm = WARMRewardModel(backbone=backbone, d_model=d_model)
    return rm


# ---------------------------------------------------------------------------
# Test 1: WARMEnsemble.average_weights returns state dict with same keys
# ---------------------------------------------------------------------------


def test_average_weights_same_keys():
    """average_weights() must return a state dict with the same keys as the models."""
    models = [_make_linear(seed=i) for i in range(3)]
    ensemble = WARMEnsemble(models)
    averaged = ensemble.average_weights()

    ref_keys = set(models[0].state_dict().keys())
    assert set(averaged.keys()) == ref_keys, (
        f"averaged keys {set(averaged.keys())} != expected {ref_keys}"
    )


# ---------------------------------------------------------------------------
# Test 2: Average of identical models = original model weights
# ---------------------------------------------------------------------------


def test_average_identical_models_equals_original():
    """Averaging k identical models must return the same weights."""
    base = _make_linear(seed=42)
    models = [copy.deepcopy(base) for _ in range(4)]
    ensemble = WARMEnsemble(models)
    averaged = ensemble.average_weights()

    for key, val in base.state_dict().items():
        assert torch.allclose(averaged[key].float(), val.float(), atol=1e-5), (
            f"Average of identical models mismatch at key '{key}'"
        )


# ---------------------------------------------------------------------------
# Test 3: Average of two models: each weight is mean of corresponding weights
# ---------------------------------------------------------------------------


def test_average_two_models_is_mean():
    """Average of two models must equal (W_a + W_b) / 2 for each parameter."""
    model_a = _make_linear(seed=1)
    model_b = _make_linear(seed=2)
    ensemble = WARMEnsemble([model_a, model_b])
    averaged = ensemble.average_weights()

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    for key in sd_a:
        expected = (sd_a[key].float() + sd_b[key].float()) / 2.0
        assert torch.allclose(averaged[key].float(), expected, atol=1e-5), (
            f"Average of two models mismatch at key '{key}'"
        )


# ---------------------------------------------------------------------------
# Test 4: get_averaged_model returns nn.Module with correct weights
# ---------------------------------------------------------------------------


def test_get_averaged_model_weights():
    """get_averaged_model must load averaged weights into a copy of base_model."""
    models = [_make_linear(seed=i) for i in range(3)]
    ensemble = WARMEnsemble(models)
    base = _make_linear(seed=99)  # fresh model to load into

    averaged_model = ensemble.get_averaged_model(base)
    expected_sd = ensemble.average_weights()

    assert isinstance(averaged_model, nn.Module)
    for key in expected_sd:
        assert torch.allclose(
            averaged_model.state_dict()[key].float(),
            expected_sd[key].float(),
            atol=1e-5,
        ), f"get_averaged_model weight mismatch at key '{key}'"


# ---------------------------------------------------------------------------
# Test 5: predict output shape is (B,)
# ---------------------------------------------------------------------------


def test_predict_output_shape():
    """predict() must return a (B,) tensor."""
    B, d = 6, 8
    models = [_make_reward_model(d_model=d, seed=i) for i in range(3)]
    ensemble = WARMEnsemble(models)
    x = torch.randn(B, d)
    out = ensemble.predict(x)
    assert out.shape == (B,), f"predict output shape {out.shape} != ({B},)"


# ---------------------------------------------------------------------------
# Test 6: Ensemble predict ≠ any single model (different weights → different outputs)
# ---------------------------------------------------------------------------


def test_predict_differs_from_single_model():
    """predict() (ensemble average) must differ from any individual model output."""
    B, d = 4, 8
    # Use models with very different seeds so weights differ substantially.
    models = [_make_reward_model(d_model=d, seed=i * 100) for i in range(3)]
    ensemble = WARMEnsemble(models)
    x = torch.randn(B, d)

    ensemble_out = ensemble.predict(x)

    for idx, model in enumerate(models):
        with torch.no_grad():
            single_out = model(x)
        # They should not all be identical — at least one model differs.
        if not torch.allclose(ensemble_out, single_out, atol=1e-6):
            return  # found a difference, test passes

    pytest.fail(
        "predict() returned identical output to all individual models — "
        "expected at least one to differ."
    )


# ---------------------------------------------------------------------------
# Test 7: WARMInterpolation.interpolate(0.0) returns model_a weights exactly
# ---------------------------------------------------------------------------


def test_interpolate_alpha0_returns_model_a():
    """interpolate(0.0) must return model_a weights exactly."""
    model_a = _make_linear(seed=1)
    model_b = _make_linear(seed=2)
    interp = WARMInterpolation(model_a, model_b)
    result = interp.interpolate(0.0)

    for key, val in model_a.state_dict().items():
        assert torch.allclose(result[key].float(), val.float(), atol=1e-6), (
            f"interpolate(0.0) mismatch at key '{key}'"
        )


# ---------------------------------------------------------------------------
# Test 8: WARMInterpolation.interpolate(1.0) returns model_b weights exactly
# ---------------------------------------------------------------------------


def test_interpolate_alpha1_returns_model_b():
    """interpolate(1.0) must return model_b weights exactly."""
    model_a = _make_linear(seed=1)
    model_b = _make_linear(seed=2)
    interp = WARMInterpolation(model_a, model_b)
    result = interp.interpolate(1.0)

    for key, val in model_b.state_dict().items():
        assert torch.allclose(result[key].float(), val.float(), atol=1e-6), (
            f"interpolate(1.0) mismatch at key '{key}'"
        )


# ---------------------------------------------------------------------------
# Test 9: WARMInterpolation.interpolate(0.5) = average of both
# ---------------------------------------------------------------------------


def test_interpolate_alpha05_is_average():
    """interpolate(0.5) must equal (W_a + W_b) / 2."""
    model_a = _make_linear(seed=3)
    model_b = _make_linear(seed=7)
    interp = WARMInterpolation(model_a, model_b)
    result = interp.interpolate(0.5)

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    for key in sd_a:
        expected = (sd_a[key].float() + sd_b[key].float()) / 2.0
        assert torch.allclose(result[key].float(), expected, atol=1e-5), (
            f"interpolate(0.5) mismatch at key '{key}'"
        )


# ---------------------------------------------------------------------------
# Test 10: sweep returns correct number of models
# ---------------------------------------------------------------------------


def test_sweep_returns_correct_number_of_models():
    """sweep(alphas, base) must return exactly len(alphas) models."""
    model_a = _make_linear(seed=0)
    model_b = _make_linear(seed=1)
    interp = WARMInterpolation(model_a, model_b)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    base = _make_linear(seed=99)
    swept = interp.sweep(alphas, base)

    assert len(swept) == len(alphas), f"sweep returned {len(swept)} models, expected {len(alphas)}"
    for m in swept:
        assert isinstance(m, nn.Module)


# ---------------------------------------------------------------------------
# Test 11: WARMRewardModel output shape (B,) for 2-D input
# ---------------------------------------------------------------------------


def test_reward_model_output_shape_2d():
    """WARMRewardModel must return (B,) for 2-D (B, d_model) input."""
    B, d = 5, 16
    rm = _make_linear_reward_model(d_model=d, seed=0)
    x = torch.randn(B, d)
    out = rm(x)
    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 12: WARMRewardModel output shape (B,) for 3-D input (uses last token)
# ---------------------------------------------------------------------------


def test_reward_model_output_shape_3d():
    """WARMRewardModel must return (B,) for 3-D (B, T, d_model) input."""
    B, T, d = 4, 10, 16
    rm = _make_linear_reward_model(d_model=d, seed=0)
    x = torch.randn(B, T, d)
    out = rm(x)
    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 13: Gradient flows through WARMRewardModel
# ---------------------------------------------------------------------------


def test_reward_model_gradient_flows():
    """Gradients must flow back through WARMRewardModel."""
    B, d = 3, 8
    rm = _make_linear_reward_model(d_model=d, seed=0)
    x = torch.randn(B, d, requires_grad=True)
    out = rm(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient w.r.t. input"
    assert torch.isfinite(x.grad).all(), "Input gradient contains non-finite values"

    for name, param in rm.named_parameters():
        assert param.grad is not None, f"No gradient for parameter '{name}'"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for parameter '{name}'"


# ---------------------------------------------------------------------------
# Test 14: WARMTrainer.train_step returns list of losses with len = n_models
# ---------------------------------------------------------------------------


def test_trainer_train_step_returns_correct_number_of_losses():
    """train_step must return a list of losses with length equal to n_models."""
    n_models = 3
    d = 8
    models = [_make_linear_reward_model(d_model=d, seed=i) for i in range(n_models)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=1e-3) for m in models]
    trainer = WARMTrainer(models=models, optimizers=optimizers)

    B = 4
    x_w = torch.randn(B, d)
    x_l = torch.randn(B, d)
    losses = trainer.train_step(x_w, x_l)

    assert len(losses) == n_models, f"train_step returned {len(losses)} losses, expected {n_models}"


# ---------------------------------------------------------------------------
# Test 15: All losses are finite
# ---------------------------------------------------------------------------


def test_trainer_losses_are_finite():
    """All per-model losses returned by train_step must be finite scalars."""
    n_models = 4
    d = 8
    models = [_make_linear_reward_model(d_model=d, seed=i) for i in range(n_models)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in models]
    trainer = WARMTrainer(models=models, optimizers=optimizers)

    B = 6
    x_w = torch.randn(B, d)
    x_l = torch.randn(B, d)
    losses = trainer.train_step(x_w, x_l)

    for idx, loss in enumerate(losses):
        assert loss.shape == torch.Size([]), f"Loss {idx} is not a scalar: shape {loss.shape}"
        assert torch.isfinite(loss), f"Loss {idx} is not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# Test 16: get_ensemble returns WARMEnsemble of correct size
# ---------------------------------------------------------------------------


def test_trainer_get_ensemble_correct_size():
    """get_ensemble() must return a WARMEnsemble containing all trainer models."""
    n_models = 5
    d = 8
    models = [_make_linear_reward_model(d_model=d, seed=i) for i in range(n_models)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=1e-3) for m in models]
    trainer = WARMTrainer(models=models, optimizers=optimizers)

    ensemble = trainer.get_ensemble()

    assert isinstance(ensemble, WARMEnsemble), (
        f"get_ensemble() returned {type(ensemble)}, expected WARMEnsemble"
    )
    assert len(ensemble._state_dicts) == n_models, (
        f"Ensemble has {len(ensemble._state_dicts)} state dicts, expected {n_models}"
    )
