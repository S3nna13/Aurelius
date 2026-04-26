"""Tests for src/training/ema.py -- ModelEMA and StochasticWeightAveraging."""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn

from src.training.ema import ModelEMA, StochasticWeightAveraging

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_model() -> nn.Module:
    """Tiny deterministic model for testing."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )


def params_list(model: nn.Module):
    """Return a list of cloned parameter tensors."""
    return [p.detach().clone() for p in model.parameters()]


def params_equal(a_list, b_list, atol: float = 1e-6) -> bool:
    return all(torch.allclose(a, b, atol=atol) for a, b in zip(a_list, b_list))


# ---------------------------------------------------------------------------
# 1. ModelEMA initializes with correct decay
# ---------------------------------------------------------------------------


def test_modelema_init_decay():
    model = make_model()
    ema = ModelEMA(model, decay=0.9999)
    assert ema.decay == 0.9999


# ---------------------------------------------------------------------------
# 2. Shadow params are initialized as copy of model params
# ---------------------------------------------------------------------------


def test_modelema_shadow_init_matches_model():
    model = make_model()
    ema = ModelEMA(model, decay=0.9999)
    for shadow, param in zip(ema.shadow_params, model.parameters()):
        assert torch.allclose(shadow.float(), param.detach().float(), atol=1e-6), (
            "Shadow params should match model params at initialization."
        )


# ---------------------------------------------------------------------------
# 3. update() changes shadow params toward model params
# ---------------------------------------------------------------------------


def test_modelema_update_moves_shadow():
    torch.manual_seed(0)
    model = make_model()
    ema = ModelEMA(model, decay=0.9)

    # Record initial shadow
    initial_shadow = [s.clone() for s in ema.shadow_params]

    # Perturb model params
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p) * 10.0)

    ema.update(model)

    changed = any(
        not torch.allclose(s, init, atol=1e-6) for s, init in zip(ema.shadow_params, initial_shadow)
    )
    assert changed, "Shadow params should change after update() when model params differ."


# ---------------------------------------------------------------------------
# 4. After update with decay=0.0, shadow matches model instantly
# ---------------------------------------------------------------------------


def test_modelema_decay_zero_instant_match():
    torch.manual_seed(1)
    model = make_model()
    ema = ModelEMA(model, decay=0.0)

    target_val = 7.0
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(target_val)

    ema.update(model)

    for shadow in ema.shadow_params:
        assert torch.allclose(shadow.float(), torch.full_like(shadow, target_val), atol=1e-5), (
            "With decay=0.0, shadow should match model params after one update."
        )


# ---------------------------------------------------------------------------
# 5. effective_decay increases with step (warmup correction)
# ---------------------------------------------------------------------------


def test_modelema_effective_decay_increases_with_step():
    model = make_model()
    ema = ModelEMA(model, decay=0.9999)
    decays = []
    for _ in range(20):
        decays.append(ema.effective_decay)
        ema._step += 1

    for i in range(1, len(decays)):
        assert decays[i] >= decays[i - 1], (
            f"effective_decay should be non-decreasing: {decays[i - 1]:.6f} -> {decays[i]:.6f}"
        )


# ---------------------------------------------------------------------------
# 6. effective_decay caps at declared decay after warmup
# ---------------------------------------------------------------------------


def test_modelema_effective_decay_caps_at_decay():
    model = make_model()
    declared_decay = 0.99
    ema = ModelEMA(model, decay=declared_decay)
    ema._step = 10_000_000
    assert ema.effective_decay <= declared_decay + 1e-9
    assert ema.effective_decay == pytest.approx(declared_decay, rel=1e-4)


# ---------------------------------------------------------------------------
# 7. copy_to() copies shadow params into model
# ---------------------------------------------------------------------------


def test_modelema_copy_to():
    torch.manual_seed(2)
    model = make_model()
    ema = ModelEMA(model, decay=0.9)

    with torch.no_grad():
        for s in ema.shadow_params:
            s.fill_(99.0)

    ema.copy_to(model)
    for p in model.parameters():
        assert torch.allclose(p, torch.full_like(p, 99.0), atol=1e-5), (
            "copy_to() should overwrite model params with shadow values."
        )


# ---------------------------------------------------------------------------
# 8. store() + restore() recovers original params
# ---------------------------------------------------------------------------


def test_modelema_store_restore():
    torch.manual_seed(3)
    model = make_model()
    ema = ModelEMA(model, decay=0.9999)

    original = params_list(model)
    ema.store(model)

    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.0)

    ema.restore(model)
    restored = params_list(model)

    assert params_equal(original, restored), "restore() should recover the params saved by store()."


# ---------------------------------------------------------------------------
# 9. average_parameters() context manager restores after exit
# ---------------------------------------------------------------------------


def test_modelema_context_manager_restores():
    torch.manual_seed(4)
    model = make_model()
    ema = ModelEMA(model, decay=0.9999)

    original = params_list(model)

    with ema.average_parameters(model):
        pass

    restored = params_list(model)
    assert params_equal(original, restored), (
        "average_parameters() should restore original params after context exit."
    )


# ---------------------------------------------------------------------------
# 10. average_parameters(): params differ inside vs outside (EMA effect)
# ---------------------------------------------------------------------------


def test_modelema_context_manager_ema_active_inside():
    torch.manual_seed(5)
    model = make_model()
    ema = ModelEMA(model, decay=0.5)

    # Run updates with model set to 100.0
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(100.0)
    for _ in range(5):
        ema.update(model)

    # Now set model to 0.0
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.0)

    outside_params = params_list(model)

    inside_params = []
    with ema.average_parameters(model):
        inside_params = params_list(model)

    any_different = any(
        not torch.allclose(i, o, atol=1e-3) for i, o in zip(inside_params, outside_params)
    )
    assert any_different, "Inside average_parameters(), model params should reflect EMA shadow."


# ---------------------------------------------------------------------------
# 11. state_dict() returns serializable dict
# ---------------------------------------------------------------------------


def test_modelema_state_dict_serializable():
    model = make_model()
    ema = ModelEMA(model, decay=0.9999)
    ema._step = 5

    sd = ema.state_dict()

    # Verify all expected keys
    assert "decay" in sd
    assert "step" in sd
    assert "shadow_params" in sd
    assert sd["step"] == 5
    assert sd["decay"] == 0.9999
    assert isinstance(sd["shadow_params"], list)
    assert all(isinstance(p, torch.Tensor) for p in sd["shadow_params"])

    # Verify torch.save compatible (use BytesIO buffer)
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    loaded_sd = torch.load(buf, weights_only=False)
    assert loaded_sd["step"] == 5


# ---------------------------------------------------------------------------
# 12. load_state_dict() round-trips correctly
# ---------------------------------------------------------------------------


def test_modelema_load_state_dict_roundtrip():
    torch.manual_seed(6)
    model = make_model()
    ema = ModelEMA(model, decay=0.9995, warmup_steps=500)
    ema._step = 10

    with torch.no_grad():
        for p in model.parameters():
            p.fill_(3.14)
    ema.update(model)

    sd = ema.state_dict()

    model2 = make_model()
    ema2 = ModelEMA(model2, decay=0.5)
    ema2.load_state_dict(sd)

    assert ema2.decay == ema.decay
    assert ema2._step == ema._step
    for s1, s2 in zip(ema.shadow_params, ema2.shadow_params):
        assert torch.allclose(s1.cpu(), s2.cpu(), atol=1e-6), (
            "Shadow params should match after load_state_dict()."
        )


# ---------------------------------------------------------------------------
# 13. get_ema_model() returns nn.Module with EMA weights, doesn't modify original
# ---------------------------------------------------------------------------


def test_modelema_get_ema_model():
    torch.manual_seed(7)
    model = make_model()
    ema = ModelEMA(model, decay=0.5)

    with torch.no_grad():
        for p in model.parameters():
            p.fill_(50.0)
    ema.update(model)

    original_params = params_list(model)

    ema_model = ema.get_ema_model(model)
    assert isinstance(ema_model, nn.Module), "get_ema_model() should return nn.Module."

    after_params = params_list(model)
    assert params_equal(original_params, after_params), (
        "get_ema_model() must not modify the original model."
    )

    for ema_p, shadow in zip(ema_model.parameters(), ema.shadow_params):
        assert torch.allclose(ema_p.float(), shadow.float(), atol=1e-5), (
            "EMA model params should match shadow params."
        )


# ---------------------------------------------------------------------------
# 14. SWA.update() returns False before swa_start_step
# ---------------------------------------------------------------------------


def test_swa_update_false_before_start():
    model = make_model()
    swa = StochasticWeightAveraging(model, swa_start_step=1000, swa_freq=100)

    for step in [0, 100, 500, 999]:
        result = swa.update(model, step)
        assert result is False, f"Should return False at step={step}."
    assert swa.n_averaged == 0


# ---------------------------------------------------------------------------
# 15. SWA.update() returns True and increments n_averaged at correct steps
# ---------------------------------------------------------------------------


def test_swa_update_true_at_correct_steps():
    model = make_model()
    swa_start = 100
    swa_freq = 50
    swa = StochasticWeightAveraging(model, swa_start_step=swa_start, swa_freq=swa_freq)

    results = {}
    for step in range(0, 350, 10):
        r = swa.update(model, step)
        results[step] = r

    expected_true = {100, 150, 200, 250, 300}
    for step, result in results.items():
        if step in expected_true:
            assert result is True, f"Expected True at step={step}."
        else:
            assert result is False, f"Expected False at step={step}."

    assert swa.n_averaged == len(expected_true), (
        f"Expected {len(expected_true)} averaged checkpoints, got {swa.n_averaged}."
    )


# ---------------------------------------------------------------------------
# 16. SWA.get_swa_model() params are average of accumulated checkpoints
# ---------------------------------------------------------------------------


def test_swa_get_swa_model_is_average():
    torch.manual_seed(8)
    model = make_model()
    swa = StochasticWeightAveraging(model, swa_start_step=0, swa_freq=1)

    for i in range(5):
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float(i + 1))  # values: 1, 2, 3, 4, 5
        swa.update(model, step=i)

    assert swa.n_averaged == 5

    swa_model = swa.get_swa_model()

    expected = 3.0  # mean of 1..5
    for p in swa_model.parameters():
        assert torch.allclose(p.float(), torch.full_like(p, expected), atol=1e-5), (
            f"SWA model param should be average of checkpoints (3.0), got {p.float().mean().item():.4f}."  # noqa: E501
        )
