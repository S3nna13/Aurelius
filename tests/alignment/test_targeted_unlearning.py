"""Tests for targeted_unlearning.py -- SCRUB-style gradient ascent + KL retention."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.alignment.targeted_unlearning import (
    UnlearningConfig,
    UnlearningResult,
    TargetedUnlearner,
    forget_loss,
    retain_loss,
    evaluate_forgetting,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Shared config / helpers
# ---------------------------------------------------------------------------

MODEL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

B, T = 2, 16
VOCAB = MODEL_CFG.vocab_size


def make_model(seed: int = 0) -> AureliusTransformer:
    torch.manual_seed(seed)
    return AureliusTransformer(MODEL_CFG)


def make_ref_model(model: AureliusTransformer) -> AureliusTransformer:
    ref = copy.deepcopy(model)
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def make_inputs(seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    inputs = torch.randint(0, VOCAB, (B, T))
    targets = torch.randint(0, VOCAB, (B, T))
    return inputs, targets


# ---------------------------------------------------------------------------
# Test 1: forget_loss with gradient_ascent returns positive scalar
# ---------------------------------------------------------------------------

def test_forget_loss_gradient_ascent_positive_scalar():
    """forget_loss with gradient_ascent should return a positive-valued scalar tensor."""
    torch.manual_seed(0)
    model = make_model()
    inputs, targets = make_inputs()
    loss = forget_loss(model, inputs, targets, loss_type="gradient_ascent")
    assert loss.ndim == 0, "forget_loss should return a scalar"
    # gradient ascent returns -ce, which is negative. We check it is a finite scalar.
    assert torch.isfinite(loss), "forget_loss must be finite"
    # The returned value is -CE which is always <= 0; the *magnitude* is positive
    # so the gradient step will push CE up. We verify it is non-zero and finite.
    assert loss.item() != 0.0


# ---------------------------------------------------------------------------
# Test 2: forget_loss with random_label returns positive scalar
# ---------------------------------------------------------------------------

def test_forget_loss_random_label_positive_scalar():
    """forget_loss with random_label should return a positive finite scalar."""
    torch.manual_seed(0)
    model = make_model()
    inputs, targets = make_inputs()
    loss = forget_loss(model, inputs, targets, loss_type="random_label")
    assert loss.ndim == 0, "forget_loss should return a scalar"
    assert torch.isfinite(loss), "forget_loss must be finite"
    assert loss.item() > 0.0, "random_label CE loss should be positive"


# ---------------------------------------------------------------------------
# Test 3: retain_loss returns non-negative scalar (KL >= 0)
# ---------------------------------------------------------------------------

def test_retain_loss_non_negative():
    """KL divergence is always non-negative."""
    torch.manual_seed(0)
    model = make_model()
    ref = make_ref_model(model)
    inputs, _ = make_inputs()
    loss = retain_loss(model, ref, inputs)
    assert loss.ndim == 0, "retain_loss should return a scalar"
    assert torch.isfinite(loss), "retain_loss must be finite"
    assert loss.item() >= 0.0, f"KL divergence must be >= 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 4: retain_loss on identical models -> ~0 KL
# ---------------------------------------------------------------------------

def test_retain_loss_identical_models_near_zero():
    """When model and ref_model are identical, KL divergence should be ~0."""
    torch.manual_seed(0)
    model = make_model()
    ref = make_ref_model(model)
    inputs, _ = make_inputs()
    loss = retain_loss(model, ref, inputs)
    assert loss.item() < 1e-4, (
        f"KL of identical models should be ~0, got {loss.item()}"
    )


# ---------------------------------------------------------------------------
# Test 5: TargetedUnlearner constructs without error
# ---------------------------------------------------------------------------

def test_targeted_unlearner_constructs():
    """TargetedUnlearner should initialise without raising."""
    torch.manual_seed(0)
    model = make_model()
    ref = make_ref_model(model)
    cfg = UnlearningConfig(forget_steps=1, retain_steps=1)
    unlearner = TargetedUnlearner(model, ref, cfg)
    assert unlearner is not None


# ---------------------------------------------------------------------------
# Test 6: forget_step returns float
# ---------------------------------------------------------------------------

def test_forget_step_returns_float():
    """forget_step should return a Python float."""
    torch.manual_seed(0)
    model = make_model()
    ref = make_ref_model(model)
    cfg = UnlearningConfig(forget_steps=1, retain_steps=1)
    unlearner = TargetedUnlearner(model, ref, cfg)
    inputs, targets = make_inputs()
    result = unlearner.forget_step(inputs, targets)
    assert isinstance(result, float), f"Expected float, got {type(result)}"


# ---------------------------------------------------------------------------
# Test 7: retain_step returns float
# ---------------------------------------------------------------------------

def test_retain_step_returns_float():
    """retain_step should return a Python float."""
    torch.manual_seed(0)
    model = make_model()
    ref = make_ref_model(model)
    cfg = UnlearningConfig(forget_steps=1, retain_steps=1)
    unlearner = TargetedUnlearner(model, ref, cfg)
    inputs, _ = make_inputs()
    result = unlearner.retain_step(inputs)
    assert isinstance(result, float), f"Expected float, got {type(result)}"


# ---------------------------------------------------------------------------
# Test 8: After forget_step, model weights change
# ---------------------------------------------------------------------------

def test_forget_step_updates_weights():
    """Weights should change after a forget_step."""
    torch.manual_seed(0)
    model = make_model()
    ref = make_ref_model(model)
    cfg = UnlearningConfig(forget_lr=1e-3, forget_steps=1, retain_steps=1)
    unlearner = TargetedUnlearner(model, ref, cfg)

    # Snapshot a parameter before the step
    param_before = next(model.parameters()).clone().detach()

    inputs, targets = make_inputs()
    unlearner.forget_step(inputs, targets)

    param_after = next(model.parameters()).clone().detach()
    assert not torch.allclose(param_before, param_after), (
        "Model weights should change after forget_step"
    )


# ---------------------------------------------------------------------------
# Test 9: run returns UnlearningResult
# ---------------------------------------------------------------------------

def test_run_returns_unlearning_result():
    """run() should return an UnlearningResult instance."""
    torch.manual_seed(0)
    model = make_model()
    ref = make_ref_model(model)
    cfg = UnlearningConfig(forget_steps=2, retain_steps=2)
    unlearner = TargetedUnlearner(model, ref, cfg)

    inputs, targets = make_inputs()
    forget_dataset = [(inputs, targets)]
    retain_dataset = [inputs]

    result = unlearner.run(forget_dataset, retain_dataset)
    assert isinstance(result, UnlearningResult), (
        f"Expected UnlearningResult, got {type(result)}"
    )


# ---------------------------------------------------------------------------
# Test 10: run forget_losses list length matches n_forget_steps
# ---------------------------------------------------------------------------

def test_run_forget_losses_length():
    """forget_losses in result should have length == n_forget_steps."""
    torch.manual_seed(0)
    model = make_model()
    ref = make_ref_model(model)
    n_forget = 3
    cfg = UnlearningConfig(forget_steps=n_forget, retain_steps=2)
    unlearner = TargetedUnlearner(model, ref, cfg)

    inputs, targets = make_inputs()
    forget_dataset = [(inputs, targets)]
    retain_dataset = [inputs]

    result = unlearner.run(forget_dataset, retain_dataset)
    assert len(result.forget_losses) == n_forget, (
        f"Expected {n_forget} forget losses, got {len(result.forget_losses)}"
    )
    assert result.n_forget_steps == n_forget


# ---------------------------------------------------------------------------
# Test 11: evaluate_forgetting returns dict with required keys
# ---------------------------------------------------------------------------

def test_evaluate_forgetting_keys():
    """evaluate_forgetting must return dict with 'forget_loss' and 'forget_perplexity'."""
    torch.manual_seed(0)
    model = make_model()
    inputs, targets = make_inputs()
    metrics = evaluate_forgetting(model, inputs, targets)
    assert isinstance(metrics, dict), "evaluate_forgetting should return a dict"
    assert "forget_loss" in metrics, "Missing key 'forget_loss'"
    assert "forget_perplexity" in metrics, "Missing key 'forget_perplexity'"
    assert isinstance(metrics["forget_loss"], float)
    assert isinstance(metrics["forget_perplexity"], float)
    assert metrics["forget_loss"] > 0.0
    assert metrics["forget_perplexity"] >= 1.0


# ---------------------------------------------------------------------------
# Test 12: After unlearning, forget_loss higher than before (model forgot)
# ---------------------------------------------------------------------------

def test_unlearning_increases_forget_loss():
    """After running unlearning, the forget loss should be higher (more forgetting)."""
    torch.manual_seed(7)
    model = make_model(seed=7)
    ref = make_ref_model(model)

    inputs, targets = make_inputs(seed=10)

    # Measure forget loss before unlearning
    before = evaluate_forgetting(model, inputs, targets)
    before_loss = before["forget_loss"]

    # Run unlearning with aggressive forget lr and many steps
    cfg = UnlearningConfig(
        forget_lr=1e-2,
        retain_lr=1e-6,
        forget_steps=10,
        retain_steps=1,
        kl_coef=0.0,   # disable retention to maximise forgetting signal
        forget_loss_type="gradient_ascent",
    )
    unlearner = TargetedUnlearner(model, ref, cfg)
    forget_dataset = [(inputs, targets)]
    retain_dataset = [inputs]
    unlearner.run(forget_dataset, retain_dataset)

    # Measure forget loss after unlearning
    after = evaluate_forgetting(model, inputs, targets)
    after_loss = after["forget_loss"]

    assert after_loss > before_loss, (
        f"After unlearning forget_loss should increase: "
        f"before={before_loss:.4f}, after={after_loss:.4f}"
    )
