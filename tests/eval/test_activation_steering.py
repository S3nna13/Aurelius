"""Tests for src/eval/activation_steering.py."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.eval.activation_steering import (
    SteeringConfig,
    compute_mean_activation,
    compute_steering_vector,
    SteeringHook,
    apply_steering,
    measure_steering_effect,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def tiny_model(tiny_cfg: AureliusConfig) -> AureliusTransformer:
    torch.manual_seed(42)
    model = AureliusTransformer(tiny_cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def d_model(tiny_cfg: AureliusConfig) -> int:
    return tiny_cfg.d_model


@pytest.fixture(scope="module")
def input_ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 256, (1, 8))


@pytest.fixture(scope="module")
def steering_vec(d_model: int) -> torch.Tensor:
    torch.manual_seed(7)
    v = torch.randn(d_model)
    return F.normalize(v, dim=0)


# ---------------------------------------------------------------------------
# 1. SteeringConfig defaults
# ---------------------------------------------------------------------------

def test_steering_config_defaults():
    cfg = SteeringConfig()
    assert cfg.layer_idx == 12
    assert cfg.coeff == 20.0
    assert cfg.mode == "add"
    assert cfg.normalize is True


# ---------------------------------------------------------------------------
# 2. compute_mean_activation - output shape (d_model,)
# ---------------------------------------------------------------------------

def test_compute_mean_activation_shape(tiny_model, d_model):
    input_ids_list = [torch.randint(0, 256, (1, 6)) for _ in range(3)]
    act = compute_mean_activation(tiny_model, input_ids_list, layer_idx=0)
    assert act.shape == (d_model,), f"Expected ({d_model},), got {act.shape}"


# ---------------------------------------------------------------------------
# 3. compute_mean_activation - same input twice gives same result
# ---------------------------------------------------------------------------

def test_compute_mean_activation_deterministic(tiny_model, d_model):
    input_ids_list = [torch.randint(0, 256, (1, 6))]
    act1 = compute_mean_activation(tiny_model, input_ids_list, layer_idx=0)
    act2 = compute_mean_activation(tiny_model, input_ids_list, layer_idx=0)
    assert torch.allclose(act1, act2), "Same input should give identical activations"


# ---------------------------------------------------------------------------
# 4. compute_steering_vector - output shape (d_model,)
# ---------------------------------------------------------------------------

def test_compute_steering_vector_shape(d_model):
    pos = torch.randn(4, d_model)
    neg = torch.randn(4, d_model)
    vec = compute_steering_vector(pos, neg)
    assert vec.shape == (d_model,), f"Expected ({d_model},), got {vec.shape}"


# ---------------------------------------------------------------------------
# 5. compute_steering_vector - is L2-normalized (norm approx 1)
# ---------------------------------------------------------------------------

def test_compute_steering_vector_normalized(d_model):
    torch.manual_seed(1)
    pos = torch.randn(4, d_model)
    neg = torch.randn(4, d_model)
    vec = compute_steering_vector(pos, neg)
    norm = vec.norm().item()
    assert abs(norm - 1.0) < 1e-5, f"Expected norm approx 1, got {norm}"


# ---------------------------------------------------------------------------
# 6. compute_steering_vector - opposite inputs give opposite directions
# ---------------------------------------------------------------------------

def test_compute_steering_vector_opposite(d_model):
    torch.manual_seed(2)
    pos = torch.randn(3, d_model)
    neg = torch.randn(3, d_model)
    vec_fwd = compute_steering_vector(pos, neg)
    vec_rev = compute_steering_vector(neg, pos)
    # They should be anti-parallel (dot product approx -1)
    dot = (vec_fwd * vec_rev).sum().item()
    assert dot < -0.99, f"Expected anti-parallel vectors, dot={dot}"


# ---------------------------------------------------------------------------
# 7. SteeringHook - as context manager (enter/exit without error)
# ---------------------------------------------------------------------------

def test_steering_hook_context_manager(tiny_model, steering_vec):
    cfg = SteeringConfig(layer_idx=0, coeff=1.0, mode="add")
    hook = SteeringHook(steering_vec, cfg)
    hook.register(tiny_model)
    with hook:
        pass  # just test that __enter__/__exit__ don't raise


# ---------------------------------------------------------------------------
# 8. apply_steering - returns shape (B, T, V)
# ---------------------------------------------------------------------------

def test_apply_steering_output_shape(tiny_model, steering_vec, tiny_cfg):
    cfg = SteeringConfig(layer_idx=0, coeff=5.0, mode="add")
    ids = torch.randint(0, 256, (2, 6))
    logits = apply_steering(tiny_model, ids, steering_vec, cfg)
    B, T, V = 2, 6, tiny_cfg.vocab_size
    assert logits.shape == (B, T, V), f"Expected {(B, T, V)}, got {logits.shape}"


# ---------------------------------------------------------------------------
# 9. apply_steering with coeff=0 gives same logits as baseline (within tolerance)
# ---------------------------------------------------------------------------

def test_apply_steering_zero_coeff_matches_baseline(tiny_model, steering_vec):
    cfg = SteeringConfig(layer_idx=0, coeff=0.0, mode="add", normalize=True)
    ids = torch.randint(0, 256, (1, 5))
    with torch.no_grad():
        _, logits_base, _ = tiny_model(ids)
    logits_steered = apply_steering(tiny_model, ids, steering_vec, cfg)
    assert torch.allclose(logits_base, logits_steered, atol=1e-5), (
        "coeff=0 steering should not change logits"
    )


# ---------------------------------------------------------------------------
# 10. apply_steering mode="add" changes logits (coeff=10)
# ---------------------------------------------------------------------------

def test_apply_steering_add_changes_logits(tiny_model, steering_vec):
    cfg = SteeringConfig(layer_idx=0, coeff=10.0, mode="add")
    ids = torch.randint(0, 256, (1, 5))
    with torch.no_grad():
        _, logits_base, _ = tiny_model(ids)
    logits_steered = apply_steering(tiny_model, ids, steering_vec, cfg)
    assert not torch.allclose(logits_base, logits_steered, atol=1e-4), (
        "mode='add' with coeff=10 should change logits"
    )


# ---------------------------------------------------------------------------
# 11. apply_steering mode="subtract" changes logits differently than "add"
# ---------------------------------------------------------------------------

def test_apply_steering_subtract_differs_from_add(tiny_model, steering_vec):
    ids = torch.randint(0, 256, (1, 5))
    cfg_add = SteeringConfig(layer_idx=0, coeff=10.0, mode="add")
    cfg_sub = SteeringConfig(layer_idx=0, coeff=10.0, mode="subtract")
    logits_add = apply_steering(tiny_model, ids, steering_vec, cfg_add)
    logits_sub = apply_steering(tiny_model, ids, steering_vec, cfg_sub)
    assert not torch.allclose(logits_add, logits_sub, atol=1e-5), (
        "mode='add' and mode='subtract' should produce different logits"
    )


# ---------------------------------------------------------------------------
# 12. measure_steering_effect - returns correct keys
# ---------------------------------------------------------------------------

def test_measure_steering_effect_keys(tiny_model, steering_vec):
    cfg = SteeringConfig(layer_idx=0, coeff=5.0, mode="add")
    ids = torch.randint(0, 256, (1, 4))
    result = measure_steering_effect(tiny_model, ids, steering_vec, cfg)
    assert set(result.keys()) == {"logit_kl", "logit_mse", "max_logit_diff"}, (
        f"Unexpected keys: {result.keys()}"
    )


# ---------------------------------------------------------------------------
# 13. measure_steering_effect - logit_mse > 0 when coeff is non-zero
# ---------------------------------------------------------------------------

def test_measure_steering_effect_mse_nonzero(tiny_model, steering_vec):
    cfg = SteeringConfig(layer_idx=0, coeff=10.0, mode="add")
    ids = torch.randint(0, 256, (1, 4))
    result = measure_steering_effect(tiny_model, ids, steering_vec, cfg)
    assert result["logit_mse"] > 0.0, (
        f"Expected logit_mse > 0 with non-zero coeff, got {result['logit_mse']}"
    )
