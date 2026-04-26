"""
Tests for src/interpretability/activation_addition.py

Tiny model configuration used throughout:
    AureliusConfig(
        n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
        head_dim=16, d_ff=128, vocab_size=256, max_seq_len=64
    )
"""

from __future__ import annotations

import torch

from src.interpretability.activation_addition import (
    ActivationAddition,
    SteeringVector,
    SteeringVectorBank,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

torch.manual_seed(42)

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
D_MODEL = TINY_CFG.d_model
N_LAYERS = TINY_CFG.n_layers


def _make_model() -> AureliusTransformer:
    torch.manual_seed(0)
    return AureliusTransformer(TINY_CFG)


def _make_ids(batch: int = 2, seq: int = 8) -> torch.Tensor:
    return torch.randint(0, TINY_CFG.vocab_size, (batch, seq))


# ---------------------------------------------------------------------------
# Test 1 — ActivationAddition instantiates without error
# ---------------------------------------------------------------------------


def test_instantiation():
    model = _make_model()
    aa = ActivationAddition(model)
    assert aa is not None
    # Model's layers should be accessible and match the config
    assert len(aa.layers) == N_LAYERS
    # layers_to_hook should default to all layers
    assert aa.layers_to_hook == list(range(N_LAYERS))
    aa.remove_hooks()


# ---------------------------------------------------------------------------
# Test 2 — get_activations returns correct shape (batch, seq, d_model)
# ---------------------------------------------------------------------------


def test_get_activations_shape():
    model = _make_model()
    aa = ActivationAddition(model)

    batch, seq = 2, 8
    input_ids = _make_ids(batch, seq)
    acts = aa.get_activations(input_ids, layer_idx=0)

    assert acts.shape == (batch, seq, D_MODEL), (
        f"Expected {(batch, seq, D_MODEL)}, got {acts.shape}"
    )
    aa.remove_hooks()


# ---------------------------------------------------------------------------
# Test 3 — extract_steering_vector returns SteeringVector with correct d_model
# ---------------------------------------------------------------------------


def test_extract_steering_vector_shape():
    model = _make_model()
    aa = ActivationAddition(model)

    pos_ids = _make_ids(2, 8)
    neg_ids = _make_ids(2, 8)
    sv = aa.extract_steering_vector(pos_ids, neg_ids, layer_idx=0, label="test")

    assert isinstance(sv, SteeringVector)
    assert sv.direction.shape == (D_MODEL,), f"Expected ({D_MODEL},), got {sv.direction.shape}"
    assert sv.layer_idx == 0
    aa.remove_hooks()


# ---------------------------------------------------------------------------
# Test 4 — Steering vector is normalised (unit norm)
# ---------------------------------------------------------------------------


def test_steering_vector_unit_norm():
    model = _make_model()
    aa = ActivationAddition(model)

    pos_ids = _make_ids(2, 8)
    neg_ids = _make_ids(2, 8)
    sv = aa.extract_steering_vector(pos_ids, neg_ids, layer_idx=0)

    norm = sv.direction.norm().item()
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"
    aa.remove_hooks()


# ---------------------------------------------------------------------------
# Test 5 — apply_steering context manager modifies activations
# ---------------------------------------------------------------------------


def test_apply_steering_modifies_activations():
    model = _make_model()
    aa = ActivationAddition(model)

    input_ids = _make_ids(1, 6)

    # Baseline activations (no steering)
    baseline = aa.get_activations(input_ids, layer_idx=0).clone()

    # Build a steering vector with a large coefficient so the effect is visible
    direction = torch.ones(D_MODEL) / (D_MODEL**0.5)
    sv = SteeringVector(direction=direction, layer_idx=0, coefficient=100.0)

    # Activations with steering applied
    with aa.apply_steering(sv):
        steered = aa.get_activations(input_ids, layer_idx=0).clone()

    assert not torch.allclose(baseline, steered, atol=1e-4), (
        "Activations should differ when steering is applied"
    )
    aa.remove_hooks()


# ---------------------------------------------------------------------------
# Test 6 — After context manager exits, activations return to normal
# ---------------------------------------------------------------------------


def test_apply_steering_reverts_after_context():
    model = _make_model()
    aa = ActivationAddition(model)

    input_ids = _make_ids(1, 6)

    baseline = aa.get_activations(input_ids, layer_idx=0).clone()

    direction = torch.ones(D_MODEL) / (D_MODEL**0.5)
    sv = SteeringVector(direction=direction, layer_idx=0, coefficient=100.0)

    with aa.apply_steering(sv):
        pass  # enter and immediately exit

    after = aa.get_activations(input_ids, layer_idx=0).clone()

    assert torch.allclose(baseline, after, atol=1e-5), (
        "Activations should revert to baseline after context manager exits"
    )
    aa.remove_hooks()


# ---------------------------------------------------------------------------
# Test 7 — generate_steered returns a longer sequence than the input
# ---------------------------------------------------------------------------


def test_generate_steered_returns_longer_sequence():
    model = _make_model()
    aa = ActivationAddition(model)

    prompt_len = 5
    max_new = 10
    input_ids = _make_ids(1, prompt_len)

    direction = torch.randn(D_MODEL)
    direction = direction / direction.norm()
    sv = SteeringVector(direction=direction, layer_idx=0, coefficient=1.0)

    output = aa.generate_steered(input_ids, [sv], max_new_tokens=max_new)

    assert output.shape[1] == prompt_len + max_new, (
        f"Expected seq_len {prompt_len + max_new}, got {output.shape[1]}"
    )
    aa.remove_hooks()


# ---------------------------------------------------------------------------
# Test 8 — SteeringVectorBank.add and get work correctly
# ---------------------------------------------------------------------------


def test_steering_vector_bank_add_get():
    bank = SteeringVectorBank()
    direction = torch.randn(D_MODEL)
    direction = direction / direction.norm()
    sv = SteeringVector(direction=direction, layer_idx=1, coefficient=2.0, label="test")

    bank.add(sv, "test_vector")
    retrieved = bank.get("test_vector")

    assert retrieved is sv
    assert torch.allclose(retrieved.direction, sv.direction)
    assert retrieved.layer_idx == 1
    assert retrieved.coefficient == 2.0


# ---------------------------------------------------------------------------
# Test 9 — compose returns a vector of the same shape as inputs
# ---------------------------------------------------------------------------


def test_steering_vector_bank_compose_shape():
    bank = SteeringVectorBank()

    for i, name in enumerate(["a", "b", "c"]):
        d = torch.randn(D_MODEL)
        d = d / d.norm()
        bank.add(SteeringVector(direction=d, layer_idx=0, coefficient=1.0, label=name), name)

    composed = bank.compose(["a", "b", "c"])

    assert composed.direction.shape == (D_MODEL,), (
        f"Expected ({D_MODEL},), got {composed.direction.shape}"
    )


# ---------------------------------------------------------------------------
# Test 10 — remove_hooks cleans up; subsequent forward passes raise no error
# ---------------------------------------------------------------------------


def test_remove_hooks_no_error_on_forward():
    model = _make_model()
    aa = ActivationAddition(model)
    aa.remove_hooks()

    # After removing hooks a plain forward pass should still work
    input_ids = _make_ids(1, 4)
    with torch.no_grad():
        loss, logits, _ = model(input_ids)
    assert logits.shape == (1, 4, TINY_CFG.vocab_size)


# ---------------------------------------------------------------------------
# Test 11 — Different layer indices give different steering vectors
# ---------------------------------------------------------------------------


def test_different_layers_give_different_vectors():
    model = _make_model()
    aa = ActivationAddition(model, layers_to_hook=[0, 1])

    pos_ids = _make_ids(2, 8)
    neg_ids = _make_ids(2, 8)

    sv0 = aa.extract_steering_vector(pos_ids, neg_ids, layer_idx=0)
    sv1 = aa.extract_steering_vector(pos_ids, neg_ids, layer_idx=1)

    # Directions should differ because residual stream grows deeper in the network
    assert not torch.allclose(sv0.direction, sv1.direction, atol=1e-4), (
        "Steering vectors from different layers should differ"
    )
    aa.remove_hooks()
