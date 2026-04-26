"""
Tests for src/interpretability/representation_engineering.py

Tiny model configuration used throughout:
    AureliusConfig(
        n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
        head_dim=16, d_ff=128, vocab_size=256, max_seq_len=64
    )
"""

from __future__ import annotations

import pytest
import torch

from src.interpretability.representation_engineering import (
    RepEngConfig,
    RepresentationExtractor,
    SteeringVectorBank,
    apply_steering_hook,
    extract_concept_direction,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

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
# Test 1 — RepresentationExtractor extracts shape (B, T, d_model)
# ---------------------------------------------------------------------------


def test_extractor_shape():
    torch.manual_seed(1)
    model = _make_model()
    extractor = RepresentationExtractor(model, layer_idx=0)

    batch, seq = 3, 10
    input_ids = _make_ids(batch, seq)
    hiddens = extractor.extract(input_ids)

    assert hiddens.shape == (batch, seq, D_MODEL), (
        f"Expected shape {(batch, seq, D_MODEL)}, got {hiddens.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Different inputs produce different representations
# ---------------------------------------------------------------------------


def test_extractor_different_inputs_differ():
    torch.manual_seed(2)
    model = _make_model()
    extractor = RepresentationExtractor(model, layer_idx=0)

    ids_a = _make_ids(2, 8)
    # Construct ids_b that is guaranteed to differ from ids_a
    ids_b = (ids_a + 1) % TINY_CFG.vocab_size

    hiddens_a = extractor.extract(ids_a)
    hiddens_b = extractor.extract(ids_b)

    assert not torch.allclose(hiddens_a, hiddens_b, atol=1e-4), (
        "Different input tokens should produce different representations"
    )


# ---------------------------------------------------------------------------
# Test 3 — extract_concept_direction returns (d_model,) tensor
# ---------------------------------------------------------------------------


def test_extract_concept_direction_shape():
    torch.manual_seed(3)
    model = _make_model()

    pos_ids = _make_ids(4, 8)
    neg_ids = _make_ids(4, 8)

    direction = extract_concept_direction(model, pos_ids, neg_ids, layer_idx=0)

    assert direction.shape == (D_MODEL,), f"Expected shape ({D_MODEL},), got {direction.shape}"


# ---------------------------------------------------------------------------
# Test 4 — extract_concept_direction with normalize=True produces unit vector
# ---------------------------------------------------------------------------


def test_extract_concept_direction_unit_norm():
    torch.manual_seed(4)
    model = _make_model()

    pos_ids = _make_ids(4, 8)
    neg_ids = _make_ids(4, 8)

    direction = extract_concept_direction(model, pos_ids, neg_ids, layer_idx=0, normalize=True)

    norm = direction.norm().item()
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


# ---------------------------------------------------------------------------
# Test 5 — SteeringVectorBank add and get roundtrip
# ---------------------------------------------------------------------------


def test_bank_add_get_roundtrip():
    torch.manual_seed(5)
    bank = SteeringVectorBank()

    vec = torch.randn(D_MODEL)
    bank.add("test", vec)

    retrieved = bank.get("test")
    assert torch.allclose(retrieved, vec), "Retrieved vector should equal the stored vector"


# ---------------------------------------------------------------------------
# Test 6 — SteeringVectorBank.get missing key raises KeyError
# ---------------------------------------------------------------------------


def test_bank_get_missing_key_raises():
    bank = SteeringVectorBank()

    with pytest.raises(KeyError):
        bank.get("nonexistent")


# ---------------------------------------------------------------------------
# Test 7 — SteeringVectorBank.list_names returns correct names
# ---------------------------------------------------------------------------


def test_bank_list_names():
    torch.manual_seed(7)
    bank = SteeringVectorBank()
    names = ["alpha", "beta", "gamma"]

    for name in names:
        bank.add(name, torch.randn(D_MODEL))

    assert sorted(bank.list_names()) == sorted(names), (
        f"Expected {sorted(names)}, got {sorted(bank.list_names())}"
    )


# ---------------------------------------------------------------------------
# Test 8 — SteeringVectorBank.remove removes the entry
# ---------------------------------------------------------------------------


def test_bank_remove():
    torch.manual_seed(8)
    bank = SteeringVectorBank()

    bank.add("to_remove", torch.randn(D_MODEL))
    bank.add("to_keep", torch.randn(D_MODEL))

    bank.remove("to_remove")

    assert "to_remove" not in bank.list_names(), "'to_remove' should be gone"
    assert "to_keep" in bank.list_names(), "'to_keep' should still be present"

    # Removing again should raise KeyError
    with pytest.raises(KeyError):
        bank.remove("to_remove")


# ---------------------------------------------------------------------------
# Test 9 — SteeringVectorBank.compose returns (d_model,) tensor
# ---------------------------------------------------------------------------


def test_bank_compose_shape():
    torch.manual_seed(9)
    bank = SteeringVectorBank()

    for name in ["x", "y", "z"]:
        v = torch.randn(D_MODEL)
        v = v / v.norm()
        bank.add(name, v)

    composed = bank.compose(["x", "y", "z"])

    assert composed.shape == (D_MODEL,), f"Expected shape ({D_MODEL},), got {composed.shape}"


# ---------------------------------------------------------------------------
# Test 10 — SteeringVectorBank.project_out removes the component (dot ≈ 0)
# ---------------------------------------------------------------------------


def test_bank_project_out_removes_component():
    torch.manual_seed(10)
    bank = SteeringVectorBank()

    # Store a unit-norm direction
    direction = torch.randn(D_MODEL)
    direction = direction / direction.norm()
    bank.add("dir", direction)

    # Create a vector with a known component along the direction
    orthogonal = torch.randn(D_MODEL)
    orthogonal = orthogonal - direction * orthogonal.dot(direction)  # make it orthogonal
    vec = orthogonal + 5.0 * direction  # deliberate component along direction

    projected = bank.project_out(vec, "dir")

    dot = projected.dot(direction).abs().item()
    assert dot < 1e-4, (
        f"After projection the dot product with the direction should be ~0, got {dot}"
    )


# ---------------------------------------------------------------------------
# Test 11 — SteeringVectorBank.interpolate: alpha=0 → vec_b, alpha=1 → vec_a
# ---------------------------------------------------------------------------


def test_bank_interpolate_boundary():
    torch.manual_seed(11)
    bank = SteeringVectorBank()

    vec_a = torch.randn(D_MODEL)
    vec_b = torch.randn(D_MODEL)
    bank.add("a", vec_a)
    bank.add("b", vec_b)

    result_alpha1 = bank.interpolate("a", "b", alpha=1.0)
    result_alpha0 = bank.interpolate("a", "b", alpha=0.0)

    assert torch.allclose(result_alpha1, vec_a, atol=1e-6), "alpha=1 should return vec_a"
    assert torch.allclose(result_alpha0, vec_b, atol=1e-6), "alpha=0 should return vec_b"

    # Also check a midpoint
    result_mid = bank.interpolate("a", "b", alpha=0.5)
    expected_mid = 0.5 * vec_a + 0.5 * vec_b
    assert torch.allclose(result_mid, expected_mid, atol=1e-6), (
        "alpha=0.5 should return the midpoint"
    )


# ---------------------------------------------------------------------------
# Test 12 — apply_steering_hook changes model output compared to no hook
# ---------------------------------------------------------------------------


def test_apply_steering_hook_changes_output():
    torch.manual_seed(12)
    model = _make_model()

    input_ids = _make_ids(2, 8)

    # Baseline forward pass
    with torch.no_grad():
        _, logits_baseline, _ = model(input_ids)

    # Apply a large steering vector at layer 0
    steering_vec = torch.ones(D_MODEL)
    handle = apply_steering_hook(model, steering_vec, layer_idx=0, scale=10.0)
    try:
        with torch.no_grad():
            _, logits_steered, _ = model(input_ids)
    finally:
        handle.remove()

    assert not torch.allclose(logits_baseline, logits_steered, atol=1e-4), (
        "Applying a steering hook should change the model output"
    )

    # After hook removal, output should return to baseline
    with torch.no_grad():
        _, logits_after_remove, _ = model(input_ids)

    assert torch.allclose(logits_baseline, logits_after_remove, atol=1e-5), (
        "After removing the hook, output should return to the baseline"
    )


# ---------------------------------------------------------------------------
# Test 13 — RepresentationExtractor works with negative layer_idx (-1)
# ---------------------------------------------------------------------------


def test_extractor_negative_layer_idx():
    torch.manual_seed(13)
    model = _make_model()

    extractor_last = RepresentationExtractor(model, layer_idx=-1)
    extractor_explicit = RepresentationExtractor(model, layer_idx=N_LAYERS - 1)

    input_ids = _make_ids(2, 6)
    hiddens_last = extractor_last.extract(input_ids)
    hiddens_explicit = extractor_explicit.extract(input_ids)

    assert torch.allclose(hiddens_last, hiddens_explicit, atol=1e-6), (
        "layer_idx=-1 should produce the same result as layer_idx=N_LAYERS-1"
    )


# ---------------------------------------------------------------------------
# Test 14 — RepEngConfig dataclass has expected defaults
# ---------------------------------------------------------------------------


def test_rep_eng_config_defaults():
    cfg = RepEngConfig()
    assert cfg.layer_idx == -1
    assert cfg.batch_size == 8
    assert cfg.n_directions == 1
    assert cfg.normalize is True


# ---------------------------------------------------------------------------
# Test 15 — extract_concept_direction with normalize=False does NOT normalize
# ---------------------------------------------------------------------------


def test_extract_concept_direction_unnormalized():
    torch.manual_seed(15)
    model = _make_model()

    pos_ids = _make_ids(4, 8)
    neg_ids = _make_ids(4, 8)

    direction_norm = extract_concept_direction(model, pos_ids, neg_ids, layer_idx=0, normalize=True)
    direction_raw = extract_concept_direction(model, pos_ids, neg_ids, layer_idx=0, normalize=False)

    # The unnormalized version should have the same orientation as normalized
    # (dot product should be close to its norm, i.e. they point the same way)
    # but a different magnitude (unless the raw norm happened to be 1.0)
    raw_norm = direction_raw.norm().item()
    # If the raw norm is not 1, the two tensors should differ
    if abs(raw_norm - 1.0) > 1e-4:
        assert not torch.allclose(direction_norm, direction_raw, atol=1e-4), (
            "Normalized and unnormalized directions should differ in magnitude"
        )
