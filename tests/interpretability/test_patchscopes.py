"""
Tests for src/interpretability/patchscopes.py

Tiny model configuration used throughout:
    n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
    head_dim=16, d_ff=128, vocab_size=256, max_seq_len=64
"""

from __future__ import annotations

import pytest
import torch

from src.interpretability.patchscopes import Patchscope, PatchscopeConfig
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Tiny test configuration
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

B = 2  # batch size
T = 8  # sequence length


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model() -> AureliusTransformer:
    torch.manual_seed(0)
    m = AureliusTransformer(TINY_CFG)
    m.eval()
    return m


@pytest.fixture(scope="module")
def source_ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, TINY_CFG.vocab_size, (B, T))


@pytest.fixture(scope="module")
def target_ids() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randint(0, TINY_CFG.vocab_size, (B, T))


def _make_scope(
    model: AureliusTransformer,
    source_layer: int = -1,
    source_position: int = -1,
    target_layer: int = 0,
    target_position: int = -1,
) -> Patchscope:
    cfg = PatchscopeConfig(
        source_layer=source_layer,
        source_position=source_position,
        target_layer=target_layer,
        target_position=target_position,
    )
    return Patchscope(model, cfg)


# ---------------------------------------------------------------------------
# Test 1: extract() output shape is (B, d_model)
# ---------------------------------------------------------------------------


def test_extract_shape(model, source_ids):
    scope = _make_scope(model)
    hidden = scope.extract(source_ids)
    assert hidden.shape == (B, TINY_CFG.d_model), (
        f"Expected ({B}, {TINY_CFG.d_model}), got {hidden.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2: inject_and_decode() output shape is (B, T, vocab_size)
# ---------------------------------------------------------------------------


def test_inject_and_decode_shape(model, source_ids, target_ids):
    scope = _make_scope(model)
    hidden = scope.extract(source_ids)
    logits = scope.inject_and_decode(target_ids, hidden)
    assert logits.shape == (B, T, TINY_CFG.vocab_size), (
        f"Expected ({B}, {T}, {TINY_CFG.vocab_size}), got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3: injection changes output versus unmodified forward
# ---------------------------------------------------------------------------


def test_injection_changes_output(model, source_ids, target_ids):
    scope = _make_scope(model)
    hidden = scope.extract(source_ids)

    # Run the patched forward.
    patched_logits = scope.inject_and_decode(target_ids, hidden)

    # Run the clean (unmodified) forward.
    with torch.no_grad():
        _, clean_logits, _ = model(target_ids)

    # At least the injected position should differ.
    assert not torch.allclose(patched_logits, clean_logits, atol=1e-6), (
        "Patched logits should differ from unmodified forward logits."
    )


# ---------------------------------------------------------------------------
# Test 4: run() matches extract() + inject_and_decode()
# ---------------------------------------------------------------------------


def test_run_matches_extract_then_inject(model, source_ids, target_ids):
    scope = _make_scope(model)

    hidden = scope.extract(source_ids)
    expected = scope.inject_and_decode(target_ids, hidden)
    result = scope.run(source_ids, target_ids)

    assert torch.allclose(result, expected, atol=1e-6), (
        "run() output does not match extract() + inject_and_decode()."
    )


# ---------------------------------------------------------------------------
# Test 5: determinism under torch.manual_seed
# ---------------------------------------------------------------------------


def test_determinism(model, source_ids, target_ids):
    scope = _make_scope(model)

    torch.manual_seed(42)
    out1 = scope.run(source_ids, target_ids)

    torch.manual_seed(42)
    out2 = scope.run(source_ids, target_ids)

    assert torch.allclose(out1, out2, atol=1e-7), (
        "Patchscope results are not deterministic under the same seed."
    )


# ---------------------------------------------------------------------------
# Test 6: no NaN/Inf in extracted hidden states
# ---------------------------------------------------------------------------


def test_extract_no_nan_inf(model, source_ids):
    scope = _make_scope(model)
    hidden = scope.extract(source_ids)
    assert torch.isfinite(hidden).all(), "Extracted hidden state contains NaN or Inf."


# ---------------------------------------------------------------------------
# Test 7: no NaN/Inf in output logits
# ---------------------------------------------------------------------------


def test_inject_no_nan_inf(model, source_ids, target_ids):
    scope = _make_scope(model)
    logits = scope.run(source_ids, target_ids)
    assert torch.isfinite(logits).all(), "Output logits contain NaN or Inf."


# ---------------------------------------------------------------------------
# Test 8: different source_layer -> different extracted state
# ---------------------------------------------------------------------------


def test_different_source_layers_differ(model, source_ids):
    scope0 = _make_scope(model, source_layer=0)
    scope1 = _make_scope(model, source_layer=1)

    h0 = scope0.extract(source_ids)
    h1 = scope1.extract(source_ids)

    assert not torch.allclose(h0, h1, atol=1e-6), (
        "Hidden states from different layers should differ."
    )


# ---------------------------------------------------------------------------
# Test 9: different source_position -> different extracted state
# ---------------------------------------------------------------------------


def test_different_source_positions_differ(model, source_ids):
    scope_pos0 = _make_scope(model, source_position=0)
    scope_pos1 = _make_scope(model, source_position=1)

    h0 = scope_pos0.extract(source_ids)
    h1 = scope_pos1.extract(source_ids)

    assert not torch.allclose(h0, h1, atol=1e-6), (
        "Hidden states from different token positions should differ."
    )


# ---------------------------------------------------------------------------
# Test 10: batch size > 1 works
# ---------------------------------------------------------------------------


def test_batch_size_gt_1(model):
    torch.manual_seed(7)
    big_batch = 4
    src = torch.randint(0, TINY_CFG.vocab_size, (big_batch, T))
    tgt = torch.randint(0, TINY_CFG.vocab_size, (big_batch, T))

    scope = _make_scope(model)
    logits = scope.run(src, tgt)

    assert logits.shape == (big_batch, T, TINY_CFG.vocab_size), (
        f"Expected ({big_batch}, {T}, {TINY_CFG.vocab_size}), got {logits.shape}"
    )
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Test 11: source_layer=0 (first layer) works
# ---------------------------------------------------------------------------


def test_source_layer_zero(model, source_ids, target_ids):
    scope = _make_scope(model, source_layer=0)
    hidden = scope.extract(source_ids)
    assert hidden.shape == (B, TINY_CFG.d_model)
    assert torch.isfinite(hidden).all()

    logits = scope.inject_and_decode(target_ids, hidden)
    assert logits.shape == (B, T, TINY_CFG.vocab_size)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Test 12: inject zero vector -> output differs from normal forward
# ---------------------------------------------------------------------------


def test_inject_zero_vector_differs_from_clean(model, target_ids):
    scope = _make_scope(model)

    # Build an all-zero hidden vector.
    zero_hidden = torch.zeros(B, TINY_CFG.d_model)
    patched_logits = scope.inject_and_decode(target_ids, zero_hidden)

    with torch.no_grad():
        _, clean_logits, _ = model(target_ids)

    assert not torch.allclose(patched_logits, clean_logits, atol=1e-6), (
        "Injecting a zero vector should change the output logits."
    )
