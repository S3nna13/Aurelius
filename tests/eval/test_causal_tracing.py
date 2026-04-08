"""Tests for causal tracing / activation patching (src/eval/causal_tracing.py)."""
from __future__ import annotations

import torch
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.eval.causal_tracing import (
    PatchingResult,
    get_hidden_states,
    patch_and_forward,
    causal_trace,
)


# ---------------------------------------------------------------------------
# Shared tiny model fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model() -> AureliusTransformer:
    config = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    model = AureliusTransformer(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def seq_len() -> int:
    return 4


@pytest.fixture(scope="module")
def clean_ids(seq_len) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 256, (1, seq_len))


@pytest.fixture(scope="module")
def corrupted_ids(seq_len) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 256, (1, seq_len))


@pytest.fixture(scope="module")
def target_token_id() -> int:
    return 42


# ---------------------------------------------------------------------------
# Tests for get_hidden_states
# ---------------------------------------------------------------------------

def test_get_hidden_states_returns_all_layers(tiny_model, clean_ids):
    """Dict must have one entry per layer."""
    hs = get_hidden_states(tiny_model, clean_ids)
    assert len(hs) == tiny_model.config.n_layers


def test_get_hidden_states_shape(tiny_model, clean_ids):
    """Each hidden state must be (1, S, D)."""
    S = clean_ids.shape[1]
    D = tiny_model.config.d_model
    hs = get_hidden_states(tiny_model, clean_ids)
    for layer_idx, tensor in hs.items():
        assert tensor.shape == (1, S, D), (
            f"Layer {layer_idx}: expected (1, {S}, {D}), got {tensor.shape}"
        )


# ---------------------------------------------------------------------------
# Tests for patch_and_forward
# ---------------------------------------------------------------------------

def test_patch_and_forward_returns_float(tiny_model, clean_ids, corrupted_ids, target_token_id):
    """Should return a float probability in [0, 1]."""
    hs = get_hidden_states(tiny_model, clean_ids)
    prob = patch_and_forward(
        model=tiny_model,
        corrupted_ids=corrupted_ids,
        patch_hidden=hs[0],
        layer_idx=0,
        position=0,
        target_token_id=target_token_id,
    )
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# Tests for causal_trace
# ---------------------------------------------------------------------------

def test_causal_trace_returns_results(tiny_model, clean_ids, corrupted_ids, target_token_id):
    """Should return a non-empty list."""
    results = causal_trace(
        model=tiny_model,
        clean_ids=clean_ids,
        corrupted_ids=corrupted_ids,
        target_token_id=target_token_id,
        layers=[0],
        positions=[0],
    )
    assert len(results) > 0


def test_causal_trace_result_fields(tiny_model, clean_ids, corrupted_ids, target_token_id):
    """Each PatchingResult must have valid layer and position values."""
    S = clean_ids.shape[1]
    results = causal_trace(
        model=tiny_model,
        clean_ids=clean_ids,
        corrupted_ids=corrupted_ids,
        target_token_id=target_token_id,
        layers=[0, 1],
        positions=list(range(S)),
    )
    for r in results:
        assert isinstance(r, PatchingResult)
        assert 0 <= r.layer < tiny_model.config.n_layers
        assert 0 <= r.position < S


def test_restoration_score_range(tiny_model, clean_ids, corrupted_ids, target_token_id):
    """All restoration scores must be in [0, 1]."""
    results = causal_trace(
        model=tiny_model,
        clean_ids=clean_ids,
        corrupted_ids=corrupted_ids,
        target_token_id=target_token_id,
        layers=[0, 1],
        positions=[0, 1],
    )
    for r in results:
        score = r.restoration_score
        assert 0.0 <= score <= 1.0, f"Score out of range: {score} for layer={r.layer}, pos={r.position}"


def test_restoration_score_full(tiny_model, clean_ids, target_token_id):
    """Patching clean-into-clean: patched_prob should equal clean_prob."""
    results = causal_trace(
        model=tiny_model,
        clean_ids=clean_ids,
        corrupted_ids=clean_ids,   # same input = no corruption
        target_token_id=target_token_id,
        layers=[0],
        positions=[0],
    )
    r = results[0]
    assert abs(r.patched_prob - r.clean_prob) < 1e-4, (
        f"Expected patched_prob approx clean_prob when patching clean into clean, "
        f"got patched={r.patched_prob:.6f}, clean={r.clean_prob:.6f}"
    )


def test_causal_trace_subset_layers(tiny_model, clean_ids, corrupted_ids, target_token_id):
    """layers=[0] should return only results with layer == 0."""
    results = causal_trace(
        model=tiny_model,
        clean_ids=clean_ids,
        corrupted_ids=corrupted_ids,
        target_token_id=target_token_id,
        layers=[0],
    )
    assert len(results) > 0
    assert all(r.layer == 0 for r in results), (
        f"Expected all results to have layer=0, got layers: {[r.layer for r in results]}"
    )
