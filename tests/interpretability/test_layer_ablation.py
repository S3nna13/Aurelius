"""
tests/interpretability/test_layer_ablation.py

Unit tests for src/interpretability/layer_ablation.py.

Uses a tiny AureliusTransformer so tests run fast on CPU.
"""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.interpretability.layer_ablation import (
    AblationConfig,
    AblationResult,
    AblationType,
    LayerAblator,
)

# ---------------------------------------------------------------------------
# Shared tiny model config
# ---------------------------------------------------------------------------

TINY_CONFIG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

torch.manual_seed(42)

BATCH = 1
SEQ   = 8


def _make_model() -> AureliusTransformer:
    torch.manual_seed(0)
    return AureliusTransformer(TINY_CONFIG).eval()


def _make_input() -> torch.Tensor:
    return torch.randint(0, TINY_CONFIG.vocab_size, (BATCH, SEQ))


def _metric_fn(logits: torch.Tensor) -> float:
    """Mean log-probability of the last token -- higher is better."""
    last = logits[:, -1, :]          # (B, vocab)
    log_probs = torch.log_softmax(last, dim=-1)
    return log_probs.mean().item()


# ---------------------------------------------------------------------------
# Test 1: AblationConfig defaults
# ---------------------------------------------------------------------------

def test_ablation_config_defaults():
    cfg = AblationConfig()
    assert cfg.ablation_type == AblationType.ZERO or cfg.ablation_type == "zero"
    assert cfg.noise_std == 0.1
    assert cfg.n_seeds == 3


# ---------------------------------------------------------------------------
# Test 2: AblationType has all required values
# ---------------------------------------------------------------------------

def test_ablation_type_values():
    assert AblationType.ZERO   == "zero"
    assert AblationType.MEAN   == "mean"
    assert AblationType.NOISE  == "noise"
    assert AblationType.FREEZE == "freeze"


# ---------------------------------------------------------------------------
# Test 3: AblationResult has all required fields
# ---------------------------------------------------------------------------

def test_ablation_result_fields():
    r = AblationResult(
        component="layer_0",
        ablation_type="zero",
        metric_before=-1.0,
        metric_after=-2.0,
        delta=-1.0,
        relative_impact=1.0,
    )
    assert hasattr(r, "component")
    assert hasattr(r, "ablation_type")
    assert hasattr(r, "metric_before")
    assert hasattr(r, "metric_after")
    assert hasattr(r, "delta")
    assert hasattr(r, "relative_impact")


# ---------------------------------------------------------------------------
# Test 4: ablate_layer context manager runs without error
# ---------------------------------------------------------------------------

def test_ablate_layer_context_manager_runs():
    model = _make_model()
    ablator = LayerAblator(model, _metric_fn)
    ids = _make_input()

    with ablator.ablate_layer(0, AblationType.ZERO, ids) as logits:
        assert logits is not None
        assert logits.shape == (BATCH, SEQ, TINY_CONFIG.vocab_size)


# ---------------------------------------------------------------------------
# Test 5: Zero ablation produces different output than no ablation
# ---------------------------------------------------------------------------

def test_zero_ablation_changes_output():
    model = _make_model()
    ablator = LayerAblator(model, _metric_fn)
    ids = _make_input()

    # Baseline (no ablation)
    baseline_logits = ablator._forward(ids)

    # Zero ablation on layer 0
    with ablator.ablate_layer(0, AblationType.ZERO, ids) as ablated_logits:
        assert not torch.allclose(baseline_logits, ablated_logits), (
            "Zero ablation should change the model output"
        )


# ---------------------------------------------------------------------------
# Test 6: run_full_ablation returns one AblationResult per layer
# ---------------------------------------------------------------------------

def test_run_full_ablation_result_count():
    model = _make_model()
    ablator = LayerAblator(model, _metric_fn)
    ids = _make_input()

    results = ablator.run_full_ablation(ids, AblationType.ZERO)

    assert len(results) == TINY_CONFIG.n_layers, (
        f"Expected {TINY_CONFIG.n_layers} results, got {len(results)}"
    )
    for r in results:
        assert isinstance(r, AblationResult)


# ---------------------------------------------------------------------------
# Test 7: run_full_ablation results have valid delta values
# ---------------------------------------------------------------------------

def test_run_full_ablation_delta_values():
    model = _make_model()
    ablator = LayerAblator(model, _metric_fn)
    ids = _make_input()

    results = ablator.run_full_ablation(ids, AblationType.ZERO)

    for r in results:
        expected_delta = r.metric_after - r.metric_before
        assert abs(r.delta - expected_delta) < 1e-6, (
            f"delta mismatch for {r.component}: "
            f"stored {r.delta:.6f} vs computed {expected_delta:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 8: compute_layer_importance returns tensor of correct length
# ---------------------------------------------------------------------------

def test_compute_layer_importance_shape():
    model = _make_model()
    ablator = LayerAblator(model, _metric_fn)
    ids = _make_input()

    results = ablator.run_full_ablation(ids, AblationType.ZERO)
    importance = ablator.compute_layer_importance(results)

    assert isinstance(importance, torch.Tensor)
    assert importance.shape == (TINY_CONFIG.n_layers,), (
        f"Expected shape ({TINY_CONFIG.n_layers},), got {importance.shape}"
    )


# ---------------------------------------------------------------------------
# Test 9: All importance values >= 0
# ---------------------------------------------------------------------------

def test_importance_values_non_negative():
    model = _make_model()
    ablator = LayerAblator(model, _metric_fn)
    ids = _make_input()

    results = ablator.run_full_ablation(ids, AblationType.ZERO)
    importance = ablator.compute_layer_importance(results)

    assert (importance >= 0).all(), (
        f"All importance values should be >= 0, got {importance}"
    )


# ---------------------------------------------------------------------------
# Test 10: get_redundant_layers with large threshold returns all layers
# ---------------------------------------------------------------------------

def test_get_redundant_layers_threshold_returns_all():
    model = _make_model()
    ablator = LayerAblator(model, _metric_fn)
    ids = _make_input()

    results = ablator.run_full_ablation(ids, AblationType.ZERO)

    # threshold=2.0 means relative_impact < 2.0 is redundant.
    # A tiny random model will almost certainly have all layers below 2.0.
    redundant = ablator.get_redundant_layers(results, threshold=2.0)

    all_layer_indices = list(range(TINY_CONFIG.n_layers))
    assert sorted(redundant) == all_layer_indices, (
        f"Expected all layers {all_layer_indices} to be redundant, got {redundant}"
    )


# ---------------------------------------------------------------------------
# Test 11: AblationResult.delta == metric_after - metric_before
# ---------------------------------------------------------------------------

def test_ablation_result_delta_formula():
    metric_before = -3.5
    metric_after  = -4.2
    delta = metric_after - metric_before

    r = AblationResult(
        component="layer_0",
        ablation_type="zero",
        metric_before=metric_before,
        metric_after=metric_after,
        delta=delta,
        relative_impact=abs(delta) / abs(metric_before),
    )

    assert abs(r.delta - (r.metric_after - r.metric_before)) < 1e-9, (
        "AblationResult.delta must equal metric_after - metric_before"
    )
