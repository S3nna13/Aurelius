"""
tests/interpretability/test_circuit_discovery.py

Tests for src/interpretability/circuit_discovery.py

Uses a tiny AureliusTransformer:
    n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
    d_ff=128, vocab_size=256, max_seq_len=64
"""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.interpretability.circuit_discovery import (
    CircuitConfig,
    CircuitDiscoverer,
    ComponentScore,
    compute_indirect_effect,
    visualize_circuit,
)

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

torch.manual_seed(42)

B = 1
S = 8


def _make_model():
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


def _make_ids():
    return torch.randint(0, TINY_CFG.vocab_size, (B, S))


def _metric_fn(logits):
    return float(logits[:, -1, :].mean().item())


def test_circuit_config_defaults():
    cfg = CircuitConfig()
    assert cfg.threshold == 0.5
    assert cfg.n_patches is None
    assert cfg.normalize_scores is True


def test_get_clean_activations_returns_dict_with_keys():
    model = _make_model()
    discoverer = CircuitDiscoverer(model, _metric_fn)
    clean_ids = _make_ids()
    acts = discoverer.get_clean_activations(clean_ids)
    assert isinstance(acts, dict)
    assert len(acts) > 0


def test_get_clean_activations_captures_attn_and_mlp():
    model = _make_model()
    discoverer = CircuitDiscoverer(model, _metric_fn)
    clean_ids = _make_ids()
    acts = discoverer.get_clean_activations(clean_ids)

    has_attn = any(k.startswith("attn_layer_") for k in acts)
    has_mlp = any(k.startswith("mlp_layer_") for k in acts)

    assert has_attn, "No attention activation keys found"
    assert has_mlp, "No MLP activation keys found"

    n_layers = TINY_CFG.n_layers
    n_heads = TINY_CFG.n_heads
    for li in range(n_layers):
        assert f"mlp_layer_{li}" in acts, f"Missing mlp_layer_{li}"
        for hi in range(n_heads):
            assert f"attn_layer_{li}_head_{hi}" in acts, f"Missing attn_layer_{li}_head_{hi}"


def test_patch_activation_runs_without_error():
    model = _make_model()
    discoverer = CircuitDiscoverer(model, _metric_fn)
    clean_ids = _make_ids()
    corrupted_ids = _make_ids()
    acts = discoverer.get_clean_activations(clean_ids)

    metric_val = discoverer.patch_activation(corrupted_ids, acts, "attn_layer_0_head_0")
    assert isinstance(metric_val, float)

    metric_val2 = discoverer.patch_activation(corrupted_ids, acts, "mlp_layer_0")
    assert isinstance(metric_val2, float)


def test_compute_patching_scores_returns_list_of_component_scores():
    model = _make_model()
    discoverer = CircuitDiscoverer(model, _metric_fn)
    clean_ids = _make_ids()
    corrupted_ids = _make_ids()
    scores = discoverer.compute_patching_scores(clean_ids, corrupted_ids)

    assert isinstance(scores, list)
    assert len(scores) > 0
    for cs in scores:
        assert isinstance(cs, ComponentScore)


def test_compute_patching_scores_normalised():
    model = _make_model()
    discoverer = CircuitDiscoverer(model, _metric_fn)
    clean_ids = _make_ids()
    corrupted_ids = _make_ids()
    scores = discoverer.compute_patching_scores(clean_ids, corrupted_ids)

    for cs in scores:
        assert -1.0 <= cs.score <= 2.0, f"Score {cs.score} for {cs.description} out of range"


def test_discover_circuit_threshold_zero_returns_all():
    model = _make_model()
    discoverer = CircuitDiscoverer(model, _metric_fn)
    clean_ids = _make_ids()
    corrupted_ids = _make_ids()

    all_scores = discoverer.compute_patching_scores(clean_ids, corrupted_ids)
    circuit = discoverer.discover_circuit(clean_ids, corrupted_ids, threshold=0.0)

    expected = [s for s in all_scores if s.score >= 0.0]
    assert len(circuit) == len(expected)


def test_discover_circuit_threshold_above_max_returns_empty():
    model = _make_model()
    discoverer = CircuitDiscoverer(model, _metric_fn)
    clean_ids = _make_ids()
    corrupted_ids = _make_ids()
    circuit = discoverer.discover_circuit(clean_ids, corrupted_ids, threshold=1.1)
    assert circuit == []


def test_visualize_circuit_returns_dict_with_by_layer():
    model = _make_model()
    discoverer = CircuitDiscoverer(model, _metric_fn)
    clean_ids = _make_ids()
    corrupted_ids = _make_ids()
    scores = discoverer.compute_patching_scores(clean_ids, corrupted_ids)
    viz = visualize_circuit(scores)

    assert isinstance(viz, dict)
    assert "by_layer" in viz


def test_visualize_circuit_top_components_sorted_descending():
    model = _make_model()
    discoverer = CircuitDiscoverer(model, _metric_fn)
    clean_ids = _make_ids()
    corrupted_ids = _make_ids()
    scores = discoverer.compute_patching_scores(clean_ids, corrupted_ids)
    viz = visualize_circuit(scores)

    assert "top_components" in viz
    top = viz["top_components"]
    for i in range(len(top) - 1):
        assert top[i].score >= top[i + 1].score


def test_component_score_has_all_required_fields():
    cs = ComponentScore(
        component_type="attention_head",
        layer=0,
        head=1,
        score=0.75,
        description="Layer 0 Attention Head 1",
    )
    assert hasattr(cs, "component_type")
    assert hasattr(cs, "layer")
    assert hasattr(cs, "head")
    assert hasattr(cs, "score")
    assert hasattr(cs, "description")

    assert cs.component_type == "attention_head"
    assert cs.layer == 0
    assert cs.head == 1
    assert cs.score == 0.75
    assert cs.description == "Layer 0 Attention Head 1"

    mlp = ComponentScore(
        component_type="mlp",
        layer=1,
        head=-1,
        score=0.3,
        description="Layer 1 MLP",
    )
    assert mlp.head == -1
    assert mlp.component_type == "mlp"
