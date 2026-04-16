"""
Tests for src/interpretability/concept_bottleneck.py

Tiny config: d_model=64, n_concepts=8, n_classes=3, batch_size=2, seq_len=4.
All tests use torch.manual_seed for determinism.
"""

from __future__ import annotations

import torch
import pytest

from src.interpretability.concept_bottleneck import (
    CBMConfig,
    ConceptLayer,
    ConceptBottleneckModel,
    concept_alignment_score,
    concept_intervention,
    extract_concept_importance,
)

# ---------------------------------------------------------------------------
# Shared tiny constants
# ---------------------------------------------------------------------------

D_MODEL = 64
N_CONCEPTS = 8
N_CLASSES = 3
B = 2       # batch size
T = 4       # sequence length


def _make_config(activation: str = "sigmoid") -> CBMConfig:
    return CBMConfig(
        d_model=D_MODEL,
        n_concepts=N_CONCEPTS,
        n_classes=N_CLASSES,
        dropout=0.0,  # disable dropout for deterministic tests
        concept_activation=activation,
    )


def _make_hidden(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 1. ConceptLayer output shape (B, T, n_concepts)
# ---------------------------------------------------------------------------

def test_concept_layer_output_shape():
    torch.manual_seed(0)
    cfg = _make_config()
    layer = ConceptLayer(cfg)
    layer.eval()
    hidden = _make_hidden()
    with torch.no_grad():
        out = layer(hidden)
    assert out.shape == (B, T, N_CONCEPTS), (
        f"Expected ({B}, {T}, {N_CONCEPTS}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 2. ConceptLayer values in [0, 1] when sigmoid activation
# ---------------------------------------------------------------------------

def test_concept_layer_sigmoid_range():
    torch.manual_seed(1)
    cfg = _make_config(activation="sigmoid")
    layer = ConceptLayer(cfg)
    layer.eval()
    hidden = _make_hidden(seed=1)
    with torch.no_grad():
        out = layer(hidden)
    assert out.min().item() >= 0.0, f"Min value {out.min().item()} < 0"
    assert out.max().item() <= 1.0, f"Max value {out.max().item()} > 1"


# ---------------------------------------------------------------------------
# 3. ConceptBottleneckModel forward returns tuple of 2 tensors
# ---------------------------------------------------------------------------

def test_cbm_forward_returns_tuple():
    torch.manual_seed(2)
    cfg = _make_config()
    model = ConceptBottleneckModel(cfg)
    model.eval()
    hidden = _make_hidden(seed=2)
    with torch.no_grad():
        result = model(hidden)
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2 elements, got {len(result)}"


# ---------------------------------------------------------------------------
# 4. concept_activations shape correct
# ---------------------------------------------------------------------------

def test_cbm_concept_activations_shape():
    torch.manual_seed(3)
    cfg = _make_config()
    model = ConceptBottleneckModel(cfg)
    model.eval()
    hidden = _make_hidden(seed=3)
    with torch.no_grad():
        concept_activations, _ = model(hidden)
    assert concept_activations.shape == (B, T, N_CONCEPTS), (
        f"Expected ({B}, {T}, {N_CONCEPTS}), got {concept_activations.shape}"
    )


# ---------------------------------------------------------------------------
# 5. class_logits shape correct
# ---------------------------------------------------------------------------

def test_cbm_class_logits_shape():
    torch.manual_seed(4)
    cfg = _make_config()
    model = ConceptBottleneckModel(cfg)
    model.eval()
    hidden = _make_hidden(seed=4)
    with torch.no_grad():
        _, class_logits = model(hidden)
    assert class_logits.shape == (B, T, N_CLASSES), (
        f"Expected ({B}, {T}, {N_CLASSES}), got {class_logits.shape}"
    )


# ---------------------------------------------------------------------------
# 6. concept_alignment_score returns scalar in [0, 1]
# ---------------------------------------------------------------------------

def test_concept_alignment_score_range():
    torch.manual_seed(5)
    preds = torch.rand(B * T, N_CONCEPTS)
    gt = torch.randint(0, 2, (B * T, N_CONCEPTS)).float()
    score = concept_alignment_score(preds, gt)
    assert score.shape == (), f"Expected scalar, got shape {score.shape}"
    assert 0.0 <= score.item() <= 1.0, f"Score out of [0,1]: {score.item()}"


# ---------------------------------------------------------------------------
# 7. concept_alignment_score perfect alignment -> high score
# ---------------------------------------------------------------------------

def test_concept_alignment_score_perfect():
    torch.manual_seed(6)
    N = 20
    # Ground truth: alternating 0/1 for each concept
    gt = torch.zeros(N, N_CONCEPTS)
    gt[N // 2 :, :] = 1.0  # top half positive

    # Perfect predictor: positive examples get score 1.0, negatives get 0.0
    preds = gt.clone().float()

    score = concept_alignment_score(preds, gt)
    assert score.item() > 0.9, (
        f"Expected near-perfect score (>0.9) for perfect predictor, got {score.item()}"
    )


# ---------------------------------------------------------------------------
# 8. concept_intervention changes output compared to no intervention
# ---------------------------------------------------------------------------

def test_concept_intervention_changes_output():
    torch.manual_seed(7)
    cfg = _make_config()
    model = ConceptBottleneckModel(cfg)
    model.eval()
    hidden = _make_hidden(seed=7)

    with torch.no_grad():
        _, baseline_logits = model(hidden)

    intervened_logits = concept_intervention(
        model, hidden, concept_idx=0, intervention_value=1.0
    )

    # After intervention the logits should differ from the baseline
    assert not torch.allclose(baseline_logits, intervened_logits), (
        "Expected concept intervention to change class logits"
    )


# ---------------------------------------------------------------------------
# 9. concept_intervention output shape matches class_logits
# ---------------------------------------------------------------------------

def test_concept_intervention_output_shape():
    torch.manual_seed(8)
    cfg = _make_config()
    model = ConceptBottleneckModel(cfg)
    model.eval()
    hidden = _make_hidden(seed=8)

    intervened_logits = concept_intervention(
        model, hidden, concept_idx=2, intervention_value=0.0
    )
    assert intervened_logits.shape == (B, T, N_CLASSES), (
        f"Expected ({B}, {T}, {N_CLASSES}), got {intervened_logits.shape}"
    )


# ---------------------------------------------------------------------------
# 10. extract_concept_importance returns shape (n_concepts,)
# ---------------------------------------------------------------------------

def test_extract_concept_importance_shape():
    torch.manual_seed(9)
    cfg = _make_config()
    model = ConceptBottleneckModel(cfg)
    model.eval()
    hidden = _make_hidden(seed=9)

    importance = extract_concept_importance(model, hidden)
    assert importance.shape == (N_CONCEPTS,), (
        f"Expected ({N_CONCEPTS},), got {importance.shape}"
    )


# ---------------------------------------------------------------------------
# 11. extract_concept_importance returns finite values
# ---------------------------------------------------------------------------

def test_extract_concept_importance_finite():
    torch.manual_seed(10)
    cfg = _make_config()
    model = ConceptBottleneckModel(cfg)
    model.eval()
    hidden = _make_hidden(seed=10)

    importance = extract_concept_importance(model, hidden)
    assert torch.isfinite(importance).all(), (
        f"Expected all finite importance scores, got: {importance}"
    )


# ---------------------------------------------------------------------------
# 12. ConceptLayer with relu activation — values >= 0
# ---------------------------------------------------------------------------

def test_concept_layer_relu_non_negative():
    torch.manual_seed(11)
    cfg = _make_config(activation="relu")
    layer = ConceptLayer(cfg)
    layer.eval()
    hidden = _make_hidden(seed=11)
    with torch.no_grad():
        out = layer(hidden)
    assert out.min().item() >= 0.0, (
        f"ReLU activations should be >= 0, got min {out.min().item()}"
    )


# ---------------------------------------------------------------------------
# 13. Intervention with value=0 and value=1 produce different logits
# ---------------------------------------------------------------------------

def test_concept_intervention_zero_vs_one():
    torch.manual_seed(12)
    cfg = _make_config()
    model = ConceptBottleneckModel(cfg)
    model.eval()
    hidden = _make_hidden(seed=12)

    logits_zero = concept_intervention(model, hidden, concept_idx=3, intervention_value=0.0)
    logits_one = concept_intervention(model, hidden, concept_idx=3, intervention_value=1.0)

    assert not torch.allclose(logits_zero, logits_one), (
        "Intervening with 0.0 vs 1.0 should produce different logits"
    )
