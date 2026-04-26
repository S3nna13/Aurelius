"""
Tests for src/interpretability/probe_intervention.py

Uses a tiny AureliusConfig:
    n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128,
    vocab_size=256, max_seq_len=64
"""

from __future__ import annotations

import torch

from src.interpretability.probe_intervention import (
    LinearProbe,
    ProbeConfig,
    ProbeInterventionExperiment,
    ProbeTrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared tiny constants
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

D_MODEL = TINY_CFG.d_model  # 64
N_CLASSES = 2
BATCH = 4
SEQ = 8

torch.manual_seed(42)


def _make_model() -> AureliusTransformer:
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


def _make_probe(n_classes: int = N_CLASSES) -> LinearProbe:
    return LinearProbe(D_MODEL, n_classes=n_classes)


def _make_input() -> torch.Tensor:
    return torch.randint(0, TINY_CFG.vocab_size, (BATCH, SEQ))


# ---------------------------------------------------------------------------
# Test 1: ProbeConfig defaults correct
# ---------------------------------------------------------------------------


def test_probe_config_defaults():
    cfg = ProbeConfig()
    assert cfg.d_model == 64
    assert cfg.n_classes == 2
    assert abs(cfg.lr - 0.01) < 1e-9
    assert cfg.n_epochs == 100


# ---------------------------------------------------------------------------
# Test 2: LinearProbe output shape is (batch, n_classes)
# ---------------------------------------------------------------------------


def test_linear_probe_output_shape():
    probe = _make_probe()
    h = torch.randn(BATCH, D_MODEL)
    out = probe(h)
    assert out.shape == (BATCH, N_CLASSES), f"Expected ({BATCH}, {N_CLASSES}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 3: get_direction returns (d_model,) tensor for binary probe
# ---------------------------------------------------------------------------


def test_get_direction_shape():
    probe = _make_probe(n_classes=2)
    direction = probe.get_direction()
    assert direction.shape == (D_MODEL,), f"Expected ({D_MODEL},), got {direction.shape}"


# ---------------------------------------------------------------------------
# Test 4: ProbeTrainer.fit returns dict with 'train_accuracy' key
# ---------------------------------------------------------------------------


def test_probe_trainer_fit_returns_dict_with_accuracy():
    probe = _make_probe()
    trainer = ProbeTrainer(probe, lr=0.01, n_epochs=5)
    X = torch.randn(20, D_MODEL)
    y = torch.randint(0, N_CLASSES, (20,))
    result = trainer.fit(X, y)
    assert isinstance(result, dict), "fit() should return a dict"
    assert "train_accuracy" in result, "dict must contain 'train_accuracy'"


# ---------------------------------------------------------------------------
# Test 5: Accuracy on linearly separable data > 0.9
# ---------------------------------------------------------------------------


def test_probe_accuracy_on_separable_data():
    """Linearly separable data: class 0 cluster vs class 1 cluster."""
    torch.manual_seed(0)
    n = 100
    X_pos = torch.randn(n // 2, D_MODEL) + 5.0  # class 1 far positive
    X_neg = torch.randn(n // 2, D_MODEL) - 5.0  # class 0 far negative
    X = torch.cat([X_neg, X_pos], dim=0)
    y = torch.cat(
        [torch.zeros(n // 2, dtype=torch.long), torch.ones(n // 2, dtype=torch.long)], dim=0
    )

    probe = _make_probe(n_classes=2)
    trainer = ProbeTrainer(probe, lr=0.05, n_epochs=200)
    trainer.fit(X, y)
    acc = trainer.evaluate(X, y)
    assert acc > 0.9, f"Expected accuracy > 0.9 on separable data, got {acc:.4f}"


# ---------------------------------------------------------------------------
# Test 6: ProbeTrainer.evaluate returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_probe_trainer_evaluate_returns_valid_float():
    probe = _make_probe()
    trainer = ProbeTrainer(probe, lr=0.01, n_epochs=5)
    X = torch.randn(20, D_MODEL)
    y = torch.randint(0, N_CLASSES, (20,))
    trainer.fit(X, y)
    acc = trainer.evaluate(X, y)
    assert isinstance(acc, float), "evaluate() should return a float"
    assert 0.0 <= acc <= 1.0, f"Accuracy {acc} is outside [0, 1]"


# ---------------------------------------------------------------------------
# Test 7: extract_representations returns (batch, d_model) tensor
# ---------------------------------------------------------------------------


def test_extract_representations_shape():
    model = _make_model()
    probe = _make_probe()
    exp = ProbeInterventionExperiment(model, probe, layer_idx=0)
    input_ids = _make_input()
    reps = exp.extract_representations(input_ids)
    assert reps.shape == (BATCH, D_MODEL), f"Expected ({BATCH}, {D_MODEL}), got {reps.shape}"


# ---------------------------------------------------------------------------
# Test 8: intervention_effect runs without error
# ---------------------------------------------------------------------------


def test_intervention_effect_runs():
    model = _make_model()
    probe = _make_probe()
    # Train probe briefly so get_direction is valid
    X = torch.randn(20, D_MODEL)
    y = torch.randint(0, N_CLASSES, (20,))
    ProbeTrainer(probe, lr=0.01, n_epochs=5).fit(X, y)

    exp = ProbeInterventionExperiment(model, probe, layer_idx=0)
    input_ids = _make_input()
    delta = exp.intervention_effect(input_ids, intervention_strength=1.0)
    assert isinstance(delta, float), "intervention_effect() should return a float"


# ---------------------------------------------------------------------------
# Test 9: causal_scrubbing returns float
# ---------------------------------------------------------------------------


def test_causal_scrubbing_returns_float():
    model = _make_model()
    probe = _make_probe()
    X = torch.randn(20, D_MODEL)
    y = torch.randint(0, N_CLASSES, (20,))
    ProbeTrainer(probe, lr=0.01, n_epochs=5).fit(X, y)

    exp = ProbeInterventionExperiment(model, probe, layer_idx=0)
    clean_ids = _make_input()
    corrupted_ids = _make_input()
    result = exp.causal_scrubbing(clean_ids, corrupted_ids)
    assert isinstance(result, float), "causal_scrubbing() should return a float"


# ---------------------------------------------------------------------------
# Test 10: ProbeInterventionExperiment works with different layer indices
# ---------------------------------------------------------------------------


def test_experiment_different_layer_indices():
    model = _make_model()
    probe = _make_probe()
    X = torch.randn(20, D_MODEL)
    y = torch.randint(0, N_CLASSES, (20,))
    ProbeTrainer(probe, lr=0.01, n_epochs=5).fit(X, y)

    input_ids = _make_input()
    for layer_idx in range(TINY_CFG.n_layers):
        exp = ProbeInterventionExperiment(model, probe, layer_idx=layer_idx)
        reps = exp.extract_representations(input_ids)
        assert reps.shape == (BATCH, D_MODEL), (
            f"Layer {layer_idx}: expected ({BATCH}, {D_MODEL}), got {reps.shape}"
        )
