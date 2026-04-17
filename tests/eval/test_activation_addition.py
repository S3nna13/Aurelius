"""Tests for src/eval/activation_addition.py.

All tests use tiny dims: d_model=32, n_layers=4, B=2.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

from aurelius.eval.activation_addition import (
    SteeringVector,
    ActivationAddition,
    SteeringVectorExtractor,
    RepresentationDatabase,
)


# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

D_MODEL = 32
N_LAYERS = 4
B = 2
T = 8


def _make_layers(n: int = N_LAYERS, d: int = D_MODEL) -> List[nn.Module]:
    """Return a list of simple Linear layers to act as stand-in transformer layers."""
    layers = []
    for _ in range(n):
        layers.append(nn.Linear(d, d, bias=False))
    return layers


def _make_sv(layer_idx: int = 0, d: int = D_MODEL, seed: int = 42) -> SteeringVector:
    torch.manual_seed(seed)
    v = torch.randn(d)
    v = F.normalize(v, dim=0)
    return SteeringVector(direction=v, layer_idx=layer_idx)


def _make_model_fn(layers: List[nn.Module]):
    """Return a model_fn that runs each layer sequentially and collects hidden states.

    Uses the sum of input_ids as a seed so pos and neg inputs produce distinct hidden
    states even when the layers are deterministic.
    """
    def model_fn(input_ids: Tensor) -> List[Tensor]:
        # input_ids: (B, T) — embed as float to get (B, T, d_model)
        B_, T_ = input_ids.shape
        d = layers[0].weight.shape[0]
        # Use a data-dependent seed so pos != neg inputs → different hidden states
        seed = int(input_ids.sum().item()) % (2 ** 31)
        gen = torch.Generator()
        gen.manual_seed(seed)
        hidden = torch.randn(B_, T_, d, generator=gen)
        hiddens: List[Tensor] = []
        for layer in layers:
            hidden = layer(hidden)
            hiddens.append(hidden)
        return hiddens
    return model_fn


# ---------------------------------------------------------------------------
# 1. SteeringVector has correct fields
# ---------------------------------------------------------------------------

def test_steering_vector_fields():
    """SteeringVector stores direction and layer_idx."""
    direction = torch.randn(D_MODEL)
    sv = SteeringVector(direction=direction, layer_idx=2)
    assert sv.layer_idx == 2
    assert sv.direction is direction
    assert sv.direction.shape == (D_MODEL,)


# ---------------------------------------------------------------------------
# 2. ActivationAddition: hook count before entering context is 0
# ---------------------------------------------------------------------------

def test_activation_addition_no_hooks_before_context():
    """No hooks are installed before entering the context manager."""
    layers = _make_layers()
    aa = ActivationAddition(layers)
    sv = _make_sv(layer_idx=0)
    aa.add_vector(sv, alpha=1.0)
    # _handles should be empty before __enter__
    assert len(aa._handles) == 0


# ---------------------------------------------------------------------------
# 3. ActivationAddition: hook count inside context equals number of vectors
# ---------------------------------------------------------------------------

def test_activation_addition_hooks_installed_during_context():
    """Hooks are installed for all registered vectors during the context."""
    layers = _make_layers()
    aa = ActivationAddition(layers)
    aa.add_vector(_make_sv(layer_idx=0), alpha=1.0)
    aa.add_vector(_make_sv(layer_idx=1, seed=7), alpha=1.0)
    with aa:
        assert len(aa._handles) == 2


# ---------------------------------------------------------------------------
# 4. ActivationAddition: hook count after exiting context is 0
# ---------------------------------------------------------------------------

def test_activation_addition_hooks_removed_after_context():
    """All hooks are removed after exiting the context manager."""
    layers = _make_layers()
    aa = ActivationAddition(layers)
    aa.add_vector(_make_sv(layer_idx=0), alpha=1.0)
    with aa:
        pass
    assert len(aa._handles) == 0


# ---------------------------------------------------------------------------
# 5. Steering actually changes model output
# ---------------------------------------------------------------------------

def test_steering_changes_output():
    """Model output differs with vs. without steering (alpha != 0)."""
    d = D_MODEL
    layer = nn.Linear(d, d, bias=False)
    layers = [layer]

    torch.manual_seed(1)
    x = torch.randn(B, T, d)

    with torch.no_grad():
        baseline = layer(x).clone()

    aa = ActivationAddition(layers)
    sv = _make_sv(layer_idx=0)
    aa.add_vector(sv, alpha=5.0)

    with torch.no_grad():
        with aa:
            steered = layer(x).clone()

    assert not torch.allclose(baseline, steered, atol=1e-5), (
        "Steering should change the layer output"
    )


# ---------------------------------------------------------------------------
# 6. SteeringVectorExtractor.extract returns correct layer_idx
# ---------------------------------------------------------------------------

def test_extractor_layer_idx():
    """Extracted SteeringVector carries the requested layer_idx."""
    layers = _make_layers()
    model_fn = _make_model_fn(layers)
    extractor = SteeringVectorExtractor()
    pos = torch.randint(0, 10, (B, T))
    neg = torch.randint(0, 10, (B, T))
    sv = extractor.extract(model_fn, pos, neg, layer_idx=2)
    assert sv.layer_idx == 2


# ---------------------------------------------------------------------------
# 7. Extracted direction is unit-norm
# ---------------------------------------------------------------------------

def test_extractor_direction_unit_norm():
    """Direction returned by extractor is L2-normalised."""
    layers = _make_layers()
    model_fn = _make_model_fn(layers)
    extractor = SteeringVectorExtractor()
    torch.manual_seed(99)
    pos = torch.randint(0, 10, (B, T))
    neg = torch.randint(0, 10, (B, T))
    sv = extractor.extract(model_fn, pos, neg, layer_idx=1)
    norm = sv.direction.norm().item()
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


# ---------------------------------------------------------------------------
# 8. Extracted direction shape is (d_model,)
# ---------------------------------------------------------------------------

def test_extractor_direction_shape():
    """Extracted direction has shape (d_model,)."""
    layers = _make_layers()
    model_fn = _make_model_fn(layers)
    extractor = SteeringVectorExtractor()
    pos = torch.randint(0, 10, (B, T))
    neg = torch.randint(0, 10, (B, T))
    sv = extractor.extract(model_fn, pos, neg, layer_idx=0)
    assert sv.direction.shape == (D_MODEL,)


# ---------------------------------------------------------------------------
# 9. RepresentationDatabase: add / get round-trip
# ---------------------------------------------------------------------------

def test_repr_database_add_get():
    """add then get returns the same SteeringVector."""
    db = RepresentationDatabase()
    sv = _make_sv(layer_idx=0)
    db.add("honesty", sv)
    retrieved = db.get("honesty")
    assert retrieved is sv


# ---------------------------------------------------------------------------
# 10. RepresentationDatabase: list_names
# ---------------------------------------------------------------------------

def test_repr_database_list_names():
    """list_names returns all added names."""
    db = RepresentationDatabase()
    db.add("happy", _make_sv(layer_idx=0, seed=1))
    db.add("sad", _make_sv(layer_idx=1, seed=2))
    names = db.list_names()
    assert set(names) == {"happy", "sad"}


# ---------------------------------------------------------------------------
# 11. RepresentationDatabase: remove deletes the entry
# ---------------------------------------------------------------------------

def test_repr_database_remove():
    """remove deletes the named entry; get raises KeyError afterwards."""
    db = RepresentationDatabase()
    sv = _make_sv(layer_idx=0)
    db.add("courage", sv)
    db.remove("courage")
    assert "courage" not in db.list_names()
    with pytest.raises(KeyError):
        db.get("courage")


# ---------------------------------------------------------------------------
# 12. remove_vector removes only the targeted layer
# ---------------------------------------------------------------------------

def test_remove_vector_removes_only_target():
    """remove_vector for layer 0 leaves the layer 1 vector intact."""
    layers = _make_layers()
    aa = ActivationAddition(layers)
    aa.add_vector(_make_sv(layer_idx=0, seed=1), alpha=1.0)
    aa.add_vector(_make_sv(layer_idx=1, seed=2), alpha=1.0)
    aa.remove_vector(0)
    assert 0 not in aa._vectors
    assert 1 in aa._vectors


# ---------------------------------------------------------------------------
# 13. alpha=0 steering produces identical output as no steering
# ---------------------------------------------------------------------------

def test_alpha_zero_identical_to_no_steering():
    """With alpha=0 the steered output is numerically identical to baseline."""
    d = D_MODEL
    layer = nn.Linear(d, d, bias=False)
    layers = [layer]

    torch.manual_seed(3)
    x = torch.randn(B, T, d)

    with torch.no_grad():
        baseline = layer(x).clone()

    aa = ActivationAddition(layers)
    sv = _make_sv(layer_idx=0)
    aa.add_vector(sv, alpha=0.0)

    with torch.no_grad():
        with aa:
            steered = layer(x).clone()

    assert torch.allclose(baseline, steered, atol=1e-6), (
        "alpha=0 should not change the output"
    )


# ---------------------------------------------------------------------------
# 14. Works with B=1, T=1 (minimal batch/seq)
# ---------------------------------------------------------------------------

def test_works_with_b1_t1():
    """ActivationAddition works for a single token in a single batch."""
    d = D_MODEL
    layer = nn.Linear(d, d, bias=False)
    layers = [layer]

    torch.manual_seed(5)
    x = torch.randn(1, 1, d)

    aa = ActivationAddition(layers)
    sv = _make_sv(layer_idx=0)
    aa.add_vector(sv, alpha=2.0)

    with torch.no_grad():
        baseline = layer(x).clone()

    with torch.no_grad():
        with aa:
            steered = layer(x).clone()

    # Should differ (alpha != 0)
    assert not torch.allclose(baseline, steered, atol=1e-5)


# ---------------------------------------------------------------------------
# 15. Extractor works with B=1, T=1
# ---------------------------------------------------------------------------

def test_extractor_b1_t1():
    """SteeringVectorExtractor handles B=1, T=1 inputs without error."""
    layers = _make_layers()
    model_fn = _make_model_fn(layers)
    extractor = SteeringVectorExtractor()
    pos = torch.randint(0, 10, (1, 1))
    neg = torch.randint(0, 10, (1, 1))
    sv = extractor.extract(model_fn, pos, neg, layer_idx=0)
    assert sv.direction.shape == (D_MODEL,)
    assert abs(sv.direction.norm().item() - 1.0) < 1e-5
