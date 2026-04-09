"""Tests for src/eval/probing_advanced.py — 12 tests covering all public APIs."""

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.eval.probing_advanced import (
    ProbingConfig,
    LinearProbeV2,
    MLPProbe,
    extract_layer_representations,
    train_probe,
    estimate_mutual_information,
    MultiLayerProber,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_cfg():
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
def small_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def input_ids():
    """Short sequence: B=4, T=4."""
    return torch.randint(0, 256, (4, 4))


@pytest.fixture
def binary_labels():
    return torch.tensor([0, 1, 0, 1], dtype=torch.long)


# ---------------------------------------------------------------------------
# 1. ProbingConfig defaults
# ---------------------------------------------------------------------------

def test_probing_config_defaults():
    cfg = ProbingConfig()
    assert cfg.probe_type == "linear"
    assert cfg.hidden_dim == 128
    assert cfg.n_layers_to_probe is None
    assert cfg.n_epochs == 10
    assert cfg.lr == 1e-3
    assert cfg.batch_size == 32


# ---------------------------------------------------------------------------
# 2. LinearProbeV2 forward shape
# ---------------------------------------------------------------------------

def test_linear_probe_v2_forward_shape():
    probe = LinearProbeV2(input_dim=64, n_classes=3)
    x = torch.randn(8, 64)
    out = probe(x)
    assert out.shape == (8, 3)


# ---------------------------------------------------------------------------
# 3. MLPProbe forward shape
# ---------------------------------------------------------------------------

def test_mlp_probe_forward_shape():
    probe = MLPProbe(input_dim=64, hidden_dim=32, n_classes=5)
    x = torch.randn(16, 64)
    out = probe(x)
    assert out.shape == (16, 5)


# ---------------------------------------------------------------------------
# 4. extract_layer_representations returns dict with correct keys
# ---------------------------------------------------------------------------

def test_extract_layer_representations_keys(small_model, input_ids):
    layer_indices = [0, 1]
    result = extract_layer_representations(small_model, input_ids, layer_indices)
    assert set(result.keys()) == {0, 1}


# ---------------------------------------------------------------------------
# 5. extract_layer_representations hidden state shape (B, T, D)
# ---------------------------------------------------------------------------

def test_extract_layer_representations_shape(small_model, input_ids):
    result = extract_layer_representations(small_model, input_ids, [0])
    hidden = result[0]
    B, T = input_ids.shape
    D = 64  # d_model from small_cfg
    assert hidden.shape == (B, T, D)


# ---------------------------------------------------------------------------
# 6. train_probe returns correct keys
# ---------------------------------------------------------------------------

def test_train_probe_returns_correct_keys():
    probe = LinearProbeV2(input_dim=16, n_classes=2)
    X = torch.randn(20, 16)
    y = torch.randint(0, 2, (20,))
    cfg = ProbingConfig(n_epochs=2)
    result = train_probe(probe, X, y, cfg)
    assert "train_acc" in result
    assert "final_loss" in result


# ---------------------------------------------------------------------------
# 7. train_probe accuracy in [0, 1]
# ---------------------------------------------------------------------------

def test_train_probe_accuracy_range():
    probe = LinearProbeV2(input_dim=16, n_classes=2)
    X = torch.randn(20, 16)
    y = torch.randint(0, 2, (20,))
    cfg = ProbingConfig(n_epochs=3)
    result = train_probe(probe, X, y, cfg)
    assert 0.0 <= result["train_acc"] <= 1.0


# ---------------------------------------------------------------------------
# 8. estimate_mutual_information returns float
# ---------------------------------------------------------------------------

def test_estimate_mi_returns_float():
    torch.manual_seed(42)
    X = torch.randn(50, 8)
    y = torch.randint(0, 2, (50,))
    mi = estimate_mutual_information(X, y, n_bins=5)
    assert isinstance(mi, float)


# ---------------------------------------------------------------------------
# 9. estimate_mutual_information perfectly correlated -> high MI
# ---------------------------------------------------------------------------

def test_estimate_mi_perfectly_correlated():
    # X is perfectly correlated with y: class 0 gets large negative values, class 1 large positive
    N = 100
    y = torch.cat([torch.zeros(N // 2, dtype=torch.long), torch.ones(N // 2, dtype=torch.long)])
    X = torch.cat([
        torch.full((N // 2, 4), -10.0),
        torch.full((N // 2, 4), +10.0),
    ])
    mi = estimate_mutual_information(X, y, n_bins=10)
    # For perfect binary separation, MI should be close to 1 bit
    assert mi > 0.5, f"Expected MI > 0.5 bits for perfectly correlated data, got {mi}"


# ---------------------------------------------------------------------------
# 10. MultiLayerProber.probe_all_layers returns dict with layer keys
# ---------------------------------------------------------------------------

def test_multi_layer_prober_keys(small_model, input_ids, binary_labels):
    cfg = ProbingConfig(n_epochs=2)
    prober = MultiLayerProber(small_model, cfg)
    results = prober.probe_all_layers(input_ids, binary_labels, layer_indices=[0, 1])
    assert set(results.keys()) == {0, 1}


# ---------------------------------------------------------------------------
# 11. MultiLayerProber layer accuracies in [0, 1]
# ---------------------------------------------------------------------------

def test_multi_layer_prober_accuracy_range(small_model, input_ids, binary_labels):
    cfg = ProbingConfig(n_epochs=2)
    prober = MultiLayerProber(small_model, cfg)
    results = prober.probe_all_layers(input_ids, binary_labels, layer_indices=[0, 1])
    for idx, info in results.items():
        assert 0.0 <= info["train_acc"] <= 1.0, (
            f"Layer {idx} accuracy {info['train_acc']} out of [0, 1]"
        )


# ---------------------------------------------------------------------------
# 12. MLPProbe output shape with different input sizes
# ---------------------------------------------------------------------------

def test_mlp_probe_various_input_sizes():
    for input_dim, hidden_dim, n_classes, batch in [(32, 16, 2, 4), (128, 64, 10, 8)]:
        probe = MLPProbe(input_dim=input_dim, hidden_dim=hidden_dim, n_classes=n_classes)
        x = torch.randn(batch, input_dim)
        out = probe(x)
        assert out.shape == (batch, n_classes), (
            f"Expected ({batch}, {n_classes}), got {out.shape}"
        )
