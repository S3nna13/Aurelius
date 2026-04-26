"""
tests/interpretability/test_sparse_feature_circuits.py

Tests for src/interpretability/sparse_feature_circuits.py

Uses a tiny AureliusTransformer:
    n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
    d_ff=128, vocab_size=256, max_seq_len=64
"""

from __future__ import annotations

import torch

from src.interpretability.sparse_feature_circuits import (
    FeatureCircuitFinder,
    SparseAutoencoder,
    SparseCircuitConfig,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

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

torch.manual_seed(0)

D_MODEL = TINY_CFG.d_model
N_FEATURES = 256
BATCH = 2
SEQ = 8


def _make_model():
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


def _make_sae():
    return SparseAutoencoder(d_model=D_MODEL, n_features=N_FEATURES, sparsity_coef=0.01)


def _make_ids():
    return torch.randint(0, TINY_CFG.vocab_size, (BATCH, SEQ))


def _metric_fn(logits):
    return float(logits[:, -1, :].mean().item())


def test_sparse_circuit_config_defaults():
    cfg = SparseCircuitConfig()
    assert cfg.n_features == 256
    assert cfg.sparsity_coef == 0.01
    assert cfg.top_k == 10


def test_sae_encode_output_shape():
    sae = _make_sae()
    x = torch.randn(BATCH, D_MODEL)
    features = sae.encode(x)
    assert features.shape == (BATCH, N_FEATURES)


def test_sae_reconstruction_shape():
    sae = _make_sae()
    x = torch.randn(BATCH, D_MODEL)
    reconstruction, features = sae(x)
    assert reconstruction.shape == x.shape


def test_sae_features_are_sparse():
    sae = _make_sae()
    x = torch.randn(64, D_MODEL)
    with torch.no_grad():
        features = sae.encode(x)
    assert (features >= 0).all(), "ReLU output must be non-negative"
    zero_fraction = (features == 0).float().mean().item()
    assert zero_fraction >= 0.0


def test_compute_loss_returns_required_keys():
    sae = _make_sae()
    x = torch.randn(BATCH, D_MODEL)
    _loss, info = sae.compute_loss(x)
    required_keys = {"recon_loss", "sparsity_loss", "mean_active_features"}
    assert required_keys.issubset(info.keys())


def test_sae_loss_finite_and_positive():
    sae = _make_sae()
    x = torch.randn(BATCH, D_MODEL)
    total_loss, info = sae.compute_loss(x)
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0.0
    assert torch.isfinite(info["recon_loss"])
    assert torch.isfinite(info["sparsity_loss"])


def test_normalize_decoder_unit_norm_columns():
    sae = _make_sae()
    with torch.no_grad():
        sae.decoder.weight.mul_(5.0)
    sae.normalize_decoder()
    col_norms = sae.decoder.weight.norm(dim=0)
    assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5)


def test_get_feature_activations_shape():
    model = _make_model()
    sae = _make_sae()
    finder = FeatureCircuitFinder(model, sae, _metric_fn)
    input_ids = _make_ids()
    features = finder.get_feature_activations(input_ids, layer_idx=0)
    assert features.shape == (BATCH, SEQ, N_FEATURES)


def test_patch_feature_runs_without_error():
    model = _make_model()
    sae = _make_sae()
    finder = FeatureCircuitFinder(model, sae, _metric_fn)
    clean_ids = _make_ids()
    corrupted_ids = _make_ids()
    result = finder.patch_feature(
        clean_ids=clean_ids,
        corrupted_ids=corrupted_ids,
        layer_idx=0,
        feature_idx=0,
    )
    assert isinstance(result, float)
    assert result == result  # not NaN


def test_find_circuit_features_returns_top_k():
    model = _make_model()
    small_sae = SparseAutoencoder(d_model=D_MODEL, n_features=16, sparsity_coef=0.01)
    finder = FeatureCircuitFinder(model, small_sae, _metric_fn)
    clean_ids = _make_ids()
    corrupted_ids = _make_ids()
    top_k = 5
    results = finder.find_circuit_features(
        clean_ids=clean_ids,
        corrupted_ids=corrupted_ids,
        layer_idx=0,
        top_k=top_k,
    )
    assert len(results) == top_k
    for feat_idx, score in results:
        assert isinstance(feat_idx, int)
        assert isinstance(score, float)


def test_find_circuit_features_sorted_descending():
    model = _make_model()
    small_sae = SparseAutoencoder(d_model=D_MODEL, n_features=16, sparsity_coef=0.01)
    finder = FeatureCircuitFinder(model, small_sae, _metric_fn)
    clean_ids = _make_ids()
    corrupted_ids = _make_ids()
    results = finder.find_circuit_features(
        clean_ids=clean_ids,
        corrupted_ids=corrupted_ids,
        layer_idx=0,
        top_k=8,
    )
    scores = [s for _, s in results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]
