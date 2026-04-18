"""Tests for src/security/model_fingerprint.py."""

from __future__ import annotations

import math

import pytest
import torch
import torch.optim as optim

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.security.model_fingerprint import FingerprintConfig, ModelFingerprint

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MODEL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

FP_CFG = FingerprintConfig(n_keys=4, key_len=4, seed=0, embed_strength=0.1)


@pytest.fixture(scope="module")
def model() -> AureliusTransformer:
    torch.manual_seed(7)
    return AureliusTransformer(MODEL_CFG)


@pytest.fixture(scope="module")
def fp(model: AureliusTransformer) -> ModelFingerprint:
    return ModelFingerprint(model, FP_CFG)


# ---------------------------------------------------------------------------
# 1. Instantiation
# ---------------------------------------------------------------------------


def test_instantiation(model: AureliusTransformer) -> None:
    mf = ModelFingerprint(model, FP_CFG)
    assert isinstance(mf, ModelFingerprint)


# ---------------------------------------------------------------------------
# 2-4. generate_keys shapes
# ---------------------------------------------------------------------------


def test_generate_keys_returns_tuple(fp: ModelFingerprint) -> None:
    result = fp.generate_keys()
    assert isinstance(result, tuple) and len(result) == 2


def test_key_ids_shape(fp: ModelFingerprint) -> None:
    key_ids, _ = fp.generate_keys()
    assert key_ids.shape == (FP_CFG.n_keys, FP_CFG.key_len)


def test_target_ids_shape(fp: ModelFingerprint) -> None:
    _, target_ids = fp.generate_keys()
    assert target_ids.shape == (FP_CFG.n_keys,)


# ---------------------------------------------------------------------------
# 5-6. Valid vocab ranges
# ---------------------------------------------------------------------------


def test_key_ids_valid_range(fp: ModelFingerprint) -> None:
    key_ids, _ = fp.generate_keys()
    assert int(key_ids.min()) >= 0
    assert int(key_ids.max()) < MODEL_CFG.vocab_size


def test_target_ids_valid_range(fp: ModelFingerprint) -> None:
    _, target_ids = fp.generate_keys()
    assert int(target_ids.min()) >= 0
    assert int(target_ids.max()) < MODEL_CFG.vocab_size


# ---------------------------------------------------------------------------
# 7. Determinism
# ---------------------------------------------------------------------------


def test_same_seed_same_keys(model: AureliusTransformer) -> None:
    cfg_a = FingerprintConfig(n_keys=4, key_len=4, seed=0, embed_strength=0.1)
    cfg_b = FingerprintConfig(n_keys=4, key_len=4, seed=0, embed_strength=0.1)
    mf_a = ModelFingerprint(model, cfg_a)
    mf_b = ModelFingerprint(model, cfg_b)
    assert torch.equal(mf_a.key_ids, mf_b.key_ids)
    assert torch.equal(mf_a.target_ids, mf_b.target_ids)


# ---------------------------------------------------------------------------
# 8-10. embed_fingerprint
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embed_losses(model: AureliusTransformer) -> list:
    mf = ModelFingerprint(model, FP_CFG)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return mf.embed_fingerprint(optimizer, n_steps=3)


def test_embed_fingerprint_returns_list(embed_losses: list) -> None:
    assert isinstance(embed_losses, list)
    assert all(isinstance(v, float) for v in embed_losses)


def test_embed_fingerprint_length(embed_losses: list) -> None:
    assert len(embed_losses) == 3


def test_embed_fingerprint_finite_losses(embed_losses: list) -> None:
    assert all(math.isfinite(v) for v in embed_losses)


# ---------------------------------------------------------------------------
# 11-12. verify
# ---------------------------------------------------------------------------


def test_verify_returns_bool_float(fp: ModelFingerprint) -> None:
    result = fp.verify()
    assert isinstance(result, tuple) and len(result) == 2
    verified, accuracy = result
    assert isinstance(verified, bool)
    assert isinstance(accuracy, float)


def test_verify_accuracy_range(fp: ModelFingerprint) -> None:
    _, accuracy = fp.verify()
    assert 0.0 <= accuracy <= 1.0


# ---------------------------------------------------------------------------
# 13-14. extract_signature
# ---------------------------------------------------------------------------


def test_extract_signature_shape(fp: ModelFingerprint) -> None:
    sig = fp.extract_signature(layer_idx=0)
    assert sig.shape == (2,)


def test_extract_signature_finite(fp: ModelFingerprint) -> None:
    sig = fp.extract_signature(layer_idx=0)
    assert all(math.isfinite(v.item()) for v in sig)
