"""Tests for src/training/batch_sampler.py"""

import numpy as np
import pytest
import torch

from src.training.batch_sampler import (
    BATCH_SAMPLER,
    BatchSamplerConfig,
    RandomOffsetBatchSampler,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_data(n_tokens: int = 5000) -> np.ndarray:
    return np.arange(n_tokens, dtype=np.int64)


# ---------------------------------------------------------------------------
# Tests: get_batch() — shape
# ---------------------------------------------------------------------------


def test_get_batch_x_shape():
    cfg = BatchSamplerConfig(block_size=32, batch_size=4)
    sampler = RandomOffsetBatchSampler(data=make_data(), config=cfg)
    x, y = sampler.get_batch()
    assert x.shape == (4, 32)


def test_get_batch_y_shape():
    cfg = BatchSamplerConfig(block_size=32, batch_size=4)
    sampler = RandomOffsetBatchSampler(data=make_data(), config=cfg)
    x, y = sampler.get_batch()
    assert y.shape == (4, 32)


def test_get_batch_dtype_int64():
    cfg = BatchSamplerConfig(block_size=16, batch_size=2)
    sampler = RandomOffsetBatchSampler(data=make_data(), config=cfg)
    x, y = sampler.get_batch()
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


def test_get_batch_y_is_x_shifted_by_one():
    """y[i] should equal x[i] shifted left by 1 token."""
    data = np.arange(200, dtype=np.int64)
    cfg = BatchSamplerConfig(block_size=10, batch_size=1)
    sampler = RandomOffsetBatchSampler(data=data, config=cfg)
    x, y = sampler.get_batch()
    # For a monotone sequence, y = x + 1 (each token is index+1)
    assert torch.all(y == x + 1)


def test_get_batch_offsets_within_bounds():
    data = make_data(100)
    cfg = BatchSamplerConfig(block_size=10, batch_size=8)
    sampler = RandomOffsetBatchSampler(data=data, config=cfg)
    for _ in range(20):
        x, y = sampler.get_batch()
        assert x.shape == (8, 10)
        assert y.shape == (8, 10)


def test_get_batch_with_explicit_data_arg():
    data = make_data(500)
    cfg = BatchSamplerConfig(block_size=16, batch_size=3)
    sampler = RandomOffsetBatchSampler(config=cfg)  # no data in __init__
    x, y = sampler.get_batch(data=data)
    assert x.shape == (3, 16)


def test_get_batch_no_data_raises():
    sampler = RandomOffsetBatchSampler()
    with pytest.raises(ValueError, match="No data provided"):
        sampler.get_batch()


def test_get_batch_returns_tensors():
    cfg = BatchSamplerConfig(block_size=8, batch_size=2)
    sampler = RandomOffsetBatchSampler(data=make_data(), config=cfg)
    x, y = sampler.get_batch()
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


# ---------------------------------------------------------------------------
# Tests: from_array()
# ---------------------------------------------------------------------------


def test_from_array_returns_new_sampler():
    data = make_data()
    cfg = BatchSamplerConfig(block_size=8, batch_size=2)
    base = RandomOffsetBatchSampler(config=cfg)
    new = base.from_array(data)
    assert new is not base
    assert new.data is data


def test_from_array_inherits_config():
    data = make_data()
    cfg = BatchSamplerConfig(block_size=64, batch_size=5)
    base = RandomOffsetBatchSampler(config=cfg)
    new = base.from_array(data)
    assert new.config.block_size == 64
    assert new.config.batch_size == 5


# ---------------------------------------------------------------------------
# Tests: estimate_steps_to_tokens()
# ---------------------------------------------------------------------------


def test_estimate_steps_basic():
    cfg = BatchSamplerConfig(block_size=1024, batch_size=12)
    sampler = RandomOffsetBatchSampler(config=cfg)
    tokens_per_step = 12 * 1024  # 12288
    total_tokens = tokens_per_step * 100
    assert sampler.estimate_steps_to_tokens(total_tokens) == 100


def test_estimate_steps_with_epochs():
    cfg = BatchSamplerConfig(block_size=512, batch_size=8)
    sampler = RandomOffsetBatchSampler(config=cfg)
    tokens_per_step = 8 * 512
    total_tokens = tokens_per_step * 50
    assert sampler.estimate_steps_to_tokens(total_tokens, epochs=2.0) == 100


def test_estimate_steps_returns_int():
    cfg = BatchSamplerConfig(block_size=64, batch_size=4)
    sampler = RandomOffsetBatchSampler(config=cfg)
    result = sampler.estimate_steps_to_tokens(100_000)
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Tests: module-level singleton
# ---------------------------------------------------------------------------


def test_singleton_is_instance():
    assert isinstance(BATCH_SAMPLER, RandomOffsetBatchSampler)


def test_singleton_has_default_config():
    assert BATCH_SAMPLER.config.block_size == 1024
    assert BATCH_SAMPLER.config.batch_size == 12
