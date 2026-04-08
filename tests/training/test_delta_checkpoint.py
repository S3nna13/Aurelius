"""Tests for src/training/delta_checkpoint.py."""
from __future__ import annotations

import tempfile

import pytest
import torch

from src.training.delta_checkpoint import (
    DeltaConfig,
    compute_delta,
    apply_delta,
    compress_delta,
    delta_size_bytes,
    quantize_delta,
    DeltaCheckpointManager,
)


# ---------------------------------------------------------------------------
# Helper – small 4×4 state dicts
# ---------------------------------------------------------------------------

def _state_a() -> dict:
    return {"w": torch.ones(4, 4), "b": torch.zeros(4)}


def _state_b() -> dict:
    return {"w": torch.ones(4, 4) * 2.0, "b": torch.ones(4)}


# ---------------------------------------------------------------------------
# 1. DeltaConfig defaults
# ---------------------------------------------------------------------------

def test_delta_config_defaults():
    cfg = DeltaConfig()
    assert cfg.compression == "none"
    assert cfg.top_k_ratio == pytest.approx(0.1)
    assert cfg.threshold == pytest.approx(1e-4)
    assert cfg.dtype == "float32"


# ---------------------------------------------------------------------------
# 2. compute_delta – shape preserved
# ---------------------------------------------------------------------------

def test_compute_delta_shape():
    a, b = _state_a(), _state_b()
    delta = compute_delta(a, b)
    assert set(delta.keys()) == set(b.keys())
    for key in delta:
        assert delta[key].shape == b[key].shape


# ---------------------------------------------------------------------------
# 3. compute_delta – values correct (delta = b - a)
# ---------------------------------------------------------------------------

def test_compute_delta_values():
    a, b = _state_a(), _state_b()
    delta = compute_delta(a, b)
    # w: 2 - 1 = 1; b: 1 - 0 = 1
    assert torch.allclose(delta["w"], torch.ones(4, 4))
    assert torch.allclose(delta["b"], torch.ones(4))


# ---------------------------------------------------------------------------
# 4. apply_delta – roundtrip: apply_delta(a, compute_delta(a, b)) ≈ b
# ---------------------------------------------------------------------------

def test_apply_delta_roundtrip():
    a, b = _state_a(), _state_b()
    delta = compute_delta(a, b)
    reconstructed = apply_delta(a, delta)
    for key in b:
        assert torch.allclose(reconstructed[key].float(), b[key].float(), atol=1e-5), (
            f"Roundtrip failed for key {key!r}"
        )


# ---------------------------------------------------------------------------
# 5. compress_delta – "none" is a passthrough
# ---------------------------------------------------------------------------

def test_compress_delta_none_passthrough():
    a, b = _state_a(), _state_b()
    delta = compute_delta(a, b)
    cfg = DeltaConfig(compression="none")
    compressed = compress_delta(delta, cfg)
    for key in delta:
        assert torch.allclose(compressed[key], delta[key])


# ---------------------------------------------------------------------------
# 6. compress_delta – "top_k" zeros out low-magnitude values
# ---------------------------------------------------------------------------

def test_compress_delta_top_k():
    # Build a delta with mixed magnitudes
    delta = {"w": torch.tensor([0.001, 10.0, 0.002, 5.0])}
    cfg = DeltaConfig(compression="top_k", top_k_ratio=0.5)  # keep top 50% = 2 elements
    compressed = compress_delta(delta, cfg)
    # The two large values should survive; small ones zeroed
    nonzero = compressed["w"].nonzero(as_tuple=False).numel()
    assert nonzero == 2


# ---------------------------------------------------------------------------
# 7. compress_delta – "threshold" zeros values below threshold
# ---------------------------------------------------------------------------

def test_compress_delta_threshold():
    delta = {"w": torch.tensor([0.00001, 1.0, 0.00002, 2.0])}
    cfg = DeltaConfig(compression="threshold", threshold=0.001)
    compressed = compress_delta(delta, cfg)
    # Values below 0.001 should be zeroed
    assert compressed["w"][0].item() == 0.0
    assert compressed["w"][2].item() == 0.0
    # Large values should survive
    assert compressed["w"][1].item() != 0.0
    assert compressed["w"][3].item() != 0.0


# ---------------------------------------------------------------------------
# 8. delta_size_bytes – positive int
# ---------------------------------------------------------------------------

def test_delta_size_bytes_positive():
    a, b = _state_a(), _state_b()
    delta = compute_delta(a, b)
    size = delta_size_bytes(delta)
    assert isinstance(size, int)
    assert size > 0


# ---------------------------------------------------------------------------
# 9. DeltaCheckpointManager – save/load roundtrip (base)
# ---------------------------------------------------------------------------

def test_manager_save_load_base():
    tmpdir = tempfile.mkdtemp()
    cfg = DeltaConfig()
    manager = DeltaCheckpointManager(tmpdir, cfg)

    state = _state_a()
    manager.save_base(state, name="base")
    loaded = manager.load("base")

    for key in state:
        assert torch.allclose(loaded[key].float(), state[key].float())


# ---------------------------------------------------------------------------
# 10. DeltaCheckpointManager – save_delta / load roundtrip
# ---------------------------------------------------------------------------

def test_manager_save_load_delta():
    tmpdir = tempfile.mkdtemp()
    cfg = DeltaConfig(compression="none")
    manager = DeltaCheckpointManager(tmpdir, cfg)

    a, b = _state_a(), _state_b()
    manager.save_base(a, name="base")
    manager.save_delta(b, name="step1")

    reconstructed = manager.load("step1")
    for key in b:
        assert torch.allclose(reconstructed[key].float(), b[key].float(), atol=1e-5), (
            f"Delta load roundtrip failed for key {key!r}"
        )


# ---------------------------------------------------------------------------
# 11. quantize_delta – result dtype is int8
# ---------------------------------------------------------------------------

def test_quantize_delta_dtype_int8():
    a, b = _state_a(), _state_b()
    delta = compute_delta(a, b)
    quantized = quantize_delta(delta, bits=8)

    # Each value is a dict with "quantized" (int8) and "scale" (float)
    for key, val in quantized.items():
        assert "quantized" in val, f"Missing 'quantized' key for {key!r}"
        assert "scale" in val, f"Missing 'scale' key for {key!r}"
        assert val["quantized"].dtype == torch.int8, (
            f"Expected int8 for {key!r}, got {val['quantized'].dtype}"
        )
        assert isinstance(val["scale"], float)
