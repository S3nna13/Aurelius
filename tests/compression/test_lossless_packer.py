"""Tests for src/compression/lossless_packer.py — 8+ tests."""
import pytest
import torch
from src.compression.lossless_packer import LosslessPacker, PackingStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def packer():
    return LosslessPacker()


def _int_tensor(values, shape=None):
    t = torch.tensor(values, dtype=torch.int32)
    if shape:
        t = t.reshape(shape)
    return t


def _float_tensor(values, shape=None):
    t = torch.tensor(values, dtype=torch.float32)
    if shape:
        t = t.reshape(shape)
    return t


# ---------------------------------------------------------------------------
# 1. BIT_PACK
# ---------------------------------------------------------------------------

def test_bit_pack_roundtrip_4x4(packer):
    t = _int_tensor(list(range(16)), shape=(4, 4)).clamp(0, 15)
    data = packer.pack(t, PackingStrategy.BIT_PACK)
    out = packer.unpack(data, PackingStrategy.BIT_PACK, torch.int32, (4, 4))
    assert out.shape == (4, 4)
    assert torch.all(out == t)


def test_bit_pack_odd_number_of_elements(packer):
    t = _int_tensor([3, 7, 1, 5, 15], shape=(5,)).clamp(0, 15)
    data = packer.pack(t, PackingStrategy.BIT_PACK)
    out = packer.unpack(data, PackingStrategy.BIT_PACK, torch.int32, (5,))
    assert torch.all(out == t)


def test_bit_pack_produces_bytes(packer):
    t = _int_tensor([0, 15, 0, 15], shape=(4,))
    data = packer.pack(t, PackingStrategy.BIT_PACK)
    assert isinstance(data, bytes)


# ---------------------------------------------------------------------------
# 2. RUN_LENGTH
# ---------------------------------------------------------------------------

def test_run_length_roundtrip_repeated_values(packer):
    t = _float_tensor([1.0, 1.0, 1.0, 2.0, 2.0, 3.0], shape=(6,))
    data = packer.pack(t, PackingStrategy.RUN_LENGTH)
    out = packer.unpack(data, PackingStrategy.RUN_LENGTH, torch.float32, (6,))
    assert torch.allclose(out, t, atol=1e-5)


def test_run_length_roundtrip_4x4(packer):
    t = torch.zeros(4, 4, dtype=torch.float32)
    data = packer.pack(t, PackingStrategy.RUN_LENGTH)
    out = packer.unpack(data, PackingStrategy.RUN_LENGTH, torch.float32, (4, 4))
    assert out.shape == (4, 4)
    assert torch.allclose(out, t, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. DELTA_ENCODE
# ---------------------------------------------------------------------------

def test_delta_encode_roundtrip_sorted_indices(packer):
    t = _float_tensor([0.0, 1.0, 2.0, 3.0, 4.0], shape=(5,))
    data = packer.pack(t, PackingStrategy.DELTA_ENCODE)
    out = packer.unpack(data, PackingStrategy.DELTA_ENCODE, torch.float32, (5,))
    assert torch.allclose(out, t, atol=1e-4)


def test_delta_encode_roundtrip_4x4(packer):
    t = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    data = packer.pack(t, PackingStrategy.DELTA_ENCODE)
    out = packer.unpack(data, PackingStrategy.DELTA_ENCODE, torch.float32, (4, 4))
    assert out.shape == (4, 4)
    assert torch.allclose(out, t, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. COMBINED
# ---------------------------------------------------------------------------

def test_combined_roundtrip(packer):
    t = _int_tensor(list(range(16)), shape=(4, 4)).clamp(0, 15)
    data = packer.pack(t, PackingStrategy.COMBINED)
    out = packer.unpack(data, PackingStrategy.COMBINED, torch.int32, (4, 4))
    assert out.shape == (4, 4)
    assert torch.all(out == t)


def test_combined_returns_bytes(packer):
    t = _int_tensor([0, 1, 2, 3], shape=(4,)).clamp(0, 15)
    data = packer.pack(t, PackingStrategy.COMBINED)
    assert isinstance(data, bytes)


# ---------------------------------------------------------------------------
# 5. PackingStrategy enum
# ---------------------------------------------------------------------------

def test_packing_strategy_enum_values():
    assert PackingStrategy.BIT_PACK == "bit_pack"
    assert PackingStrategy.RUN_LENGTH == "run_length"
    assert PackingStrategy.DELTA_ENCODE == "delta_encode"
    assert PackingStrategy.COMBINED == "combined"


# ---------------------------------------------------------------------------
# 6. Unknown strategy raises
# ---------------------------------------------------------------------------

def test_unknown_strategy_raises(packer):
    with pytest.raises((ValueError, AttributeError, KeyError)):
        packer.pack(torch.zeros(4), "unknown_strategy")
