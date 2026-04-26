"""Tests for GGUF format export."""

from __future__ import annotations

import io
import struct

import pytest
import torch
import torch.nn as nn

from src.inference.gguf_export import (
    GGUF_MAGIC,
    GGUF_TENSOR_BF16,
    GGUF_TENSOR_F16,
    GGUF_TENSOR_F32,
    GGUF_TYPE_STRING,
    GGUF_TYPE_UINT32,
    GGUF_VERSION,
    export_to_gguf,
    tensor_to_gguf_type,
    write_kv_string,
    write_kv_uint32,
    write_string,
)

# ---------------------------------------------------------------------------
# Small test model fixture
# ---------------------------------------------------------------------------


def make_small_model() -> nn.Module:
    """Minimal transformer-like model with known parameter shapes."""
    d_model = 64
    d_ff = 128
    vocab_size = 256

    model = nn.ModuleDict(
        {
            "embed": nn.Embedding(vocab_size, d_model),
            "layer0_attn": nn.Linear(d_model, d_model, bias=False),
            "layer0_ffn": nn.Linear(d_model, d_ff, bias=False),
            "layer1_attn": nn.Linear(d_model, d_model, bias=False),
            "layer1_ffn": nn.Linear(d_model, d_ff, bias=False),
            "norm": nn.LayerNorm(d_model),
            "lm_head": nn.Linear(d_model, vocab_size, bias=False),
        }
    )
    return model


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


def test_write_string_format():
    """write_string produces uint64 length prefix followed by UTF-8 bytes."""
    buf = io.BytesIO()
    s = "hello"
    write_string(buf, s)
    data = buf.getvalue()

    # First 8 bytes = uint64 length (little-endian)
    length = struct.unpack_from("<Q", data, 0)[0]
    assert length == len(s.encode("utf-8"))

    # Remaining bytes = UTF-8 string
    assert data[8:] == s.encode("utf-8")


def test_write_kv_string_format():
    """write_kv_string produces key string, type=8 (STRING), then value string."""
    buf = io.BytesIO()
    write_kv_string(buf, "general.name", "aurelius")
    data = buf.getvalue()

    offset = 0

    # Key: uint64 len + bytes
    key_len = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    key_bytes = data[offset : offset + key_len]
    offset += key_len
    assert key_bytes == b"general.name"

    # Type: uint32 = GGUF_TYPE_STRING (8)
    vtype = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    assert vtype == GGUF_TYPE_STRING

    # Value: uint64 len + bytes
    val_len = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    val_bytes = data[offset : offset + val_len]
    assert val_bytes == b"aurelius"


def test_write_kv_uint32_format():
    """write_kv_uint32 produces key string, type=4 (UINT32), then 4-byte value."""
    buf = io.BytesIO()
    write_kv_uint32(buf, "n_layers", 24)
    data = buf.getvalue()

    offset = 0

    # Key
    key_len = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    key_bytes = data[offset : offset + key_len]
    offset += key_len
    assert key_bytes == b"n_layers"

    # Type
    vtype = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    assert vtype == GGUF_TYPE_UINT32

    # Value
    value = struct.unpack_from("<I", data, offset)[0]
    assert value == 24


# ---------------------------------------------------------------------------
# Unit tests for tensor_to_gguf_type
# ---------------------------------------------------------------------------


def test_tensor_to_gguf_type_f32():
    t = torch.zeros(4, 4, dtype=torch.float32)
    assert tensor_to_gguf_type(t) == GGUF_TENSOR_F32


def test_tensor_to_gguf_type_f16():
    t = torch.zeros(4, 4, dtype=torch.float16)
    assert tensor_to_gguf_type(t) == GGUF_TENSOR_F16


def test_tensor_to_gguf_type_bf16():
    t = torch.zeros(4, 4, dtype=torch.bfloat16)
    assert tensor_to_gguf_type(t) == GGUF_TENSOR_BF16


# ---------------------------------------------------------------------------
# Integration tests for export_to_gguf
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model():
    return make_small_model()


@pytest.fixture
def gguf_file(tmp_path, small_model):
    """Export the small model and return (path, stats)."""
    out = tmp_path / "model.gguf"
    stats = export_to_gguf(small_model, out, model_name="test_model", quantize_f16=False)
    return out, stats


def test_export_creates_file(tmp_path, small_model):
    """export_to_gguf creates a file at the specified path."""
    out = tmp_path / "model.gguf"
    assert not out.exists()
    export_to_gguf(small_model, out, quantize_f16=False)
    assert out.exists()


def test_export_magic_number(gguf_file):
    """First 4 bytes of the GGUF file must be the magic number 0x46554747."""
    path, _ = gguf_file
    raw = path.read_bytes()
    magic = struct.unpack_from("<I", raw, 0)[0]
    assert magic == GGUF_MAGIC  # 0x46554747


def test_export_version(gguf_file):
    """Bytes 4-7 must encode GGUF version 3 as uint32 little-endian."""
    path, _ = gguf_file
    raw = path.read_bytes()
    version = struct.unpack_from("<I", raw, 4)[0]
    assert version == GGUF_VERSION  # 3


def test_export_returns_stats(gguf_file):
    """export_to_gguf returns a dict with n_tensors, file_size_bytes, n_kv."""
    _, stats = gguf_file
    assert "n_tensors" in stats
    assert "file_size_bytes" in stats
    assert "n_kv" in stats
    assert isinstance(stats["n_tensors"], int)
    assert isinstance(stats["file_size_bytes"], int)
    assert isinstance(stats["n_kv"], int)


def test_export_file_nonempty(gguf_file):
    """The exported file must be non-empty."""
    _, stats = gguf_file
    assert stats["file_size_bytes"] > 0
