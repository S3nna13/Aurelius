"""GGUF format export for llama.cpp compatibility.

Writes model weights to GGUF binary format (version 3).
Enables loading Aurelius models in llama.cpp, ollama, etc.
"""
from __future__ import annotations

import struct
import io
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn


# GGUF constants
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

# Metadata value types
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING  = 8
GGUF_TYPE_UINT64  = 10

# Tensor data types
GGUF_TENSOR_F32  = 0
GGUF_TENSOR_F16  = 1
GGUF_TENSOR_BF16 = 30


def write_string(buf: io.BytesIO, s: str) -> None:
    """Write GGUF string: uint64 length + bytes."""
    encoded = s.encode("utf-8")
    buf.write(struct.pack("<Q", len(encoded)))
    buf.write(encoded)


def write_kv_string(buf: io.BytesIO, key: str, value: str) -> None:
    """Write metadata KV pair with string value."""
    write_string(buf, key)
    buf.write(struct.pack("<I", GGUF_TYPE_STRING))
    write_string(buf, value)


def write_kv_uint32(buf: io.BytesIO, key: str, value: int) -> None:
    """Write metadata KV pair with uint32 value."""
    write_string(buf, key)
    buf.write(struct.pack("<I", GGUF_TYPE_UINT32))
    buf.write(struct.pack("<I", value))


def write_kv_float32(buf: io.BytesIO, key: str, value: float) -> None:
    """Write metadata KV pair with float32 value."""
    write_string(buf, key)
    buf.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
    buf.write(struct.pack("<f", value))


def tensor_to_gguf_type(t: torch.Tensor) -> int:
    """Map tensor dtype to GGUF tensor type constant."""
    if t.dtype == torch.float32:
        return GGUF_TENSOR_F32
    elif t.dtype == torch.float16:
        return GGUF_TENSOR_F16
    elif t.dtype == torch.bfloat16:
        return GGUF_TENSOR_BF16
    else:
        # Convert to float32
        return GGUF_TENSOR_F32


@dataclass
class GGUFTensorInfo:
    """Info about one tensor to write."""
    name: str
    tensor: torch.Tensor
    gguf_type: int


def _collect_tensors(
    model: nn.Module,
    quantize_f16: bool,
) -> list[GGUFTensorInfo]:
    """Collect all named parameters from the model as GGUFTensorInfo entries.

    Parameters that are not float32/float16/bfloat16 are cast to float32 first.
    If quantize_f16 is True, float32 tensors are downcast to float16.
    """
    infos: list[GGUFTensorInfo] = []
    seen_data_ptrs: set[int] = set()

    for name, param in model.named_parameters():
        # Skip tied weights (same storage already recorded)
        data_ptr = param.data_ptr()
        if data_ptr in seen_data_ptrs:
            continue
        seen_data_ptrs.add(data_ptr)

        t = param.detach().cpu()

        # Ensure we have a supported dtype
        if t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            t = t.float()

        if quantize_f16 and t.dtype == torch.float32:
            t = t.half()

        gguf_type = tensor_to_gguf_type(t)
        infos.append(GGUFTensorInfo(name=name, tensor=t, gguf_type=gguf_type))

    return infos


def _build_metadata(
    model: nn.Module,
    model_name: str,
    arch: str,
    buf: io.BytesIO,
) -> int:
    """Write metadata KV pairs to buf; return the count written."""
    kv_count = 0

    # Standard llama.cpp general metadata
    write_kv_string(buf, "general.architecture", arch)
    kv_count += 1

    write_kv_string(buf, "general.name", model_name)
    kv_count += 1

    # Try to extract config attributes if present
    cfg = getattr(model, "config", None)

    if cfg is not None:
        arch_prefix = arch  # e.g. "llama"

        if hasattr(cfg, "n_layers"):
            write_kv_uint32(buf, f"{arch_prefix}.block_count", cfg.n_layers)
            kv_count += 1

        if hasattr(cfg, "d_model"):
            write_kv_uint32(buf, f"{arch_prefix}.embedding_length", cfg.d_model)
            kv_count += 1

        if hasattr(cfg, "d_ff"):
            write_kv_uint32(buf, f"{arch_prefix}.feed_forward_length", cfg.d_ff)
            kv_count += 1

        if hasattr(cfg, "n_heads"):
            write_kv_uint32(buf, f"{arch_prefix}.attention.head_count", cfg.n_heads)
            kv_count += 1

        if hasattr(cfg, "n_kv_heads"):
            write_kv_uint32(buf, f"{arch_prefix}.attention.head_count_kv", cfg.n_kv_heads)
            kv_count += 1

        if hasattr(cfg, "vocab_size"):
            write_kv_uint32(buf, f"{arch_prefix}.vocab_size", cfg.vocab_size)
            kv_count += 1

        if hasattr(cfg, "max_seq_len"):
            write_kv_uint32(buf, f"{arch_prefix}.context_length", cfg.max_seq_len)
            kv_count += 1

        if hasattr(cfg, "rope_theta"):
            write_kv_float32(buf, f"{arch_prefix}.rope.freq_base", cfg.rope_theta)
            kv_count += 1

        if hasattr(cfg, "rms_norm_eps"):
            write_kv_float32(buf, f"{arch_prefix}.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps)
            kv_count += 1

    return kv_count


def export_to_gguf(
    model: nn.Module,
    output_path: str | Path,
    model_name: str = "aurelius",
    arch: str = "llama",    # architecture string for llama.cpp
    quantize_f16: bool = True,   # convert to f16 for smaller files
    alignment: int = 32,
) -> dict[str, int]:
    """Export model to GGUF format.

    Writes:
    1. Header (magic, version, tensor count, kv count)
    2. Metadata KV pairs (model name, arch, vocab size, etc.)
    3. Tensor header entries (name, dims, dtype, offset)
    4. Alignment padding
    5. Tensor data

    Returns dict with stats: {"n_tensors": int, "file_size_bytes": int, "n_kv": int}
    """
    output_path = Path(output_path)

    # --- Collect tensors ---
    tensor_infos = _collect_tensors(model, quantize_f16)
    n_tensors = len(tensor_infos)

    # --- Build metadata section first to know kv count ---
    meta_buf = io.BytesIO()
    n_kv = _build_metadata(model, model_name, arch, meta_buf)
    meta_bytes = meta_buf.getvalue()

    # --- Build tensor info (header entries) section ---
    # We need to compute data offsets, which means we need tensor byte sizes first.
    # Data starts after the full header; we'll compute the base offset once we know
    # all header sizes, then do a second pass for offsets.

    def _tensor_raw_bytes(info: GGUFTensorInfo) -> bytes:
        """Return the raw bytes for tensor data, correctly typed."""
        t = info.tensor
        if info.gguf_type == GGUF_TENSOR_F32:
            t = t.float().contiguous()
        elif info.gguf_type == GGUF_TENSOR_F16:
            t = t.half().contiguous()
        elif info.gguf_type == GGUF_TENSOR_BF16:
            t = t.bfloat16().contiguous()
        else:
            t = t.float().contiguous()
        return t.numpy().tobytes()

    # Pre-compute raw data for each tensor
    tensor_data_list: list[bytes] = [_tensor_raw_bytes(info) for info in tensor_infos]

    def _write_tensor_header_entry(
        buf: io.BytesIO,
        info: GGUFTensorInfo,
        data_offset: int,
    ) -> None:
        """Write a single tensor info entry in the GGUF header."""
        write_string(buf, info.name)
        # n_dims as uint32
        n_dims = info.tensor.ndim
        buf.write(struct.pack("<I", n_dims))
        # dims as uint64 each (GGUF stores dims in reverse: innermost first)
        shape = list(reversed(info.tensor.shape))
        for dim in shape:
            buf.write(struct.pack("<Q", dim))
        # dtype as uint32
        buf.write(struct.pack("<I", info.gguf_type))
        # data offset as uint64
        buf.write(struct.pack("<Q", data_offset))

    # --- Compute size of header entries to determine data section start offset ---
    # We'll do a dry-run to measure tensor header size.
    dry_buf = io.BytesIO()
    fake_offset = 0
    for info in tensor_infos:
        _write_tensor_header_entry(dry_buf, info, fake_offset)
    tensor_headers_size = len(dry_buf.getvalue())

    # Full header = magic(4) + version(4) + tensor_count(8) + kv_count(8)
    #             + metadata_bytes + tensor_headers
    fixed_header_size = 4 + 4 + 8 + 8
    header_total = fixed_header_size + len(meta_bytes) + tensor_headers_size

    # Data section starts at the next alignment boundary after the header
    def _align_up(n: int, align: int) -> int:
        return (n + align - 1) // align * align

    data_section_start = _align_up(header_total, alignment)

    # --- Compute per-tensor data offsets (each tensor also aligned) ---
    offsets: list[int] = []
    current_offset = 0  # relative to data_section_start
    for raw_data in tensor_data_list:
        offsets.append(current_offset)
        current_offset += len(raw_data)
        current_offset = _align_up(current_offset, alignment)

    # --- Write actual tensor header entries with real offsets ---
    th_buf = io.BytesIO()
    for info, off in zip(tensor_infos, offsets):
        _write_tensor_header_entry(th_buf, info, off)
    tensor_headers_bytes = th_buf.getvalue()

    # --- Assemble and write the file ---
    with open(output_path, "wb") as f:
        # 1. Magic + version
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))

        # 2. Counts
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))

        # 3. Metadata KV pairs
        f.write(meta_bytes)

        # 4. Tensor header entries
        f.write(tensor_headers_bytes)

        # 5. Padding to alignment boundary
        current_pos = fixed_header_size + len(meta_bytes) + len(tensor_headers_bytes)
        pad_needed = data_section_start - current_pos
        if pad_needed > 0:
            f.write(b"\x00" * pad_needed)

        # 6. Tensor data (each chunk padded to alignment)
        for raw_data in tensor_data_list:
            f.write(raw_data)
            chunk_pad = _align_up(len(raw_data), alignment) - len(raw_data)
            if chunk_pad > 0:
                f.write(b"\x00" * chunk_pad)

    file_size = output_path.stat().st_size
    return {
        "n_tensors": n_tensors,
        "file_size_bytes": file_size,
        "n_kv": n_kv,
    }
