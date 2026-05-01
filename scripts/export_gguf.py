#!/usr/bin/env python3
"""safetensors → GGUF converter for Ollama/llama.cpp deployment.

Usage:
    python scripts/export_gguf.py \\
        --checkpoint checkpoints/aurelius-v1 \\
        --output models/aurelius-v1.gguf \\
        --config aurelius_1_3b

Supports:
    - Safetensors checkpoint loading
    - GGUF format v3 output
    - Optional 8-bit quantization during export
    - Metadata embedding (model name, description, parameters)
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
from pathlib import Path
from typing import Any

import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

logger = logging.getLogger(__name__)

# GGUF magic bytes and version
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGUF tensor types
GGUF_TYPE_F32 = 0
GGUF_TYPE_F16 = 1
GGUF_TYPE_Q8_0 = 7  # 8-bit block quantization


def _gguf_type_from_dtype(dtype: torch.dtype, quantize: bool = False) -> int:
    if quantize:
        return GGUF_TYPE_Q8_0
    if dtype == torch.float16:
        return GGUF_TYPE_F16
    return GGUF_TYPE_F32


def _gguf_value(value: Any) -> bytes:
    """Encode a Python value into GGUF metadata bytes."""
    if isinstance(value, bool):
        return struct.pack("<B", 1 if value else 0) + b"\x00" * 3
    if isinstance(value, int):
        if -(2**31) <= value < 2**31:
            return struct.pack("<i", value)
        return struct.pack("<q", value)
    if isinstance(value, float):
        return struct.pack("<f", value)
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return struct.pack("<I", len(encoded)) + encoded
    if isinstance(value, (list, tuple)):
        items = b"".join(_gguf_value(v) for v in value)
        return struct.pack("<I", len(value)) + items
    if isinstance(value, dict):
        return _gguf_value(json.dumps(value))
    return struct.pack("<B", 0) + b"\x00" * 3


def export_gguf(
    checkpoint_dir: str | Path,
    output_path: str | Path,
    config_name: str = "aurelius_1_3b",
    quantize: bool = False,
    model_name: str = "Aurelius",
    description: str = "Aurelius LLM",
) -> None:
    """Export a safetensors checkpoint to GGUF format.

    Args:
        checkpoint_dir: Path to directory containing .safetensors files.
        output_path: Output .gguf file path.
        config_name: AureliusConfig classmethod name (e.g., 'aurelius_1_3b').
        quantize: Whether to apply 8-bit quantization during export.
        model_name: Model name for GGUF metadata.
        description: Description for GGUF metadata.
    """
    ckpt_dir = Path(checkpoint_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load config
    config_fn = getattr(AureliusConfig, config_name)
    config = config_fn()

    # Count total parameters
    total_params = config.n_layers * (
        config.d_model * config.d_ff * 3  # SwiGLU: gate + up + down
        + 4 * config.d_model * config.head_dim * config.n_heads  # Q, K, V, O
    ) + config.vocab_size * config.d_model * 2  # embed + lm_head
    total_params += config.n_layers * config.d_model * 4  # 2x RMSNorm per layer

    # Load state dict
    from safetensors.torch import load_file

    state_dict: dict[str, torch.Tensor] = {}
    for f in sorted(ckpt_dir.glob("*.safetensors")):
        part = load_file(str(f))
        state_dict.update(part)

    logger.info("Loaded %d tensors from %s", len(state_dict), ckpt_dir)

    # Build GGUF file
    with open(output_path, "wb") as f:
        # Header
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", total_params))

        # Metadata key-value pairs
        metadata: dict[str, Any] = {
            "general.name": model_name,
            "general.description": description,
            "general.architecture": "aurelius",
            "general.parameter_count": total_params,
            "general.file_type": 1 if not quantize else 7,
            "aurelius.context_length": config.max_seq_len,
            "aurelius.embedding_length": config.d_model,
            "aurelius.block_count": config.n_layers,
            "aurelius.head_count": config.n_heads,
            "aurelius.head_count_kv": config.n_kv_heads,
            "aurelius.feed_forward_length": config.d_ff,
            "aurelius.rope.dimension_count": config.head_dim,
            "aurelius.rope.freq_base": config.rope_theta,
            "aurelius.vocab_size": config.vocab_size,
            "tokenizer.ggml.model": "bpe",
            "tokenizer.ggml.vocab_size": config.vocab_size,
        }

        f.write(struct.pack("<Q", len(metadata)))
        for key, value in metadata.items():
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<Q", len(key_bytes)) + key_bytes)
            value_bytes = _gguf_value(value)
            if isinstance(value, bool):
                f.write(struct.pack("<I", 0))  # bool type
            elif isinstance(value, int):
                f.write(struct.pack("<I", 2 if value > 2**31 else 1))  # int32/int64
            elif isinstance(value, float):
                f.write(struct.pack("<I", 3))  # float32
            elif isinstance(value, str):
                f.write(struct.pack("<I", 4))  # string
            else:
                f.write(struct.pack("<I", 4))  # fallback string
            f.write(value_bytes)

        # Tensor info
        tensor_keys = list(state_dict.keys())
        f.write(struct.pack("<Q", len(tensor_keys)))

        tensor_offset = 0
        tensor_infos: list[tuple[str, int, list[int], int]] = []

        for name in tensor_keys:
            t = state_dict[name]
            gguf_type = _gguf_type_from_dtype(t.dtype, quantize)
            shape = list(t.shape)
            n_elems = t.numel()
            element_size = 2 if gguf_type == GGUF_TYPE_F16 else 4
            if quantize:
                element_size = 1  # Q8_0 uses 1 byte per element
            tensor_size = n_elems * element_size
            tensor_infos.append((name, gguf_type, shape, tensor_offset))
            tensor_offset += tensor_size

        # Write tensor info
        for name, gguf_type, shape, offset in tensor_infos:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<Q", len(name_bytes)) + name_bytes)
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", gguf_type))
            f.write(struct.pack("<Q", offset))

        # Write tensor data
        for name in tensor_keys:
            t = state_dict[name]
            if quantize:
                # Simple 8-bit quantization per tensor
                scale = t.abs().max() / 127.0
                quantized = (t / scale).round().clamp(-127, 127).to(torch.int8)
                f.write(quantized.numpy().tobytes())
            else:
                t = t.contiguous()
                if t.dtype == torch.float32:
                    f.write(t.numpy().tobytes())
                else:
                    t = t.to(torch.float32)
                    f.write(t.numpy().tobytes())

    logger.info("Exported GGUF to %s (%d tensors, %.2f GB)", output_path, len(tensor_keys), output_path.stat().st_size / (1024**3))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Export Aurelius to GGUF")
    parser.add_argument("--checkpoint", required=True, help="Path to safetensors directory")
    parser.add_argument("--output", required=True, help="Output .gguf file path")
    parser.add_argument("--config", default="aurelius_1_3b", help="Config classmethod name")
    parser.add_argument("--quantize", action="store_true", help="Apply 8-bit quantization")
    parser.add_argument("--name", default="Aurelius", help="Model name in GGUF metadata")
    parser.add_argument("--desc", default="Aurelius LLM", help="Model description")
    args = parser.parse_args()

    export_gguf(
        checkpoint_dir=args.checkpoint,
        output_path=args.output,
        config_name=args.config,
        quantize=args.quantize,
        model_name=args.name,
        description=args.desc,
    )


if __name__ == "__main__":
    main()
