"""Lossless tensor packing: bit-pack, run-length, delta encoding, and combined."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch


class PackingStrategy(str, Enum):
    BIT_PACK = "bit_pack"
    RUN_LENGTH = "run_length"
    DELTA_ENCODE = "delta_encode"
    COMBINED = "combined"


class LosslessPacker:
    """Pack and unpack tensors using lossless compression strategies.

    Strategies
    ----------
    BIT_PACK     : pack int4 weights into uint8 (2 per byte)
    RUN_LENGTH   : run-length encode repeated scalar values
    DELTA_ENCODE : store first element then successive differences
    COMBINED     : BIT_PACK then RUN_LENGTH on the resulting bytes
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pack(self, tensor: torch.Tensor, strategy: PackingStrategy) -> bytes:
        """Serialise *tensor* using *strategy* and return raw bytes."""
        if strategy == PackingStrategy.BIT_PACK:
            return self._bit_pack(tensor)
        elif strategy == PackingStrategy.RUN_LENGTH:
            return self._run_length_pack(tensor)
        elif strategy == PackingStrategy.DELTA_ENCODE:
            return self._delta_pack(tensor)
        elif strategy == PackingStrategy.COMBINED:
            return self._combined_pack(tensor)
        else:
            raise ValueError(f"Unknown PackingStrategy: {strategy}")

    def unpack(
        self,
        data: bytes,
        strategy: PackingStrategy,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Reconstruct a tensor from *data* produced by :meth:`pack`."""
        if strategy == PackingStrategy.BIT_PACK:
            return self._bit_unpack(data, dtype, shape)
        elif strategy == PackingStrategy.RUN_LENGTH:
            return self._run_length_unpack(data, dtype, shape)
        elif strategy == PackingStrategy.DELTA_ENCODE:
            return self._delta_unpack(data, dtype, shape)
        elif strategy == PackingStrategy.COMBINED:
            return self._combined_unpack(data, dtype, shape)
        else:
            raise ValueError(f"Unknown PackingStrategy: {strategy}")

    # ------------------------------------------------------------------
    # BIT_PACK: clamp to [0,15] and pack two int4 values per byte
    # ------------------------------------------------------------------

    def _bit_pack(self, tensor: torch.Tensor) -> bytes:
        """Pack values clamped to [0, 15] as int4 pairs into uint8 bytes."""
        flat = tensor.reshape(-1)
        # clamp to [0, 15] to fit in 4 bits
        clamped = flat.to(torch.int32).clamp(0, 15).tolist()
        n = len(clamped)
        # header: original element count (uint32)
        header = struct.pack(">I", n)
        # pad to even length
        if n % 2 == 1:
            clamped.append(0)
        packed = bytearray()
        for i in range(0, len(clamped), 2):
            hi = clamped[i] & 0xF
            lo = clamped[i + 1] & 0xF
            packed.append((hi << 4) | lo)
        return header + bytes(packed)

    def _bit_unpack(
        self, data: bytes, dtype: torch.dtype, shape: tuple[int, ...]
    ) -> torch.Tensor:
        n = struct.unpack(">I", data[:4])[0]
        raw = data[4:]
        values: list[int] = []
        for byte in raw:
            values.append((byte >> 4) & 0xF)
            values.append(byte & 0xF)
        values = values[:n]
        return torch.tensor(values, dtype=dtype).reshape(shape)

    # ------------------------------------------------------------------
    # RUN_LENGTH: (value, count) pairs
    # ------------------------------------------------------------------

    def _run_length_pack(self, tensor: torch.Tensor) -> bytes:
        flat = tensor.reshape(-1).tolist()
        if not flat:
            return struct.pack(">I", 0)
        runs: list[tuple[Any, int]] = []
        cur = flat[0]
        cnt = 1
        for v in flat[1:]:
            if v == cur:
                cnt += 1
            else:
                runs.append((cur, cnt))
                cur = v
                cnt = 1
        runs.append((cur, cnt))

        # encode: n_elements (uint32) | n_runs (uint32) | pairs as float32, uint32
        n_elements = len(flat)
        n_runs = len(runs)
        header = struct.pack(">II", n_elements, n_runs)
        body = bytearray()
        for val, count in runs:
            body += struct.pack(">fI", float(val), count)
        return header + bytes(body)

    def _run_length_unpack(
        self, data: bytes, dtype: torch.dtype, shape: tuple[int, ...]
    ) -> torch.Tensor:
        n_elements, n_runs = struct.unpack(">II", data[:8])
        offset = 8
        values: list[float] = []
        for _ in range(n_runs):
            val, count = struct.unpack(">fI", data[offset: offset + 8])
            offset += 8
            values.extend([val] * count)
        values = values[:n_elements]
        return torch.tensor(values, dtype=dtype).reshape(shape)

    # ------------------------------------------------------------------
    # DELTA_ENCODE: first element + deltas
    # ------------------------------------------------------------------

    def _delta_pack(self, tensor: torch.Tensor) -> bytes:
        flat = tensor.reshape(-1).tolist()
        n = len(flat)
        header = struct.pack(">I", n)
        if n == 0:
            return header
        first = struct.pack(">f", float(flat[0]))
        deltas = bytearray()
        for i in range(1, n):
            deltas += struct.pack(">f", float(flat[i] - flat[i - 1]))
        return header + first + bytes(deltas)

    def _delta_unpack(
        self, data: bytes, dtype: torch.dtype, shape: tuple[int, ...]
    ) -> torch.Tensor:
        (n,) = struct.unpack(">I", data[:4])
        if n == 0:
            return torch.zeros(shape, dtype=dtype)
        (first,) = struct.unpack(">f", data[4:8])
        values = [first]
        offset = 8
        for _ in range(n - 1):
            (delta,) = struct.unpack(">f", data[offset: offset + 4])
            offset += 4
            values.append(values[-1] + delta)
        return torch.tensor(values, dtype=dtype).reshape(shape)

    # ------------------------------------------------------------------
    # COMBINED: BIT_PACK then RUN_LENGTH on the raw bytes
    # ------------------------------------------------------------------

    def _combined_pack(self, tensor: torch.Tensor) -> bytes:
        bit_packed = self._bit_pack(tensor)
        # treat bytes as uint8 tensor then run-length encode
        byte_tensor = torch.tensor(list(bit_packed), dtype=torch.float32)
        rle_of_bytes = self._run_length_pack(byte_tensor)
        # prefix with original byte count (uint32)
        n_bytes = len(bit_packed)
        return struct.pack(">I", n_bytes) + rle_of_bytes

    def _combined_unpack(
        self, data: bytes, dtype: torch.dtype, shape: tuple[int, ...]
    ) -> torch.Tensor:
        (n_bytes,) = struct.unpack(">I", data[:4])
        rle_data = data[4:]
        byte_tensor = self._run_length_unpack(
            rle_data, torch.float32, (n_bytes,)
        )
        bit_packed_bytes = bytes(byte_tensor.to(torch.uint8).tolist())
        return self._bit_unpack(bit_packed_bytes, dtype, shape)
