"""Float compression via lossy quantization for storage efficiency."""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass
class FloatCompressor:
    """Compress float values to reduced precision for storage.

    Downsamples 32-bit floats to user-specified bit widths.
    """

    bits: int = 16

    def compress(self, values: list[float]) -> bytes:
        scale = max(abs(v) for v in values) if values else 1.0
        max_int = (1 << (self.bits - 1)) - 1
        data = bytearray()
        for v in values:
            q = int(round(v / scale * max_int))
            q = max(-max_int - 1, min(max_int, q))
            data.extend(struct.pack(">i", q)[-4:])  # pack as 4 bytes
        return bytes(data)

    def decompress(self, data: bytes, scale: float, count: int) -> list[float]:
        max_int = (1 << (self.bits - 1)) - 1
        values = []
        for i in range(count):
            start = (i * 4) % len(data)
            q = struct.unpack(">i", data[start : start + 4])[0]
            values.append(q / max_int * scale)
        return values


FLOAT_COMPRESSOR = FloatCompressor()
