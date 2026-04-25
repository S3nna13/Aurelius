"""INT4 bit-packing utilities for Aurelius.

INT4 (4-bit integer) quantization stores two values per byte.  This module
provides:

* :func:`pack_int4`   — pack a list of int4 values into raw bytes
* :func:`unpack_int4` — unpack bytes back to a list of int4 values
* :class:`Int4Tensor` — lightweight container for a packed INT4 buffer

Packing convention
------------------
``values[2*i]`` occupies the **high nibble** and ``values[2*i+1]`` occupies
the **low nibble** of byte ``i``.  If the value list has an odd length the
final byte is padded with a zero low nibble.
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# Clamp helper
# ---------------------------------------------------------------------------

def _clamp_int4(v: int) -> int:
    """Clamp *v* to the unsigned INT4 range [0, 15]."""
    if v < 0:
        return 0
    if v > 15:
        return 15
    return int(v)


# ---------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------

def pack_int4(values: List[int]) -> bytes:
    """Pack a list of INT4 values into a :class:`bytes` object.

    Each value is clamped to [0, 15].  Pairs of consecutive values are packed
    into a single byte: ``values[2i]`` → high nibble, ``values[2i+1]`` → low
    nibble.  If *values* has an odd length, the last byte is padded with a
    zero in the low nibble position.

    Args:
        values: List of integers to pack.  Need not be pre-clamped.

    Returns:
        Packed bytes of length ``ceil(len(values) / 2)``.
    """
    result: List[int] = []
    i = 0
    while i < len(values):
        high = _clamp_int4(values[i])
        low = _clamp_int4(values[i + 1]) if i + 1 < len(values) else 0
        result.append((high << 4) | low)
        i += 2
    return bytes(result)


def unpack_int4(data: bytes, count: int) -> List[int]:
    """Unpack *count* INT4 values from a :class:`bytes` object.

    Args:
        data:  Packed bytes produced by :func:`pack_int4`.
        count: Number of values to unpack.  Must be ≤ ``len(data) * 2``.

    Returns:
        List of ``count`` integers, each in [0, 15].

    Raises:
        ValueError: If *count* exceeds the number of values that *data* can
                    hold (``len(data) * 2``).
    """
    max_count = len(data) * 2
    if count > max_count:
        raise ValueError(
            f"count={count} exceeds the capacity of {len(data)} bytes "
            f"(max {max_count} values)."
        )
    out: List[int] = []
    for byte in data:
        if len(out) >= count:
            break
        out.append((byte >> 4) & 0x0F)
        if len(out) < count:
            out.append(byte & 0x0F)
    return out


# ---------------------------------------------------------------------------
# Int4Tensor
# ---------------------------------------------------------------------------

class Int4Tensor:
    """Lightweight INT4 tensor backed by packed bytes.

    Args:
        values: List of integers to store.  Each value is clamped to [0, 15]
                before packing.

    Example::

        t = Int4Tensor([3, 7, 15, 0, 5])
        assert len(t) == 5
        assert t.tolist() == [3, 7, 15, 0, 5]
    """

    def __init__(self, values: List[int]) -> None:
        self._count: int = len(values)
        self._data: bytes = pack_int4(values)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> bytes:
        """Raw packed bytes backing this tensor."""
        return self._data

    @property
    def count(self) -> int:
        """Number of INT4 values stored (before packing)."""
        return self._count

    # ------------------------------------------------------------------
    # Sequence-like interface
    # ------------------------------------------------------------------

    def tolist(self) -> List[int]:
        """Unpack and return all values as a list of ints."""
        return unpack_int4(self._data, self._count)

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return f"Int4Tensor(count={self._count}, data={self._data!r})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

INT4_PACKING_REGISTRY: dict[str, type] = {"default": Int4Tensor}
