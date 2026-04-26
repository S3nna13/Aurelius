"""Tests for src/quantization/int4_packing.py — ≥28 tests, stdlib-only."""

from __future__ import annotations

import pytest

from src.quantization.int4_packing import (
    INT4_PACKING_REGISTRY,
    Int4Tensor,
    pack_int4,
    unpack_int4,
)

# ---------------------------------------------------------------------------
# pack_int4
# ---------------------------------------------------------------------------


class TestPackInt4:
    def test_pack_empty(self):
        result = pack_int4([])
        assert result == b""

    def test_pack_single_value_high_nibble(self):
        # value 5 → high nibble, padded low nibble = 0
        result = pack_int4([5])
        assert len(result) == 1
        assert result[0] == 0x50

    def test_pack_pair(self):
        # 3 in high nibble, 7 in low nibble → 0x37
        result = pack_int4([3, 7])
        assert result == bytes([0x37])

    def test_pack_pair_zero_high(self):
        result = pack_int4([0, 15])
        assert result == bytes([0x0F])

    def test_pack_four_values_two_bytes(self):
        result = pack_int4([1, 2, 3, 4])
        assert len(result) == 2
        assert result[0] == 0x12
        assert result[1] == 0x34

    def test_pack_odd_length_pads_zero(self):
        # [5, 6, 7] → bytes [0x56, 0x70]
        result = pack_int4([5, 6, 7])
        assert len(result) == 2
        assert result[0] == 0x56
        assert result[1] == 0x70

    def test_pack_clamp_upper(self):
        # 16 → 15 (0xF)
        result = pack_int4([16])
        assert result[0] == 0xF0

    def test_pack_clamp_lower(self):
        # -1 → 0
        result = pack_int4([-1, 3])
        assert result[0] == 0x03

    def test_pack_clamp_both_nibbles(self):
        # [-5, 20] → [0, 15] → 0x0F
        result = pack_int4([-5, 20])
        assert result[0] == 0x0F

    def test_pack_returns_bytes_type(self):
        result = pack_int4([1, 2])
        assert isinstance(result, bytes)

    def test_pack_max_values(self):
        result = pack_int4([15, 15])
        assert result == bytes([0xFF])

    def test_pack_all_zeros(self):
        result = pack_int4([0, 0, 0, 0])
        assert result == bytes([0x00, 0x00])


# ---------------------------------------------------------------------------
# unpack_int4
# ---------------------------------------------------------------------------


class TestUnpackInt4:
    def test_unpack_empty_bytes_count_zero(self):
        result = unpack_int4(b"", 0)
        assert result == []

    def test_unpack_single_high_nibble(self):
        result = unpack_int4(bytes([0x50]), 1)
        assert result == [5]

    def test_unpack_pair(self):
        result = unpack_int4(bytes([0x37]), 2)
        assert result == [3, 7]

    def test_unpack_four_values(self):
        result = unpack_int4(bytes([0x12, 0x34]), 4)
        assert result == [1, 2, 3, 4]

    def test_unpack_odd_count_ignores_pad(self):
        # packed [5, 6, 7] → [0x56, 0x70]; unpack 3 values
        data = pack_int4([5, 6, 7])
        result = unpack_int4(data, 3)
        assert result == [5, 6, 7]

    def test_unpack_count_exceeds_capacity_raises(self):
        with pytest.raises(ValueError):
            unpack_int4(bytes([0x12]), 3)

    def test_unpack_returns_list(self):
        result = unpack_int4(bytes([0xAB]), 2)
        assert isinstance(result, list)

    def test_unpack_values_in_range(self):
        data = bytes([0xFF, 0x00, 0xA5])
        result = unpack_int4(data, 6)
        for v in result:
            assert 0 <= v <= 15

    def test_unpack_roundtrip_even(self):
        original = [0, 1, 2, 3, 14, 15, 8, 9]
        data = pack_int4(original)
        result = unpack_int4(data, len(original))
        assert result == original

    def test_unpack_roundtrip_odd(self):
        original = [7, 3, 11]
        data = pack_int4(original)
        result = unpack_int4(data, len(original))
        assert result == original


# ---------------------------------------------------------------------------
# Int4Tensor
# ---------------------------------------------------------------------------


class TestInt4Tensor:
    def test_len_empty(self):
        t = Int4Tensor([])
        assert len(t) == 0

    def test_len_single(self):
        t = Int4Tensor([7])
        assert len(t) == 1

    def test_len_multiple(self):
        t = Int4Tensor([1, 2, 3, 4, 5])
        assert len(t) == 5

    def test_count_property(self):
        t = Int4Tensor([1, 2, 3])
        assert t.count == 3

    def test_data_property_type(self):
        t = Int4Tensor([3, 7])
        assert isinstance(t.data, bytes)

    def test_data_property_value(self):
        t = Int4Tensor([3, 7])
        assert t.data == bytes([0x37])

    def test_tolist_empty(self):
        t = Int4Tensor([])
        assert t.tolist() == []

    def test_tolist_single(self):
        t = Int4Tensor([9])
        assert t.tolist() == [9]

    def test_tolist_roundtrip_even(self):
        values = [0, 5, 10, 15]
        t = Int4Tensor(values)
        assert t.tolist() == values

    def test_tolist_roundtrip_odd(self):
        values = [1, 3, 7]
        t = Int4Tensor(values)
        assert t.tolist() == values

    def test_clamp_applied_on_init(self):
        t = Int4Tensor([16, -1])  # → [15, 0]
        assert t.tolist() == [15, 0]

    def test_packed_byte_count_even(self):
        t = Int4Tensor([1, 2, 3, 4])
        assert len(t.data) == 2

    def test_packed_byte_count_odd(self):
        t = Int4Tensor([1, 2, 3])
        assert len(t.data) == 2

    def test_repr_contains_count(self):
        t = Int4Tensor([1, 2])
        assert "count=2" in repr(t)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_exists(self):
        assert isinstance(INT4_PACKING_REGISTRY, dict)

    def test_default_key(self):
        assert "default" in INT4_PACKING_REGISTRY

    def test_default_maps_to_int4tensor(self):
        assert INT4_PACKING_REGISTRY["default"] is Int4Tensor

    def test_registry_instantiable(self):
        cls = INT4_PACKING_REGISTRY["default"]
        obj = cls([1, 2, 3])
        assert isinstance(obj, Int4Tensor)
