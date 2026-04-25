"""Tests for float compressor."""
from __future__ import annotations

import pytest

from src.compression.float_compressor import FloatCompressor


class TestFloatCompressor:
    def test_compress_decompress_roundtrip(self):
        fc = FloatCompressor(bits=16)
        values = [1.0, 2.5, -3.0, 0.5]
        data = fc.compress(values)
        result = fc.decompress(data, scale=max(abs(v) for v in values) or 1.0, count=len(values))
        for orig, dec in zip(values, result):
            assert abs(orig - dec) < 0.01

    def test_empty_values(self):
        fc = FloatCompressor()
        data = fc.compress([])
        assert data == b""

    def test_negative_values(self):
        fc = FloatCompressor(bits=16)
        values = [-100.0, -50.0, 0.0]
        data = fc.compress(values)
        result = fc.decompress(data, scale=100.0, count=len(values))
        assert result[0] < -50.0
        assert result[2] == 0.0

    def test_different_bit_widths(self):
        fc8 = FloatCompressor(bits=8)
        values = [1.0, 2.0]
        data = fc8.compress(values)
        result = fc8.decompress(data, scale=2.0, count=2)
        assert abs(result[0] - 1.0) < 0.1