"""Tests for INT4 quantizer."""

from __future__ import annotations

from src.quantization.int4_quantizer import INT4Quantizer


class TestINT4Quantizer:
    def test_roundtrip(self):
        q = INT4Quantizer()
        tensor = [[1.0, -2.0, 3.0]]
        quantized, scale = q.quantize(tensor)
        deq = q.dequantize(quantized, scale)
        for orig, dec in zip(tensor[0], deq[0]):
            assert abs(orig - dec) < 0.2

    def test_quantize_bounds(self):
        q = INT4Quantizer()
        qv, _ = q.quantize([[100.0]])
        assert qv[0][0] == 7

    def test_empty(self):
        q = INT4Quantizer()
        qv, scale = q.quantize([[]])
        assert qv == [[]]
        assert scale > 0

    def test_negative_values(self):
        q = INT4Quantizer()
        qv, scale = q.quantize([[-5.0, 0.0, 5.0]])
        assert qv[0][0] < 0
        assert qv[0][2] > 0
