"""Tests for FP8 quantizer."""

from __future__ import annotations

from src.quantization.fp8_quantizer import FP8Quantizer


class TestFP8Quantizer:
    def test_quantize_dequantize_roundtrip(self):
        q = FP8Quantizer(max_val=448.0)
        tensor = [[1.0, 2.0], [3.0, 4.0]]
        quantized, scale = q.quantize(tensor)
        dequantized = q.dequantize(quantized, scale)

        for orig_row, deq_row in zip(tensor, dequantized):
            for o, d in zip(orig_row, deq_row):
                assert abs(o - d) < 0.01

    def test_quantize_scales_properly(self):
        q = FP8Quantizer()
        _, scale = q.quantize([[100.0]])
        assert scale > 1.0

    def test_empty_tensor(self):
        q = FP8Quantizer()
        quantized, scale = q.quantize([[]])
        assert quantized == [[]]
        assert scale == 448.0
