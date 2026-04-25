"""Tests for FP16 quantizer."""
from __future__ import annotations

import pytest
import torch

from src.quantization.fp16_quantizer import FP16Quantizer


class TestFP16Quantizer:
    def test_quantize_dequantize_roundtrip(self):
        q = FP16Quantizer()
        tensor = [[1.0, 2.0], [3.0, 4.0]]
        half_tensor, scale = q.quantize(tensor)
        dequantized = q.dequantize(half_tensor, scale)

        for orig_row, deq_row in zip(tensor, dequantized):
            for o, d in zip(orig_row, deq_row):
                assert abs(o - d) / max(1.0, abs(o)) < 1e-3

    def test_quantize_returns_half_precision_values(self):
        q = FP16Quantizer()
        half_tensor, _ = q.quantize([[1.5, 2.5], [3.5, 4.5]])
        for row in half_tensor:
            for v in row:
                assert torch.tensor([v]).half().item() == v

    def test_scale_factor_computation(self):
        q = FP16Quantizer()
        _, scale = q.quantize([[100.0, 200.0]])
        expected_scale = 200.0 / 65504.0
        assert abs(scale - expected_scale) < 1e-8

    def test_empty_tensor(self):
        q = FP16Quantizer()
        half_tensor, scale = q.quantize([[]])
        assert half_tensor == [[]]
        assert scale == 1.0

    def test_different_shapes(self):
        q = FP16Quantizer()
        shapes = [
            [[1.0]],
            [[1.0, 2.0, 3.0]],
            [[1.0], [2.0], [3.0]],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        ]
        for tensor in shapes:
            half_tensor, scale = q.quantize(tensor)
            dequantized = q.dequantize(half_tensor, scale)
            assert len(dequantized) == len(tensor)
            for orig_row, deq_row in zip(tensor, dequantized):
                assert len(deq_row) == len(orig_row)

    def test_negative_values(self):
        q = FP16Quantizer()
        tensor = [[-1.0, 0.0], [1.0, -2.0]]
        half_tensor, scale = q.quantize(tensor)
        dequantized = q.dequantize(half_tensor, scale)
        for orig_row, deq_row in zip(tensor, dequantized):
            for o, d in zip(orig_row, deq_row):
                assert abs(o - d) / max(1.0, abs(o)) < 1e-3
