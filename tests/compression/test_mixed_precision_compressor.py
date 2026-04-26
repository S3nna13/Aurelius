"""Tests for mixed_precision_compressor."""
from __future__ import annotations
import torch
from src.compression.mixed_precision_compressor import MixedPrecisionCompressor, compress_tensor
class TestCompressTensor:
    def test_fp16_half_size(self): x=torch.randn(100,100); c=compress_tensor(x,'fp16'); assert c.element_size()==2
    def test_bf16_type(self): c=compress_tensor(torch.randn(10,10),'bf16'); assert c.dtype==torch.bfloat16
    def test_fp32_no_change(self): x=torch.randn(10,10); c=compress_tensor(x,'fp32'); assert c.element_size()==4
class TestMixedPrecisionCompressor:
    def test_compress_model(self): m=torch.nn.Linear(10,10); c=MixedPrecisionCompressor(); c.compress(m,'fp16'); assert next(m.parameters()).dtype==torch.float16
    def test_compress_skip_embedding(self): m=torch.nn.Sequential(torch.nn.Linear(10,10),torch.nn.Embedding(100,32)); c=MixedPrecisionCompressor(skip_embedding=True); c.compress(m,'fp16'); assert m[0].weight.dtype==torch.float16; assert m[1].weight.dtype==torch.float32
