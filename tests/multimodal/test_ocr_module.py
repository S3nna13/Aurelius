import torch
import pytest
from src.multimodal.ocr_module import (
    OCRConfig,
    TextRegionEncoder,
    LayoutAttention,
    OCRModule,
    OCR_MODULE_REGISTRY,
)


def make_image(B=2, H=64, W=64):
    return torch.randn(B, 3, H, W)


def test_text_region_encoder_output_shape():
    cfg = OCRConfig(d_model=32, patch_size=8, stride=4)
    enc = TextRegionEncoder(cfg)
    out = enc(make_image(B=2, H=64, W=64))
    assert out.ndim == 3
    assert out.shape[0] == 2
    assert out.shape[2] == 32


def test_layout_attention_output_shape():
    cfg = OCRConfig(d_model=32, max_text_regions=8)
    attn = LayoutAttention(cfg, n_heads=4)
    queries = torch.randn(2, 8, 32)
    keys = torch.randn(2, 16, 32)
    out = attn(queries, keys)
    assert out.shape == (2, 8, 32)


def test_ocr_module_forward_shape():
    cfg = OCRConfig(d_model=32, max_text_regions=16, patch_size=8, stride=4)
    model = OCRModule(cfg)
    out = model(make_image(B=2, H=64, W=64))
    assert out.shape == (2, 16, 32)


def test_ocr_module_extract_features_alias():
    cfg = OCRConfig(d_model=32, max_text_regions=16, patch_size=8, stride=4)
    model = OCRModule(cfg)
    img = make_image(B=2, H=64, W=64)
    out1 = model(img)
    out2 = model.extract_features(img)
    assert out1.shape == out2.shape


def test_ocr_module_default_config():
    model = OCRModule()
    assert model.config.d_model == 256
    assert model.config.max_text_regions == 64


def test_ocr_module_batch_size_1():
    cfg = OCRConfig(d_model=32, max_text_regions=8, patch_size=8, stride=4)
    model = OCRModule(cfg)
    out = model(make_image(B=1, H=64, W=64))
    assert out.shape == (1, 8, 32)


def test_ocr_registry_key():
    assert "default" in OCR_MODULE_REGISTRY
    assert OCR_MODULE_REGISTRY["default"] is OCRModule
