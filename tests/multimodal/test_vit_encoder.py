import torch
import pytest
from src.multimodal.vit_encoder import (
    ViTConfig,
    ViTPatchEmbedding,
    ViTBlock,
    ViTEncoder,
    VIT_REGISTRY,
)


def make_config(image_size=32, patch_size=8, d_model=64, n_heads=4, n_layers=2):
    return ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )


def test_patch_count_math():
    cfg = make_config(image_size=32, patch_size=8)
    expected = (32 // 8) ** 2
    assert expected == 16


def test_vit_patch_embedding_output_shape():
    cfg = make_config(image_size=32, patch_size=8, d_model=64)
    embed = ViTPatchEmbedding(cfg)
    x = torch.randn(2, 3, 32, 32)
    out = embed(x)
    num_patches = (32 // 8) ** 2
    assert out.shape == (2, num_patches, 64)


def test_vit_block_output_shape():
    cfg = make_config(d_model=64, n_heads=4)
    block = ViTBlock(cfg)
    x = torch.randn(2, 17, 64)
    out = block(x)
    assert out.shape == x.shape


def test_vit_encoder_output_shape():
    cfg = make_config(image_size=32, patch_size=8, d_model=64, n_heads=4, n_layers=2)
    model = ViTEncoder(cfg)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    num_patches = (32 // 8) ** 2
    assert out.shape == (2, 1 + num_patches, 64)


def test_vit_encoder_cls_token_at_index_0():
    cfg = make_config(image_size=32, patch_size=8, d_model=64, n_heads=4, n_layers=2)
    model = ViTEncoder(cfg)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    cls = out[:, 0, :]
    assert cls.shape == (2, 64)


def test_vit_encode_cls_shape():
    cfg = make_config(image_size=32, patch_size=8, d_model=64, n_heads=4, n_layers=2)
    model = ViTEncoder(cfg)
    x = torch.randn(3, 3, 32, 32)
    cls = model.encode_cls(x)
    assert cls.shape == (3, 64)


def test_vit_encoder_default_config():
    model = ViTEncoder()
    assert model.config.variant == "B"


def test_vit_registry_keys():
    assert "ViT-B" in VIT_REGISTRY
    assert "ViT-L" in VIT_REGISTRY


def test_vit_registry_variants():
    assert VIT_REGISTRY["ViT-B"].variant == "B"
    assert VIT_REGISTRY["ViT-L"].variant == "L"


def test_vit_registry_l_config():
    cfg = VIT_REGISTRY["ViT-L"]
    assert cfg.d_model == 1024
    assert cfg.n_heads == 16
    assert cfg.n_layers == 24
