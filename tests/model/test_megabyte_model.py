"""Tests for src/model/megabyte_model.py"""

from __future__ import annotations

import pytest
import torch

from src.model.megabyte_model import MegabyteConfig, MegabyteModel, PatchEmbedding


def _small_cfg(patch_size: int = 4) -> MegabyteConfig:
    return MegabyteConfig(
        patch_size=patch_size,
        global_d_model=64,
        global_n_layers=2,
        global_n_heads=4,
        local_d_model=32,
        local_n_layers=2,
        local_n_heads=4,
        vocab_size=256,
        max_seq_len=64,
    )


def test_output_logits_shape():
    """logits should be (B, T, vocab_size)."""
    cfg = _small_cfg(patch_size=4)
    model = MegabyteModel(cfg)
    B, T = 2, 16
    ids = torch.randint(0, 256, (B, T))
    loss, logits = model(ids)
    assert logits.shape == (B, T, cfg.vocab_size)


def test_loss_is_scalar():
    """Returned loss must be a scalar tensor."""
    cfg = _small_cfg()
    model = MegabyteModel(cfg)
    ids = torch.randint(0, 256, (2, 16))
    loss, _ = model(ids)
    assert loss.shape == ()


def test_loss_backward():
    """loss.backward() must not raise and should populate gradients."""
    cfg = _small_cfg()
    model = MegabyteModel(cfg)
    ids = torch.randint(0, 256, (1, 16))
    loss, _ = model(ids)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_gradient_on_embeddings():
    """Gradients should flow back to the byte embedding table."""
    cfg = _small_cfg()
    model = MegabyteModel(cfg)
    ids = torch.randint(0, 256, (1, 8))
    loss, _ = model(ids)
    loss.backward()
    assert model.byte_embed.weight.grad is not None


def test_patch_size_1():
    """patch_size=1 is a degenerate but valid configuration."""
    cfg = _small_cfg(patch_size=1)
    model = MegabyteModel(cfg)
    ids = torch.randint(0, 256, (1, 8))
    loss, logits = model(ids)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert loss.shape == ()


def test_patch_size_4():
    """patch_size=4 standard case."""
    cfg = _small_cfg(patch_size=4)
    model = MegabyteModel(cfg)
    ids = torch.randint(0, 256, (2, 16))
    loss, logits = model(ids)
    assert logits.shape == (2, 16, cfg.vocab_size)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size: int):
    """Model should handle batch sizes 1, 2, and 4."""
    cfg = _small_cfg(patch_size=4)
    model = MegabyteModel(cfg)
    ids = torch.randint(0, 256, (batch_size, 16))
    loss, logits = model(ids)
    assert logits.shape == (batch_size, 16, cfg.vocab_size)
    assert loss.shape == ()


def test_output_dtype_float32():
    """Logits and loss must be float32."""
    cfg = _small_cfg()
    model = MegabyteModel(cfg)
    ids = torch.randint(0, 256, (1, 8))
    loss, logits = model(ids)
    assert logits.dtype == torch.float32
    assert loss.dtype == torch.float32


def test_forward_determinism():
    """Two forward passes with same input should return identical results."""
    cfg = _small_cfg()
    model = MegabyteModel(cfg)
    model.eval()
    ids = torch.randint(0, 256, (2, 16))
    loss1, logits1 = model(ids)
    loss2, logits2 = model(ids)
    assert torch.allclose(logits1, logits2)
    assert torch.isclose(loss1, loss2)


def test_vocab_size_256_logits_range():
    """Logits last dimension must equal vocab_size=256."""
    cfg = _small_cfg()
    assert cfg.vocab_size == 256
    model = MegabyteModel(cfg)
    ids = torch.randint(0, 256, (1, 8))
    _, logits = model(ids)
    assert logits.shape[-1] == 256
