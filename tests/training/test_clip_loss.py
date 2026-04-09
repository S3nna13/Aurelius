"""Tests for CLIP-style contrastive image-text alignment loss and utilities."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.clip_loss import (
    CLIPConfig,
    LearnableTemperature,
    CLIPProjection,
    clip_contrastive_loss,
    compute_retrieval_metrics,
    CLIPModel,
    CLIPTrainer,
)

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

B = 4
EMBED_DIM = 32
D_VISION = 64
D_TEXT = 64


def _make_config() -> CLIPConfig:
    return CLIPConfig(
        temperature=0.07,
        embed_dim=EMBED_DIM,
        d_vision=D_VISION,
        d_text=D_TEXT,
        normalize=True,
    )


def _make_clip_model() -> CLIPModel:
    torch.manual_seed(0)
    vision_encoder = nn.Linear(D_VISION, D_VISION)
    text_encoder = nn.Identity()   # not directly used; encode_text mean-pools token_embeddings
    cfg = _make_config()
    return CLIPModel(vision_encoder=vision_encoder, text_encoder=text_encoder, config=cfg)


def _rand_normalized(b: int, d: int) -> torch.Tensor:
    x = torch.randn(b, d)
    return nn.functional.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# 1. test_clip_config_defaults
# ---------------------------------------------------------------------------

def test_clip_config_defaults():
    """CLIPConfig should have the correct default field values."""
    cfg = CLIPConfig()
    assert cfg.temperature == 0.07
    assert cfg.embed_dim == 128
    assert cfg.d_vision == 256
    assert cfg.d_text == 64
    assert cfg.normalize is True


# ---------------------------------------------------------------------------
# 2. test_learnable_temperature_positive
# ---------------------------------------------------------------------------

def test_learnable_temperature_positive():
    """LearnableTemperature forward() must always return a positive value."""
    lt = LearnableTemperature(init_temp=0.07)
    temp = lt()
    assert temp.item() > 0.0, f"Expected positive temperature, got {temp.item()}"


# ---------------------------------------------------------------------------
# 3. test_learnable_temperature_gradient
# ---------------------------------------------------------------------------

def test_learnable_temperature_gradient():
    """Gradient must flow through LearnableTemperature."""
    lt = LearnableTemperature(init_temp=0.07)
    temp = lt()
    loss = temp * 2.0
    loss.backward()
    assert lt.log_temp.grad is not None, "log_temp should have a gradient"
    assert lt.log_temp.grad.isfinite(), "gradient should be finite"


# ---------------------------------------------------------------------------
# 4. test_clip_projection_output_shape
# ---------------------------------------------------------------------------

def test_clip_projection_output_shape():
    """CLIPProjection must map (B, in_dim) -> (B, embed_dim)."""
    torch.manual_seed(0)
    proj = CLIPProjection(in_dim=D_VISION, embed_dim=EMBED_DIM)
    x = torch.randn(B, D_VISION)
    out = proj(x)
    assert out.shape == (B, EMBED_DIM), f"Expected ({B}, {EMBED_DIM}), got {out.shape}"


# ---------------------------------------------------------------------------
# 5. test_clip_contrastive_loss_scalar
# ---------------------------------------------------------------------------

def test_clip_contrastive_loss_scalar():
    """clip_contrastive_loss must return a finite scalar tensor."""
    torch.manual_seed(0)
    img_e = _rand_normalized(B, EMBED_DIM)
    txt_e = _rand_normalized(B, EMBED_DIM)
    loss = clip_contrastive_loss(img_e, txt_e, temperature=0.07)
    assert loss.ndim == 0, "clip_contrastive_loss should return a scalar"
    assert torch.isfinite(loss), "Loss should be finite"


# ---------------------------------------------------------------------------
# 6. test_clip_contrastive_loss_identity
# ---------------------------------------------------------------------------

def test_clip_contrastive_loss_identity():
    """When image_embeds == text_embeds (diagonal sim matrix), loss should be low."""
    torch.manual_seed(0)
    embeds = _rand_normalized(B, EMBED_DIM)

    # Identical embeddings: diagonal is highest — low loss
    identity_loss = clip_contrastive_loss(embeds, embeds.clone(), temperature=0.07)

    # Shuffled (mismatched): higher loss expected
    shuffled = embeds[torch.randperm(B)]
    random_loss = clip_contrastive_loss(embeds, shuffled, temperature=0.07)

    assert identity_loss.item() < random_loss.item(), (
        f"Identity loss ({identity_loss.item():.4f}) should be less than "
        f"random loss ({random_loss.item():.4f})"
    )


# ---------------------------------------------------------------------------
# 7. test_clip_contrastive_loss_symmetry
# ---------------------------------------------------------------------------

def test_clip_contrastive_loss_symmetry():
    """Swapping image and text embeddings should yield the same loss."""
    torch.manual_seed(0)
    img_e = _rand_normalized(B, EMBED_DIM)
    txt_e = _rand_normalized(B, EMBED_DIM)

    loss_fwd = clip_contrastive_loss(img_e, txt_e, temperature=0.07)
    loss_rev = clip_contrastive_loss(txt_e, img_e, temperature=0.07)

    assert torch.allclose(loss_fwd, loss_rev, atol=1e-5), (
        f"Symmetric loss expected but got {loss_fwd.item()} vs {loss_rev.item()}"
    )


# ---------------------------------------------------------------------------
# 8. test_retrieval_metrics_perfect
# ---------------------------------------------------------------------------

def test_retrieval_metrics_perfect():
    """Identical embeddings should yield R@1 = 1.0 for both directions."""
    torch.manual_seed(0)
    embeds = _rand_normalized(B, EMBED_DIM)
    metrics = compute_retrieval_metrics(embeds, embeds.clone())

    assert metrics["i2t_r1"] == 1.0, f"i2t_r1 expected 1.0, got {metrics['i2t_r1']}"
    assert metrics["t2i_r1"] == 1.0, f"t2i_r1 expected 1.0, got {metrics['t2i_r1']}"
    assert metrics["mean_r1"] == 1.0, f"mean_r1 expected 1.0, got {metrics['mean_r1']}"


# ---------------------------------------------------------------------------
# 9. test_retrieval_metrics_keys
# ---------------------------------------------------------------------------

def test_retrieval_metrics_keys():
    """compute_retrieval_metrics must return the expected set of keys."""
    torch.manual_seed(0)
    img_e = _rand_normalized(B, EMBED_DIM)
    txt_e = _rand_normalized(B, EMBED_DIM)
    metrics = compute_retrieval_metrics(img_e, txt_e)

    expected_keys = {"i2t_r1", "i2t_r5", "t2i_r1", "t2i_r5", "mean_r1"}
    assert set(metrics.keys()) == expected_keys, (
        f"Expected keys {expected_keys}, got {set(metrics.keys())}"
    )
    # All values should be floats in [0, 1]
    for k, v in metrics.items():
        assert isinstance(v, float), f"{k} should be float, got {type(v)}"
        assert 0.0 <= v <= 1.0, f"{k} = {v} out of [0, 1]"


# ---------------------------------------------------------------------------
# 10. test_clip_model_encode_image_shape
# ---------------------------------------------------------------------------

def test_clip_model_encode_image_shape():
    """encode_image must return (B, embed_dim)."""
    torch.manual_seed(0)
    model = _make_clip_model()
    images = torch.randn(B, D_VISION)
    out = model.encode_image(images)
    assert out.shape == (B, EMBED_DIM), f"Expected ({B}, {EMBED_DIM}), got {out.shape}"


# ---------------------------------------------------------------------------
# 11. test_clip_model_encode_text_shape
# ---------------------------------------------------------------------------

def test_clip_model_encode_text_shape():
    """encode_text must return (B, embed_dim) from (B, T, D) token embeddings."""
    torch.manual_seed(0)
    model = _make_clip_model()
    T = 8
    token_embeddings = torch.randn(B, T, D_TEXT)
    out = model.encode_text(token_embeddings)
    assert out.shape == (B, EMBED_DIM), f"Expected ({B}, {EMBED_DIM}), got {out.shape}"


# ---------------------------------------------------------------------------
# 12. test_clip_trainer_step_keys
# ---------------------------------------------------------------------------

def test_clip_trainer_step_keys():
    """CLIPTrainer.train_step must return dict containing 'loss' and 'temperature'."""
    torch.manual_seed(0)
    model = _make_clip_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = CLIPTrainer(model=model, optimizer=optimizer)

    images = torch.randn(B, D_VISION)
    T = 8
    token_embeddings = torch.randn(B, T, D_TEXT)

    result = trainer.train_step(images, token_embeddings)

    assert "loss" in result, "Result must contain 'loss'"
    assert "temperature" in result, "Result must contain 'temperature'"
    assert isinstance(result["loss"], float), "'loss' should be a float"
    assert isinstance(result["temperature"], float), "'temperature' should be a float"
    assert result["temperature"] > 0.0, "temperature must be positive after step"
