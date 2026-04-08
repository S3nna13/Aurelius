"""Tests for CLIP-style contrastive alignment (symmetric InfoNCE)."""
from __future__ import annotations

import math
import torch
import pytest

from src.training.clip_alignment import (
    CLIPAlignmentConfig,
    CLIPAlignmentLayer,
    CLIPAlignmentTrainer,
    clip_loss,
    contrastive_accuracy,
    hard_negative_mining,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_normalized(B: int, D: int) -> torch.Tensor:
    x = torch.randn(B, D)
    return torch.nn.functional.normalize(x, dim=-1)


def _make_model() -> AureliusTransformer:
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=32,
    )
    return AureliusTransformer(cfg)


# ---------------------------------------------------------------------------
# clip_loss tests
# ---------------------------------------------------------------------------

def test_clip_loss_shape():
    """clip_loss must return a scalar tensor."""
    B, D = 4, 32
    text_e = _rand_normalized(B, D)
    other_e = _rand_normalized(B, D)
    log_temp = torch.tensor(math.log(0.07))

    loss = clip_loss(text_e, other_e, log_temp)

    assert loss.ndim == 0, "clip_loss should return a scalar"
    assert torch.isfinite(loss), "clip_loss should be finite"


def test_clip_loss_decreases_with_alignment():
    """Perfectly aligned embeddings should produce lower loss than random ones."""
    torch.manual_seed(0)
    B, D = 8, 32
    base = _rand_normalized(B, D)
    log_temp = torch.tensor(math.log(0.07))

    # Aligned: text == other (perfect diagonal match)
    aligned_loss = clip_loss(base, base.clone(), log_temp)

    # Misaligned: shuffle the other embeddings
    shuffled = base[torch.randperm(B)]
    random_loss = clip_loss(base, shuffled, log_temp)

    assert aligned_loss.item() < random_loss.item(), (
        f"Aligned loss ({aligned_loss.item():.4f}) should be < random loss ({random_loss.item():.4f})"
    )


def test_clip_loss_diagonal_positive():
    """When embeddings match, the diagonal (correct pair) should have max similarity."""
    torch.manual_seed(1)
    B, D = 6, 32
    embeds = _rand_normalized(B, D)
    log_temp = torch.tensor(math.log(0.07))

    # Compute similarity matrix manually
    sim = embeds @ embeds.T  # (B, B)
    # For each row, the diagonal should be the max similarity
    diag = sim.diagonal()
    row_max = sim.max(dim=1).values

    assert torch.allclose(diag, row_max, atol=1e-5), (
        "Diagonal (correct pairs) should have maximum similarity in each row"
    )


# ---------------------------------------------------------------------------
# contrastive_accuracy tests
# ---------------------------------------------------------------------------

def test_contrastive_accuracy_perfect():
    """Identical normalized embeddings → accuracy = 1.0."""
    B, D = 8, 32
    embeds = _rand_normalized(B, D)

    acc = contrastive_accuracy(embeds, embeds.clone())

    assert acc == 1.0, f"Expected 1.0, got {acc}"


def test_contrastive_accuracy_random():
    """Random embeddings should not crash and return value in [0, 1]."""
    torch.manual_seed(5)
    B, D = 16, 32
    text_e = _rand_normalized(B, D)
    other_e = _rand_normalized(B, D)

    acc = contrastive_accuracy(text_e, other_e)

    assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"
    assert isinstance(acc, float), "contrastive_accuracy should return float"


# ---------------------------------------------------------------------------
# CLIPAlignmentLayer tests
# ---------------------------------------------------------------------------

def _make_layer(
    text_dim: int = 64,
    modality_dim: int = 32,
    embedding_dim: int = 16,
    learnable_temp: bool = True,
) -> CLIPAlignmentLayer:
    cfg = CLIPAlignmentConfig(
        temperature_init=0.07,
        temperature_learnable=learnable_temp,
        embedding_dim=embedding_dim,
    )
    return CLIPAlignmentLayer(text_dim=text_dim, modality_dim=modality_dim, cfg=cfg)


def test_clip_alignment_layer_temperature_positive():
    """temperature property must always be > 0."""
    layer = _make_layer()
    assert layer.temperature.item() > 0.0, "temperature must be positive"


def test_clip_alignment_layer_learnable_temp():
    """When temperature_learnable=True, log_temperature must be an nn.Parameter."""
    layer = _make_layer(learnable_temp=True)
    assert isinstance(layer.log_temperature, torch.nn.Parameter), (
        "log_temperature should be an nn.Parameter when learnable=True"
    )
    # When not learnable
    layer_fixed = _make_layer(learnable_temp=False)
    assert not isinstance(layer_fixed.log_temperature, torch.nn.Parameter), (
        "log_temperature should NOT be an nn.Parameter when learnable=False"
    )


def test_clip_alignment_layer_output_shapes():
    """project_text and project_modality must return (B, embedding_dim)."""
    B, text_dim, modality_dim, emb_dim = 4, 64, 32, 16
    layer = _make_layer(text_dim=text_dim, modality_dim=modality_dim, embedding_dim=emb_dim)

    text_h = torch.randn(B, text_dim)
    mod_f = torch.randn(B, modality_dim)

    text_proj = layer.project_text(text_h)
    mod_proj = layer.project_modality(mod_f)

    assert text_proj.shape == (B, emb_dim), f"text_proj shape {text_proj.shape} != ({B}, {emb_dim})"
    assert mod_proj.shape == (B, emb_dim), f"mod_proj shape {mod_proj.shape} != ({B}, {emb_dim})"


def test_clip_alignment_layer_normalized():
    """project_text and project_modality outputs should have L2 norm ≈ 1.0."""
    B, text_dim, modality_dim, emb_dim = 4, 64, 32, 16
    layer = _make_layer(text_dim=text_dim, modality_dim=modality_dim, embedding_dim=emb_dim)

    text_h = torch.randn(B, text_dim)
    mod_f = torch.randn(B, modality_dim)

    text_proj = layer.project_text(text_h)
    mod_proj = layer.project_modality(mod_f)

    text_norms = text_proj.norm(dim=-1)
    mod_norms = mod_proj.norm(dim=-1)

    assert torch.allclose(text_norms, torch.ones(B), atol=1e-5), (
        f"text_proj norms not ≈ 1: {text_norms}"
    )
    assert torch.allclose(mod_norms, torch.ones(B), atol=1e-5), (
        f"mod_proj norms not ≈ 1: {mod_norms}"
    )


def test_clip_alignment_layer_forward_returns_loss():
    """forward() must return a 3-tuple (text_proj, mod_proj, loss) with scalar loss."""
    B, text_dim, modality_dim, emb_dim = 4, 64, 32, 16
    layer = _make_layer(text_dim=text_dim, modality_dim=modality_dim, embedding_dim=emb_dim)

    text_h = torch.randn(B, text_dim)
    mod_f = torch.randn(B, modality_dim)

    result = layer(text_h, mod_f)

    assert isinstance(result, tuple) and len(result) == 3, (
        "forward() must return a 3-tuple"
    )
    text_proj, mod_proj, loss = result
    assert text_proj.shape == (B, emb_dim)
    assert mod_proj.shape == (B, emb_dim)
    assert loss.ndim == 0, "loss must be a scalar"
    assert torch.isfinite(loss), "loss must be finite"


# ---------------------------------------------------------------------------
# CLIPAlignmentTrainer tests
# ---------------------------------------------------------------------------

def test_clip_alignment_trainer_train_step_metrics():
    """train_step must return dict with 'loss', 'accuracy', 'temperature' keys."""
    torch.manual_seed(7)
    model = _make_model()
    d_model = model.config.d_model  # 64

    cfg = CLIPAlignmentConfig(embedding_dim=16, temperature_learnable=True)
    layer = CLIPAlignmentLayer(text_dim=d_model, modality_dim=32, cfg=cfg)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    trainer = CLIPAlignmentTrainer(model=model, alignment_layer=layer, optimizer=optimizer, cfg=cfg)

    B, S, modality_dim = 4, 8, 32
    input_ids = torch.randint(0, 256, (B, S))
    modality_features = torch.randn(B, modality_dim)

    metrics = trainer.train_step(input_ids, modality_features)

    assert "loss" in metrics, "metrics must contain 'loss'"
    assert "accuracy" in metrics, "metrics must contain 'accuracy'"
    assert "temperature" in metrics, "metrics must contain 'temperature'"

    assert math.isfinite(metrics["loss"]), "loss should be finite"
    assert 0.0 <= metrics["accuracy"] <= 1.0, "accuracy out of range"
    assert metrics["temperature"] > 0.0, "temperature must be positive"


# ---------------------------------------------------------------------------
# hard_negative_mining tests
# ---------------------------------------------------------------------------

def test_hard_negative_mining_shape():
    """Output batch must be larger than input batch by n_hard_negatives per sample."""
    torch.manual_seed(9)
    B, D = 8, 32
    text_e = _rand_normalized(B, D)
    other_e = _rand_normalized(B, D)

    n_hard = 2
    hard_text, hard_other = hard_negative_mining(text_e, other_e, n_hard_negatives=n_hard)

    # Each sample contributes n_hard hard negatives
    expected_size = B * n_hard
    assert hard_text.shape == (expected_size, D), (
        f"hard_text shape {hard_text.shape} != ({expected_size}, {D})"
    )
    assert hard_other.shape == (expected_size, D), (
        f"hard_other shape {hard_other.shape} != ({expected_size}, {D})"
    )
