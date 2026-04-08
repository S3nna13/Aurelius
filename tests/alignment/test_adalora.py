"""Tests for AdaLoRA: rank-adaptive SVD LoRA (arXiv:2303.10512)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.alignment.adalora import AdaLoRAConfig, AdaLoRALinear, AdaLoRATrainer

# Small dimensions for fast CPU tests
IN = 32
OUT = 64
CFG = AdaLoRAConfig(init_rank=4, target_rank=2)


def make_layer() -> AdaLoRALinear:
    return AdaLoRALinear(in_features=IN, out_features=OUT, cfg=CFG)


# ---------------------------------------------------------------------------
# 1. Output shape
# ---------------------------------------------------------------------------

def test_adalora_linear_output_shape():
    layer = make_layer()
    x = torch.randn(2, 8, IN)  # (B, S, in_features)
    out = layer(x)
    assert out.shape == (2, 8, OUT), f"Expected (2, 8, {OUT}), got {out.shape}"


# ---------------------------------------------------------------------------
# 2. Base weight is frozen
# ---------------------------------------------------------------------------

def test_adalora_linear_base_weight_frozen():
    layer = make_layer()
    assert not layer.weight.requires_grad, "weight should be frozen (requires_grad=False)"


# ---------------------------------------------------------------------------
# 3. P, Q, lambda_vec are trainable
# ---------------------------------------------------------------------------

def test_adalora_linear_pqv_trainable():
    layer = make_layer()
    assert layer.P.requires_grad, "P should be trainable"
    assert layer.Q.requires_grad, "Q should be trainable"
    assert layer.lambda_vec.requires_grad, "lambda_vec should be trainable"


# ---------------------------------------------------------------------------
# 4. Orthogonality loss is non-negative
# ---------------------------------------------------------------------------

def test_orthogonality_loss_nonneg():
    layer = make_layer()
    loss = layer.orthogonality_loss()
    assert loss.item() >= 0.0, f"orthogonality_loss should be >= 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 5. prune_to_budget keeps exactly target_rank active
# ---------------------------------------------------------------------------

def test_prune_to_budget_masks_correct_count():
    layer = make_layer()
    layer.prune_to_budget(2)
    active = layer.sv_mask.sum().item()
    assert active == 2, f"Expected 2 active singular values, got {active}"


# ---------------------------------------------------------------------------
# 6. prune_to_budget keeps the most important singular values
# ---------------------------------------------------------------------------

def test_prune_to_budget_keeps_important():
    layer = make_layer()
    # Manually set importance: indices 1 and 3 are highest
    layer.importance = torch.tensor([0.1, 0.9, 0.2, 0.8])
    layer.prune_to_budget(2)
    mask = layer.sv_mask
    # Indices 1 and 3 should be kept
    assert mask[1].item() == 1.0, "Index 1 (importance=0.9) should be kept"
    assert mask[3].item() == 1.0, "Index 3 (importance=0.8) should be kept"
    assert mask[0].item() == 0.0, "Index 0 (importance=0.1) should be pruned"
    assert mask[2].item() == 0.0, "Index 2 (importance=0.2) should be pruned"


# ---------------------------------------------------------------------------
# 7. update_importance changes importance via EMA after backward
# ---------------------------------------------------------------------------

def test_update_importance_updates_ema():
    layer = make_layer()
    importance_before = layer.importance.clone()

    x = torch.randn(2, IN)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    layer.update_importance(step=1)
    importance_after = layer.importance

    assert not torch.allclose(importance_before, importance_after), (
        "importance should change after update_importance() with non-zero gradients"
    )


# ---------------------------------------------------------------------------
# 8. AdaLoRATrainer.train_step returns the expected metric keys
# ---------------------------------------------------------------------------

def test_adalora_trainer_step_returns_metrics():
    layer = make_layer()
    model = nn.Sequential(layer)
    trainer = AdaLoRATrainer(model=model, adalora_layers=[layer], cfg=CFG, lr=1e-4)

    x = torch.randn(4, IN)
    out = layer(x)
    loss = out.sum()

    metrics = trainer.train_step(loss=loss, step=1)

    assert "loss" in metrics, "metrics should contain 'loss'"
    assert "active_rank" in metrics, "metrics should contain 'active_rank'"
    assert "orth_loss" in metrics, "metrics should contain 'orth_loss'"


# ---------------------------------------------------------------------------
# 9. Rank schedule: correct values at boundaries
# ---------------------------------------------------------------------------

def test_adalora_trainer_rank_schedule():
    layer = make_layer()
    model = nn.Sequential(layer)
    cfg = AdaLoRAConfig(init_rank=4, target_rank=2, pruning_warmup_steps=10, total_steps=100)
    trainer = AdaLoRATrainer(model=model, adalora_layers=[layer], cfg=cfg, lr=1e-4)

    assert trainer.get_current_target_rank(0) == cfg.init_rank, (
        "At step 0, target rank should equal init_rank"
    )
    assert trainer.get_current_target_rank(cfg.total_steps) == cfg.target_rank, (
        f"At step {cfg.total_steps}, target rank should equal target_rank"
    )
    # Beyond total_steps should also return target_rank
    assert trainer.get_current_target_rank(cfg.total_steps + 50) == cfg.target_rank


# ---------------------------------------------------------------------------
# 10. Active rank decreases after pruning steps
# ---------------------------------------------------------------------------

def test_active_rank_decreases():
    cfg = AdaLoRAConfig(
        init_rank=4,
        target_rank=2,
        pruning_warmup_steps=2,
        total_steps=10,
        reg_lambda=0.01,
    )
    layer = AdaLoRALinear(in_features=IN, out_features=OUT, cfg=cfg)
    model = nn.Sequential(layer)
    trainer = AdaLoRATrainer(model=model, adalora_layers=[layer], cfg=cfg, lr=1e-4)

    # Measure active rank at step 0 (before any pruning)
    active_rank_step0 = layer.sv_mask.sum().item()

    # Run several steps past warmup so pruning kicks in
    for step in range(1, 51):
        x = torch.randn(4, IN)
        out = layer(x)
        loss = out.sum()
        metrics = trainer.train_step(loss=loss, step=step)

    active_rank_step50 = metrics["active_rank"]

    assert active_rank_step50 < active_rank_step0, (
        f"Active rank should decrease after pruning: "
        f"step0={active_rank_step0}, step50={active_rank_step50}"
    )
