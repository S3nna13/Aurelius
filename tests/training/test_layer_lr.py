"""Tests for layer-wise learning rate decay and warmup scheduling."""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.layer_lr import (
    LayerLRConfig,
    LayerLROptimizer,
    LayerLRTrainer,
    build_layer_param_groups,
    cosine_decay_schedule,
    layer_learning_rates,
    warmup_lr_schedule,
)

# ---------------------------------------------------------------------------
# Tiny model fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture()
def tiny_model(tiny_cfg: AureliusConfig) -> AureliusTransformer:
    return AureliusTransformer(tiny_cfg)


# ---------------------------------------------------------------------------
# 1. LayerLRConfig defaults
# ---------------------------------------------------------------------------


def test_layer_lr_config_defaults():
    cfg = LayerLRConfig()
    assert cfg.base_lr == 1e-4
    assert cfg.decay_factor == 0.9
    assert cfg.min_lr == 1e-6
    assert cfg.warmup_steps == 100
    assert cfg.decay_steps == 1000
    assert cfg.decay_type == "cosine"


# ---------------------------------------------------------------------------
# 2. layer_learning_rates output length == n_layers
# ---------------------------------------------------------------------------


def test_layer_learning_rates_length():
    lrs = layer_learning_rates(n_layers=6, base_lr=1e-3, decay_factor=0.8)
    assert len(lrs) == 6


# ---------------------------------------------------------------------------
# 3. layer_learning_rates top layer has base_lr
# ---------------------------------------------------------------------------


def test_layer_learning_rates_top_is_base_lr():
    base_lr = 1e-3
    lrs = layer_learning_rates(n_layers=5, base_lr=base_lr, decay_factor=0.85)
    assert abs(lrs[-1] - base_lr) < 1e-12, f"Top layer lr {lrs[-1]} != base_lr {base_lr}"


# ---------------------------------------------------------------------------
# 4. layer_learning_rates bottom layer has lowest lr
# ---------------------------------------------------------------------------


def test_layer_learning_rates_bottom_is_lowest():
    lrs = layer_learning_rates(n_layers=5, base_lr=1e-3, decay_factor=0.85, min_lr=1e-9)
    assert lrs[0] == min(lrs), "Bottom layer should have the smallest lr"


# ---------------------------------------------------------------------------
# 5. layer_learning_rates monotonically increasing (bottom to top)
# ---------------------------------------------------------------------------


def test_layer_learning_rates_monotone():
    lrs = layer_learning_rates(n_layers=8, base_lr=1e-3, decay_factor=0.9, min_lr=1e-9)
    for i in range(len(lrs) - 1):
        assert lrs[i] <= lrs[i + 1], f"lr[{i}]={lrs[i]} > lr[{i + 1}]={lrs[i + 1]}: not monotone"


# ---------------------------------------------------------------------------
# 6. layer_learning_rates all >= min_lr
# ---------------------------------------------------------------------------


def test_layer_learning_rates_above_min_lr():
    min_lr = 1e-6
    lrs = layer_learning_rates(n_layers=10, base_lr=1e-4, decay_factor=0.5, min_lr=min_lr)
    for i, lr in enumerate(lrs):
        assert lr >= min_lr, f"lr[{i}]={lr} is below min_lr={min_lr}"


# ---------------------------------------------------------------------------
# 7. warmup_lr_schedule at step=0: returns 0 (or near 0)
# ---------------------------------------------------------------------------


def test_warmup_at_step_zero():
    lr = warmup_lr_schedule(step=0, warmup_steps=100, base_lr=1e-3)
    assert lr == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 8. warmup_lr_schedule at step=warmup_steps: returns base_lr
# ---------------------------------------------------------------------------


def test_warmup_at_warmup_steps():
    base_lr = 3e-4
    lr = warmup_lr_schedule(step=100, warmup_steps=100, base_lr=base_lr)
    assert lr == pytest.approx(base_lr)


# ---------------------------------------------------------------------------
# 9. cosine_decay_schedule before warmup: increases
# ---------------------------------------------------------------------------


def test_cosine_decay_increases_during_warmup():
    lrs = [
        cosine_decay_schedule(
            step=s,
            warmup_steps=10,
            decay_steps=100,
            base_lr=1e-3,
            min_lr=1e-6,
        )
        for s in range(11)
    ]
    # During warmup (steps 0..9) lrs should strictly increase
    for i in range(len(lrs) - 1):
        assert lrs[i] <= lrs[i + 1], (
            f"cosine_decay_schedule not increasing during warmup at step {i}"
        )


# ---------------------------------------------------------------------------
# 10. cosine_decay_schedule after decay_steps: returns min_lr
# ---------------------------------------------------------------------------


def test_cosine_decay_after_decay_steps_returns_min_lr():
    min_lr = 1e-6
    lr = cosine_decay_schedule(
        step=9999,
        warmup_steps=10,
        decay_steps=100,
        base_lr=1e-3,
        min_lr=min_lr,
    )
    assert lr == pytest.approx(min_lr)


# ---------------------------------------------------------------------------
# 11. build_layer_param_groups returns list of dicts
# ---------------------------------------------------------------------------


def test_build_layer_param_groups_returns_list_of_dicts(tiny_model):
    config = LayerLRConfig()
    groups = build_layer_param_groups(tiny_model, config)
    assert isinstance(groups, list)
    for g in groups:
        assert isinstance(g, dict)
        assert "params" in g
        assert "lr" in g


# ---------------------------------------------------------------------------
# 12. build_layer_param_groups number of groups = n_layers + 2 (embed + head)
# ---------------------------------------------------------------------------


def test_build_layer_param_groups_count(tiny_model, tiny_cfg):
    config = LayerLRConfig()
    groups = build_layer_param_groups(tiny_model, config)
    expected = tiny_cfg.n_layers + 2  # embed + n_layers + head
    assert len(groups) == expected, f"Expected {expected} param groups, got {len(groups)}"


# ---------------------------------------------------------------------------
# 13. LayerLROptimizer instantiates
# ---------------------------------------------------------------------------


def test_layer_lr_optimizer_instantiates(tiny_model):
    config = LayerLRConfig(warmup_steps=5, decay_steps=20)
    opt = LayerLROptimizer(tiny_model, config)
    assert opt is not None
    assert opt.optimizer is not None


# ---------------------------------------------------------------------------
# 14. LayerLROptimizer.get_lrs returns list of floats
# ---------------------------------------------------------------------------


def test_layer_lr_optimizer_get_lrs(tiny_model, tiny_cfg):
    config = LayerLRConfig(warmup_steps=5, decay_steps=20)
    opt = LayerLROptimizer(tiny_model, config)
    lrs = opt.get_lrs()
    assert isinstance(lrs, list)
    assert len(lrs) == tiny_cfg.n_layers + 2
    for lr in lrs:
        assert isinstance(lr, float)


# ---------------------------------------------------------------------------
# 15. LayerLRTrainer.train_step returns dict with 'loss'
# ---------------------------------------------------------------------------


def test_layer_lr_trainer_train_step_returns_loss(tiny_model):
    config = LayerLRConfig(warmup_steps=5, decay_steps=20)
    trainer = LayerLRTrainer(tiny_model, config)
    # batch=2, seq=16 — labels will be seq[1:], so model sees 15 tokens
    input_ids = torch.randint(0, 256, (2, 16))
    result = trainer.train_step(input_ids, step=1)
    assert isinstance(result, dict)
    assert "loss" in result
    assert isinstance(result["loss"], float)
    assert result["loss"] >= 0.0


# ---------------------------------------------------------------------------
# 16. LayerLRTrainer: different layers have different LRs
# ---------------------------------------------------------------------------


def test_layer_lr_trainer_different_layers_have_different_lrs(tiny_model, tiny_cfg):
    """With decay_factor < 1 and enough layers, param groups must have distinct LRs."""
    config = LayerLRConfig(
        base_lr=1e-3,
        decay_factor=0.5,  # aggressive decay → clearly different LRs
        warmup_steps=5,
        decay_steps=20,
    )
    opt = LayerLROptimizer(tiny_model, config)
    lrs = opt.get_lrs()
    # With n_layers=2 and decay_factor=0.5 we expect 4 distinct values:
    # embed_lr, layer0_lr, layer1_lr=base_lr, head_lr
    # At minimum the embed group and head group should differ
    assert lrs[0] != lrs[-1], (
        "Embed group lr should differ from head group lr with decay_factor < 1"
    )
