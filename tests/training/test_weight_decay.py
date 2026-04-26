"""Tests for advanced weight decay strategies."""

from __future__ import annotations

import math

import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.weight_decay import (
    WeightDecayConfig,
    WeightDecayScheduler,
    build_param_groups,
    compute_layer_index,
    count_decayed_params,
    layer_wise_wd,
    scheduled_wd,
    should_decay,
)

# Small config for testing
TEST_CONFIG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


def _make_model() -> AureliusTransformer:
    return AureliusTransformer(TEST_CONFIG)


# --- WeightDecayConfig defaults ---


def test_config_defaults():
    cfg = WeightDecayConfig()
    assert cfg.base_wd == 0.1
    assert cfg.wd_schedule == "constant"
    assert cfg.layer_wise_scaling is False
    assert cfg.layer_scale_factor == 0.9
    assert cfg.exclude_patterns == ["bias", "norm", "embedding"]
    assert cfg.warmup_steps == 100
    assert cfg.total_steps == 10000


# --- should_decay ---


def test_should_decay_false_for_bias():
    assert should_decay("layers.0.attn.bias", ["bias", "norm"]) is False


def test_should_decay_true_for_weight():
    assert should_decay("layers.0.attn.w_q.weight", ["bias", "norm", "embedding"]) is True


# --- compute_layer_index ---


def test_compute_layer_index_extracts_correct_index():
    assert compute_layer_index("layers.5.attn.w_q.weight") == 5
    assert compute_layer_index("layers.0.ffn.w1.weight") == 0
    assert compute_layer_index("layers.23.attn_norm.weight") == 23


def test_compute_layer_index_returns_neg1_for_non_layer():
    assert compute_layer_index("embed.weight") == -1
    assert compute_layer_index("lm_head.weight") == -1
    assert compute_layer_index("final_norm.weight") == -1


# --- layer_wise_wd ---


def test_layer_wise_wd_earlier_layers_higher():
    n_layers = 10
    wd_first = layer_wise_wd(0.1, 0, n_layers, 0.9)
    wd_last = layer_wise_wd(0.1, 9, n_layers, 0.9)
    assert wd_first > wd_last


def test_layer_wise_wd_last_layer_lowest():
    n_layers = 10
    wds = [layer_wise_wd(0.1, i, n_layers, 0.9) for i in range(n_layers)]
    assert wds[-1] == min(wds)
    # Last layer: base_wd * scale_factor^(n_layers-1) = 0.1 * 0.9^9
    expected = 0.1 * (0.9**9)
    assert math.isclose(wds[-1], expected, rel_tol=1e-9)


# --- scheduled_wd ---


def test_scheduled_wd_constant():
    cfg = WeightDecayConfig(wd_schedule="constant")
    assert scheduled_wd(0.1, 0, cfg) == 0.1
    assert scheduled_wd(0.1, 5000, cfg) == 0.1
    assert scheduled_wd(0.1, 10000, cfg) == 0.1


def test_scheduled_wd_cosine_decreases():
    cfg = WeightDecayConfig(wd_schedule="cosine", total_steps=1000)
    wd_start = scheduled_wd(0.1, 0, cfg)
    wd_mid = scheduled_wd(0.1, 500, cfg)
    wd_end = scheduled_wd(0.1, 1000, cfg)
    assert wd_start > wd_mid > wd_end
    assert math.isclose(wd_start, 0.1, rel_tol=1e-9)
    assert math.isclose(wd_end, 0.0, abs_tol=1e-9)


def test_scheduled_wd_linear_warmup_starts_at_zero():
    cfg = WeightDecayConfig(wd_schedule="linear_warmup", warmup_steps=100)
    assert scheduled_wd(0.1, 0, cfg) == 0.0
    # Midway through warmup
    assert math.isclose(scheduled_wd(0.1, 50, cfg), 0.05, rel_tol=1e-9)
    # After warmup
    assert scheduled_wd(0.1, 100, cfg) == 0.1
    assert scheduled_wd(0.1, 500, cfg) == 0.1


# --- build_param_groups ---


def test_build_param_groups_creates_at_least_2_groups():
    model = _make_model()
    cfg = WeightDecayConfig()
    groups = build_param_groups(model, cfg)
    assert len(groups) >= 2


def test_build_param_groups_no_decay_group_has_wd_zero():
    model = _make_model()
    cfg = WeightDecayConfig()
    groups = build_param_groups(model, cfg)
    no_decay = [g for g in groups if g["name"] == "no_decay"]
    assert len(no_decay) == 1
    assert no_decay[0]["weight_decay"] == 0.0


# --- WeightDecayScheduler ---


def test_scheduler_step_updates_wd():
    model = _make_model()
    cfg = WeightDecayConfig(wd_schedule="cosine", total_steps=1000)
    groups = build_param_groups(model, cfg)
    optimizer = torch.optim.AdamW(groups, lr=1e-3)
    scheduler = WeightDecayScheduler(optimizer, cfg)

    initial_wds = scheduler.get_current_wd()
    scheduler.step(500)
    updated_wds = scheduler.get_current_wd()

    # Decay groups should have changed; no_decay should stay 0
    for name, wd in updated_wds.items():
        if "no_decay" in name:
            assert wd == 0.0
        else:
            assert wd < initial_wds[name]


# --- count_decayed_params ---


def test_count_decayed_params_totals():
    model = _make_model()
    cfg = WeightDecayConfig()
    groups = build_param_groups(model, cfg)
    counts = count_decayed_params(groups)

    assert counts["decayed"] > 0
    assert counts["non_decayed"] > 0
    assert counts["total"] == counts["decayed"] + counts["non_decayed"]

    # Total should match model parameter count
    total_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert counts["total"] == total_model_params
