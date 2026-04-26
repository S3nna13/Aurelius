"""Tests for muP scaling rules (mup_scaling.py)."""

from __future__ import annotations

import math

import pytest

from src.optimizers.mup_scaling import MuPConfig, MuPScaler

# ---------------------------------------------------------------------------
# MuPConfig defaults
# ---------------------------------------------------------------------------


def test_mupconfig_default_base_width():
    cfg = MuPConfig()
    assert cfg.base_width == 256


def test_mupconfig_default_base_lr():
    cfg = MuPConfig()
    assert cfg.base_lr == pytest.approx(1e-3)


def test_mupconfig_default_base_init_std():
    cfg = MuPConfig()
    assert cfg.base_init_std == pytest.approx(0.02)


def test_mupconfig_custom_values():
    cfg = MuPConfig(base_width=512, base_lr=3e-4, base_init_std=0.01)
    assert cfg.base_width == 512
    assert cfg.base_lr == pytest.approx(3e-4)
    assert cfg.base_init_std == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# MuPScaler construction
# ---------------------------------------------------------------------------


def test_mupscaler_default_config():
    scaler = MuPScaler()
    assert isinstance(scaler.config, MuPConfig)


def test_mupscaler_custom_config():
    cfg = MuPConfig(base_width=128)
    scaler = MuPScaler(cfg)
    assert scaler.config.base_width == 128


def test_mupscaler_none_config_uses_defaults():
    scaler = MuPScaler(None)
    assert scaler.config.base_width == 256


# ---------------------------------------------------------------------------
# scale_lr
# ---------------------------------------------------------------------------


def test_scale_lr_identity_when_width_equals_base():
    scaler = MuPScaler()
    lr = scaler.scale_lr(1e-3, 256.0, 256.0)
    assert lr == pytest.approx(1e-3)


def test_scale_lr_halves_when_double_width():
    scaler = MuPScaler()
    lr = scaler.scale_lr(1e-3, 512.0, 256.0)
    assert lr == pytest.approx(5e-4)


def test_scale_lr_doubles_when_half_width():
    scaler = MuPScaler()
    lr = scaler.scale_lr(1e-3, 128.0, 256.0)
    assert lr == pytest.approx(2e-3)


def test_scale_lr_proportional():
    scaler = MuPScaler()
    base_lr = 0.01
    lr = scaler.scale_lr(base_lr, 1024.0, 256.0)
    assert lr == pytest.approx(base_lr * 256.0 / 1024.0)


def test_scale_lr_large_width():
    scaler = MuPScaler()
    lr = scaler.scale_lr(1.0, 2048.0, 256.0)
    assert lr == pytest.approx(256.0 / 2048.0)


# ---------------------------------------------------------------------------
# scale_init_std
# ---------------------------------------------------------------------------


def test_scale_init_std_attention_identity():
    scaler = MuPScaler()
    std = scaler.scale_init_std(0.02, 256.0, 256.0, layer_type="attention")
    assert std == pytest.approx(0.02)


def test_scale_init_std_attention_double_width():
    scaler = MuPScaler()
    std = scaler.scale_init_std(0.02, 512.0, 256.0, layer_type="attention")
    assert std == pytest.approx(0.02 * math.sqrt(256.0 / 512.0))


def test_scale_init_std_attention_half_width():
    scaler = MuPScaler()
    std = scaler.scale_init_std(0.02, 128.0, 256.0, layer_type="attention")
    assert std == pytest.approx(0.02 * math.sqrt(256.0 / 128.0))


def test_scale_init_std_output_identity():
    scaler = MuPScaler()
    std = scaler.scale_init_std(0.02, 256.0, 256.0, layer_type="output")
    assert std == pytest.approx(0.02)


def test_scale_init_std_output_double_width():
    scaler = MuPScaler()
    std = scaler.scale_init_std(0.02, 512.0, 256.0, layer_type="output")
    assert std == pytest.approx(0.02 * (256.0 / 512.0))


def test_scale_init_std_output_is_linear_not_sqrt():
    scaler = MuPScaler()
    std_out = scaler.scale_init_std(0.02, 512.0, 256.0, layer_type="output")
    std_att = scaler.scale_init_std(0.02, 512.0, 256.0, layer_type="attention")
    # output decays faster than attention for width > base_width
    assert std_out < std_att


def test_scale_init_std_embedding_unchanged():
    scaler = MuPScaler()
    std = scaler.scale_init_std(0.02, 512.0, 256.0, layer_type="embedding")
    assert std == pytest.approx(0.02)


def test_scale_init_std_embedding_unchanged_large_width():
    scaler = MuPScaler()
    std = scaler.scale_init_std(0.05, 4096.0, 256.0, layer_type="embedding")
    assert std == pytest.approx(0.05)


def test_scale_init_std_unknown_type_uses_sqrt():
    scaler = MuPScaler()
    std = scaler.scale_init_std(0.02, 512.0, 256.0, layer_type="hidden")
    assert std == pytest.approx(0.02 * math.sqrt(256.0 / 512.0))


def test_scale_init_std_default_type_is_attention():
    scaler = MuPScaler()
    std_default = scaler.scale_init_std(0.02, 512.0, 256.0)
    std_attention = scaler.scale_init_std(0.02, 512.0, 256.0, layer_type="attention")
    assert std_default == pytest.approx(std_attention)


# ---------------------------------------------------------------------------
# get_param_groups
# ---------------------------------------------------------------------------


class _FakeParam:
    """Lightweight stand-in for a parameter (no torch dependency needed)."""

    def __init__(self, name: str):
        self.name = name


def _make_named(names):
    return [(n, _FakeParam(n)) for n in names]


def test_get_param_groups_returns_list():
    scaler = MuPScaler(MuPConfig(base_width=256, base_lr=1e-3))
    named = _make_named(["embed.weight", "attn.q_proj", "out_proj.weight", "fc.weight"])
    groups = scaler.get_param_groups(named, width=256.0)
    assert isinstance(groups, list)


def test_get_param_groups_dicts_have_lr():
    scaler = MuPScaler(MuPConfig(base_width=256, base_lr=1e-3))
    named = _make_named(["embed.weight", "attn.q_proj"])
    groups = scaler.get_param_groups(named, width=256.0)
    for g in groups:
        assert "lr" in g


def test_get_param_groups_dicts_have_group_name():
    scaler = MuPScaler(MuPConfig(base_width=256, base_lr=1e-3))
    named = _make_named(["embed.weight", "attn.q_proj"])
    groups = scaler.get_param_groups(named, width=256.0)
    for g in groups:
        assert "group_name" in g


def test_get_param_groups_dicts_have_params():
    scaler = MuPScaler(MuPConfig(base_width=256, base_lr=1e-3))
    named = _make_named(["attn.q_proj"])
    groups = scaler.get_param_groups(named, width=256.0)
    for g in groups:
        assert "params" in g


def test_get_param_groups_embedding_classified():
    scaler = MuPScaler(MuPConfig(base_width=256, base_lr=1e-3))
    named = _make_named(["embed.weight"])
    groups = scaler.get_param_groups(named, width=256.0)
    names = [g["group_name"] for g in groups]
    assert "embedding" in names


def test_get_param_groups_attention_classified():
    scaler = MuPScaler(MuPConfig(base_width=256, base_lr=1e-3))
    named = _make_named(["attn.q_proj"])
    groups = scaler.get_param_groups(named, width=256.0)
    names = [g["group_name"] for g in groups]
    assert "attention" in names


def test_get_param_groups_output_classified():
    scaler = MuPScaler(MuPConfig(base_width=256, base_lr=1e-3))
    named = _make_named(["out_proj.weight"])
    groups = scaler.get_param_groups(named, width=256.0)
    names = [g["group_name"] for g in groups]
    assert "output" in names


def test_get_param_groups_hidden_classified():
    scaler = MuPScaler(MuPConfig(base_width=256, base_lr=1e-3))
    named = _make_named(["fc.weight"])
    groups = scaler.get_param_groups(named, width=256.0)
    names = [g["group_name"] for g in groups]
    assert "hidden" in names


def test_get_param_groups_embedding_lr_unscaled():
    """Embedding group should use base_lr (no muP scaling)."""
    cfg = MuPConfig(base_width=256, base_lr=1e-3)
    scaler = MuPScaler(cfg)
    named = _make_named(["embed.weight"])
    groups = scaler.get_param_groups(named, width=512.0)  # width != base_width
    embed_group = next(g for g in groups if g["group_name"] == "embedding")
    assert embed_group["lr"] == pytest.approx(cfg.base_lr)


def test_get_param_groups_attention_lr_scaled():
    cfg = MuPConfig(base_width=256, base_lr=1e-3)
    scaler = MuPScaler(cfg)
    named = _make_named(["attn.q_proj"])
    groups = scaler.get_param_groups(named, width=512.0)
    attn_group = next(g for g in groups if g["group_name"] == "attention")
    expected = scaler.scale_lr(cfg.base_lr, 512.0, 256.0)
    assert attn_group["lr"] == pytest.approx(expected)


def test_get_param_groups_empty_groups_omitted():
    scaler = MuPScaler()
    named = _make_named(["attn.q_proj"])
    groups = scaler.get_param_groups(named, width=256.0)
    # Only one group should be present (no empty groups)
    assert len(groups) == 1


def test_get_param_groups_multiple_params_same_group():
    scaler = MuPScaler()
    named = _make_named(["attn.q_proj", "attn.k_proj", "attn.v_proj"])
    groups = scaler.get_param_groups(named, width=256.0)
    attn_group = next(g for g in groups if g["group_name"] == "attention")
    assert len(attn_group["params"]) == 3


# ---------------------------------------------------------------------------
# width_multiplier
# ---------------------------------------------------------------------------


def test_width_multiplier_identity():
    scaler = MuPScaler(MuPConfig(base_width=256))
    assert scaler.width_multiplier(256.0) == pytest.approx(1.0)


def test_width_multiplier_double():
    scaler = MuPScaler(MuPConfig(base_width=256))
    assert scaler.width_multiplier(512.0) == pytest.approx(2.0)


def test_width_multiplier_half():
    scaler = MuPScaler(MuPConfig(base_width=256))
    assert scaler.width_multiplier(128.0) == pytest.approx(0.5)


def test_width_multiplier_arbitrary():
    scaler = MuPScaler(MuPConfig(base_width=128))
    assert scaler.width_multiplier(1024.0) == pytest.approx(8.0)
