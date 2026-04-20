"""Integration tests for context_window_extension.

Verifies:
* CONTEXT_EXTENSION_REGISTRY exposed on src.longcontext.
* DynamicContextScaler in LONGCONTEXT_STRATEGY_REGISTRY.
* context_extension_strategy / context_target_len in AureliusConfig.
* End-to-end: construct DynamicContextScaler from AureliusConfig fields,
  compute cos/sin at 1x, 2x, 4x, 8x train_len, verify all finite.
* No prohibited external imports in the implementation file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

import src.longcontext as lc
from src.model.config import AureliusConfig


# ---------------------------------------------------------------------------
# Registry surface tests
# ---------------------------------------------------------------------------

def test_context_extension_registry_exposed() -> None:
    """CONTEXT_EXTENSION_REGISTRY must be importable from src.longcontext."""
    assert hasattr(lc, "CONTEXT_EXTENSION_REGISTRY"), (
        "CONTEXT_EXTENSION_REGISTRY not found on src.longcontext"
    )
    reg = lc.CONTEXT_EXTENSION_REGISTRY
    for key in ("linear", "ntk", "yarn", "longrope"):
        assert key in reg, f"Missing key {key!r} in CONTEXT_EXTENSION_REGISTRY"


def test_dynamic_context_scaler_in_strategy_registry() -> None:
    """DynamicContextScaler must be registered in LONGCONTEXT_STRATEGY_REGISTRY."""
    assert hasattr(lc, "LONGCONTEXT_STRATEGY_REGISTRY")
    reg = lc.LONGCONTEXT_STRATEGY_REGISTRY
    assert "dynamic_context_scaler" in reg, (
        f"'dynamic_context_scaler' not in LONGCONTEXT_STRATEGY_REGISTRY. "
        f"Keys: {list(reg.keys())}"
    )
    assert reg["dynamic_context_scaler"] is lc.DynamicContextScaler


def test_existing_registry_entries_still_present() -> None:
    """Additive changes must not remove previously registered entries."""
    expected = {
        "kv_int8",
        "attention_sinks",
        "ring_attention",
        "context_compaction",
        "kv_kivi_int4",
        "infini",
        "chunked_prefill",
        "paged_kv",
        "prefix_cache",
        "compressive_memory",
    }
    assert expected.issubset(set(lc.LONGCONTEXT_STRATEGY_REGISTRY.keys()))


# ---------------------------------------------------------------------------
# AureliusConfig integration
# ---------------------------------------------------------------------------

def test_aurelius_config_has_context_extension_fields() -> None:
    """AureliusConfig must expose context_extension_strategy and context_target_len."""
    cfg = AureliusConfig()
    assert hasattr(cfg, "context_extension_strategy"), (
        "AureliusConfig missing 'context_extension_strategy'"
    )
    assert hasattr(cfg, "context_target_len"), (
        "AureliusConfig missing 'context_target_len'"
    )
    # Defaults: OFF / none.
    assert cfg.context_extension_strategy == "none"
    assert cfg.context_target_len == 8192


def test_aurelius_config_custom_values() -> None:
    """Users can override context extension fields."""
    cfg = AureliusConfig(
        context_extension_strategy="yarn",
        context_target_len=131072,
    )
    assert cfg.context_extension_strategy == "yarn"
    assert cfg.context_target_len == 131_072


# ---------------------------------------------------------------------------
# End-to-end: DynamicContextScaler from AureliusConfig
# ---------------------------------------------------------------------------

def _make_scaler_from_config(cfg: AureliusConfig) -> lc.DynamicContextScaler:
    """Construct DynamicContextScaler from AureliusConfig fields."""
    strategy = cfg.context_extension_strategy
    if strategy == "none":
        strategy = "auto"
    return lc.DynamicContextScaler(
        strategy=strategy,
        dim=cfg.head_dim,
        base=cfg.rope_theta,
        train_len=cfg.max_seq_len,
    )


def test_end_to_end_1x_train_len() -> None:
    cfg = AureliusConfig()
    scaler = _make_scaler_from_config(cfg)
    seq_len = cfg.max_seq_len  # 8192
    cos, sin = scaler.get_cos_sin(seq_len)
    assert cos.shape == (seq_len, cfg.head_dim)
    assert torch.isfinite(cos).all(), "NaN/Inf in cos at 1x train_len"
    assert torch.isfinite(sin).all(), "NaN/Inf in sin at 1x train_len"


def test_end_to_end_2x_train_len() -> None:
    cfg = AureliusConfig()
    scaler = _make_scaler_from_config(cfg)
    seq_len = cfg.max_seq_len * 2  # 16384
    cos, sin = scaler.get_cos_sin(seq_len)
    assert cos.shape == (seq_len, cfg.head_dim)
    assert torch.isfinite(cos).all(), "NaN/Inf in cos at 2x train_len"
    assert torch.isfinite(sin).all(), "NaN/Inf in sin at 2x train_len"


def test_end_to_end_4x_train_len() -> None:
    cfg = AureliusConfig()
    scaler = _make_scaler_from_config(cfg)
    seq_len = cfg.max_seq_len * 4  # 32768
    cos, sin = scaler.get_cos_sin(seq_len)
    assert cos.shape == (seq_len, cfg.head_dim)
    assert torch.isfinite(cos).all(), "NaN/Inf in cos at 4x train_len"
    assert torch.isfinite(sin).all(), "NaN/Inf in sin at 4x train_len"


def test_end_to_end_8x_train_len() -> None:
    cfg = AureliusConfig()
    scaler = _make_scaler_from_config(cfg)
    seq_len = cfg.max_seq_len * 8  # 65536
    cos, sin = scaler.get_cos_sin(seq_len)
    assert cos.shape == (seq_len, cfg.head_dim)
    assert torch.isfinite(cos).all(), "NaN/Inf in cos at 8x train_len"
    assert torch.isfinite(sin).all(), "NaN/Inf in sin at 8x train_len"


def test_end_to_end_yarn_strategy_explicit() -> None:
    """Explicit YaRN strategy from config must produce finite outputs."""
    cfg = AureliusConfig(
        context_extension_strategy="yarn",
        context_target_len=32768,
    )
    scaler = _make_scaler_from_config(cfg)
    for mult in (1, 2, 4, 8):
        seq_len = cfg.max_seq_len * mult
        cos, sin = scaler.get_cos_sin(seq_len)
        assert torch.isfinite(cos).all(), f"NaN/Inf in cos at {mult}x with YaRN"
        assert torch.isfinite(sin).all(), f"NaN/Inf in sin at {mult}x with YaRN"


def test_end_to_end_ntk_strategy_explicit() -> None:
    """Explicit NTK strategy from config must produce finite outputs."""
    cfg = AureliusConfig(
        context_extension_strategy="ntk",
        context_target_len=65536,
    )
    scaler = _make_scaler_from_config(cfg)
    for mult in (1, 2, 4, 8):
        seq_len = cfg.max_seq_len * mult
        cos, sin = scaler.get_cos_sin(seq_len)
        assert torch.isfinite(cos).all(), f"NaN/Inf in cos at {mult}x with NTK"
        assert torch.isfinite(sin).all(), f"NaN/Inf in sin at {mult}x with NTK"


# ---------------------------------------------------------------------------
# No prohibited imports
# ---------------------------------------------------------------------------

def test_no_external_ml_deps() -> None:
    """Implementation must not import external ML libraries."""
    impl_path = Path(__file__).parent.parent.parent / "src" / "longcontext" / "context_window_extension.py"
    source = impl_path.read_text()
    banned = [
        "transformers", "einops", "trl", "xformers", "flash_attn",
        "bitsandbytes", "peft", "diffusers", "datasets", "accelerate",
        "deepspeed", "langchain", "llamaindex",
    ]
    for lib in banned:
        assert lib not in source, (
            f"Prohibited import '{lib}' found in context_window_extension.py"
        )
