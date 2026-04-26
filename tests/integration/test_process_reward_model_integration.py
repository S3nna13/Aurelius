"""Integration tests for ProcessRewardModel.

Verifies:
1. ALIGNMENT_REGISTRY["prm"] resolves to ProcessRewardModel
2. AureliusConfig.enable_prm_training=True triggers the PRM code path
   via build_model_for_training (non-default path)
3. Default config (enable_prm_training=False) returns vanilla AureliusTransformer
4. Full forward pass with PRM-enabled config is finite and correct shape
5. PRMInference.select_best works end-to-end on a tiny model
"""

from __future__ import annotations

import torch
import pytest

from src.model.config import AureliusConfig
from src.runtime.feature_flags import FeatureFlag, FEATURE_FLAG_REGISTRY
import src.alignment as alignment_pkg
from src.training.process_reward_model import ProcessRewardModel
from src.training.trainer import build_model_for_training


# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------

STEP_TOKEN_ID = 5  # < vocab_size=256


def tiny_prm_config() -> AureliusConfig:
    FEATURE_FLAG_REGISTRY.register(
        FeatureFlag(
            name="training.prm_training",
            enabled=True,
            metadata={"step_token_id": STEP_TOKEN_ID, "aggregation": "min"},
        )
    )
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return cfg


def tiny_base_config() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
        # enable_prm_training defaults to False
    )


# ---------------------------------------------------------------------------
# Test 1: ALIGNMENT_REGISTRY["prm"] == ProcessRewardModel
# ---------------------------------------------------------------------------

def test_alignment_registry_has_prm():
    assert hasattr(alignment_pkg, "ALIGNMENT_REGISTRY"), (
        "src/alignment/__init__.py must expose ALIGNMENT_REGISTRY"
    )
    registry = alignment_pkg.ALIGNMENT_REGISTRY
    assert "prm" in registry, f"'prm' not in ALIGNMENT_REGISTRY: {list(registry)}"
    assert registry["prm"] is ProcessRewardModel


# ---------------------------------------------------------------------------
# Test 2: build_model_for_training with enable_prm_training=True -> PRM
# ---------------------------------------------------------------------------

def test_build_model_prm_enabled():
    cfg = tiny_prm_config()
    model = build_model_for_training(cfg)
    assert isinstance(model, ProcessRewardModel), (
        f"Expected ProcessRewardModel, got {type(model)}"
    )
    assert model.step_token_id == STEP_TOKEN_ID


# ---------------------------------------------------------------------------
# Test 3: Default config (enable_prm_training=False) -> AureliusTransformer
# ---------------------------------------------------------------------------

def test_build_model_default_path():
    from src.model.transformer import AureliusTransformer

    FEATURE_FLAG_REGISTRY._flags.pop("training.prm_training", None)
    cfg = tiny_base_config()
    model = build_model_for_training(cfg)
    assert isinstance(model, AureliusTransformer), (
        f"Expected AureliusTransformer for default config, got {type(model)}"
    )


# ---------------------------------------------------------------------------
# Test 4: Full forward pass with PRM-enabled config
# ---------------------------------------------------------------------------

def test_prm_forward_pass_integration():
    cfg = tiny_prm_config()
    prm = ProcessRewardModel(cfg)
    prm.train(False)

    B, T = 2, 20
    torch.manual_seed(42)
    ids = torch.randint(10, 200, (B, T), dtype=torch.long)
    # Insert two step tokens per row
    ids[:, 5] = STEP_TOKEN_ID
    ids[:, 12] = STEP_TOKEN_ID

    labels = torch.randint(0, 2, (B, 2))

    with torch.no_grad():
        loss, step_rewards = prm(ids, labels=labels)

    assert loss is not None
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    assert step_rewards.shape == (B, 2)
    assert torch.isfinite(step_rewards).all(), "step_rewards contain NaN/Inf"


# ---------------------------------------------------------------------------
# Test 5: PRMInference end-to-end select_best on tiny model
# ---------------------------------------------------------------------------

def test_prm_inference_end_to_end():
    from src.training.process_reward_model import PRMInference

    cfg = tiny_prm_config()
    prm = ProcessRewardModel(cfg)
    prm.train(False)

    inference = PRMInference(prm, aggregation="min")

    # Three candidate chains, each with two steps
    candidates = [
        ["x" * 3, "y" * 3],
        ["a" * 4, "b" * 4],
        ["p" * 2, "q" * 2],
    ]
    best_idx = inference.select_best(candidates, step_token="\n\n")
    assert isinstance(best_idx, int)
    assert 0 <= best_idx < len(candidates)


# ---------------------------------------------------------------------------
# Test 6: AureliusConfig PRM fields exist with correct defaults
# ---------------------------------------------------------------------------

def test_config_prm_fields_defaults():
    FEATURE_FLAG_REGISTRY._flags.pop("training.prm_training", None)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
        head_dim=16, d_ff=128, vocab_size=256, max_seq_len=64,
    )
    assert hasattr(cfg, "enable_prm_training")
    assert cfg.enable_prm_training is False
    assert hasattr(cfg, "prm_step_token_id")
    assert cfg.prm_step_token_id == 50256
    assert hasattr(cfg, "prm_aggregation")
    assert cfg.prm_aggregation == "min"
