"""Tests for activation checkpointing policy planner."""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.checkpoint_policy import (
    CheckpointPolicyConfig,
    CheckpointStrategy,
    LayerMemoryEstimate,
    apply_checkpoint_plan,
    compute_checkpoint_plan,
    estimate_layer_memory,
)

# Small model config for tests
SMALL_CFG = AureliusConfig(
    n_layers=4,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=32,
)

BATCH_SIZE = 2
SEQ_LEN = 16
D_MODEL = 64
DTYPE_BYTES = 2


@pytest.fixture
def small_model():
    return AureliusTransformer(SMALL_CFG)


@pytest.fixture
def single_layer(small_model):
    return small_model.layers[0]


# -------------------------------------------------------------------------
# 1. test_estimate_layer_memory_returns_estimate
# -------------------------------------------------------------------------
def test_estimate_layer_memory_returns_estimate(single_layer):
    est = estimate_layer_memory(single_layer, BATCH_SIZE, SEQ_LEN, D_MODEL, DTYPE_BYTES)
    assert isinstance(est, LayerMemoryEstimate)


# -------------------------------------------------------------------------
# 2. test_estimate_layer_memory_activation_formula
# -------------------------------------------------------------------------
def test_estimate_layer_memory_activation_formula(single_layer):
    est = estimate_layer_memory(single_layer, BATCH_SIZE, SEQ_LEN, D_MODEL, DTYPE_BYTES)
    expected = 6 * BATCH_SIZE * SEQ_LEN * D_MODEL * DTYPE_BYTES
    assert est.activation_bytes == expected


# -------------------------------------------------------------------------
# 3. test_compute_plan_all_strategy
# -------------------------------------------------------------------------
def test_compute_plan_all_strategy(small_model):
    cfg = CheckpointPolicyConfig(strategy=CheckpointStrategy.ALL)
    plan = compute_checkpoint_plan(small_model, cfg, BATCH_SIZE, SEQ_LEN, D_MODEL, DTYPE_BYTES)
    assert len(plan.checkpointed_layers) == SMALL_CFG.n_layers
    assert plan.checkpointed_layers == list(range(SMALL_CFG.n_layers))


# -------------------------------------------------------------------------
# 4. test_compute_plan_none_strategy
# -------------------------------------------------------------------------
def test_compute_plan_none_strategy(small_model):
    cfg = CheckpointPolicyConfig(strategy=CheckpointStrategy.NONE)
    plan = compute_checkpoint_plan(small_model, cfg, BATCH_SIZE, SEQ_LEN, D_MODEL, DTYPE_BYTES)
    assert plan.checkpointed_layers == []


# -------------------------------------------------------------------------
# 5. test_compute_plan_uniform_interval_2
# -------------------------------------------------------------------------
def test_compute_plan_uniform_interval_2(small_model):
    cfg = CheckpointPolicyConfig(strategy=CheckpointStrategy.UNIFORM, uniform_interval=2)
    plan = compute_checkpoint_plan(small_model, cfg, BATCH_SIZE, SEQ_LEN, D_MODEL, DTYPE_BYTES)
    # With 4 layers (0,1,2,3) and interval=2 -> checkpoint layers 0 and 2
    expected = [i for i in range(SMALL_CFG.n_layers) if i % 2 == 0]
    assert plan.checkpointed_layers == expected


# -------------------------------------------------------------------------
# 6. test_compute_plan_memory_saved_nonneg
# -------------------------------------------------------------------------
def test_compute_plan_memory_saved_nonneg(small_model):
    cfg = CheckpointPolicyConfig(strategy=CheckpointStrategy.UNIFORM, uniform_interval=2)
    plan = compute_checkpoint_plan(small_model, cfg, BATCH_SIZE, SEQ_LEN, D_MODEL, DTYPE_BYTES)
    assert plan.memory_saved_bytes >= 0


# -------------------------------------------------------------------------
# 7. test_checkpoint_plan_n_checkpointed
# -------------------------------------------------------------------------
def test_checkpoint_plan_n_checkpointed(small_model):
    cfg = CheckpointPolicyConfig(strategy=CheckpointStrategy.UNIFORM, uniform_interval=2)
    plan = compute_checkpoint_plan(small_model, cfg, BATCH_SIZE, SEQ_LEN, D_MODEL, DTYPE_BYTES)
    assert plan.n_checkpointed == len(plan.checkpointed_layers)


# -------------------------------------------------------------------------
# 8. test_apply_checkpoint_plan_no_crash
# -------------------------------------------------------------------------
def test_apply_checkpoint_plan_no_crash(small_model):
    cfg = CheckpointPolicyConfig(strategy=CheckpointStrategy.UNIFORM, uniform_interval=2)
    plan = compute_checkpoint_plan(small_model, cfg, BATCH_SIZE, SEQ_LEN, D_MODEL, DTYPE_BYTES)
    # Should not raise
    apply_checkpoint_plan(small_model, plan)


# -------------------------------------------------------------------------
# 9. test_apply_checkpoint_plan_forward_works
# -------------------------------------------------------------------------
def test_apply_checkpoint_plan_forward_works(small_model):
    cfg = CheckpointPolicyConfig(strategy=CheckpointStrategy.UNIFORM, uniform_interval=2)
    plan = compute_checkpoint_plan(small_model, cfg, BATCH_SIZE, SEQ_LEN, D_MODEL, DTYPE_BYTES)
    apply_checkpoint_plan(small_model, plan)

    # Model should still produce valid output
    small_model.train()  # checkpoint requires training mode for gradient flow
    input_ids = torch.randint(0, SMALL_CFG.vocab_size, (BATCH_SIZE, SEQ_LEN))
    loss, logits, _ = small_model(input_ids, labels=input_ids)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, SMALL_CFG.vocab_size)
    assert loss is not None


# -------------------------------------------------------------------------
# 10. test_memory_optimal_within_budget
# -------------------------------------------------------------------------
def test_memory_optimal_within_budget(small_model):
    # Use a generous budget (1 GB) — should be well within reach for a tiny model
    cfg = CheckpointPolicyConfig(
        strategy=CheckpointStrategy.MEMORY_OPTIMAL,
        memory_budget_gb=1.0,
    )
    plan = compute_checkpoint_plan(small_model, cfg, BATCH_SIZE, SEQ_LEN, D_MODEL, DTYPE_BYTES)
    budget_bytes = int(cfg.memory_budget_gb * (1024**3))
    assert plan.estimated_peak_memory_bytes <= budget_bytes
