"""Tests for adapter-based continual learning (adapter_continual.py)."""
from __future__ import annotations

import torch
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.adapter_continual import (
    AdapterContinualConfig,
    AdapterContinualTrainer,
    MultiTaskAdapterLayer,
    TaskAdapter,
    compute_overlap_penalty,
    compute_task_overlap,
    freeze_adapters,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg():
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


@pytest.fixture
def small_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def ac_config():
    return AdapterContinualConfig()


@pytest.fixture
def adapter_layer(ac_config):
    return MultiTaskAdapterLayer(d_model=64, config=ac_config)


# ---------------------------------------------------------------------------
# 1. Test AdapterContinualConfig defaults
# ---------------------------------------------------------------------------

def test_adapter_continual_config_defaults():
    cfg = AdapterContinualConfig()
    assert cfg.adapter_rank == 8
    assert cfg.adapter_alpha == 16.0
    assert cfg.n_tasks == 5
    assert cfg.freeze_old_adapters is True
    assert cfg.task_overlap_penalty == 0.1


# ---------------------------------------------------------------------------
# 2. Test TaskAdapter forward output shape
# ---------------------------------------------------------------------------

def test_task_adapter_forward_output_shape():
    adapter = TaskAdapter(d_model=64, rank=4, alpha=8.0, task_id=0)
    x = torch.randn(2, 10, 64)
    out = adapter(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# 3. Test TaskAdapter forward residual (output changes from input)
# ---------------------------------------------------------------------------

def test_task_adapter_forward_residual_changes_after_training():
    """After a gradient update, the adapter output should differ from the input."""
    torch.manual_seed(42)
    adapter = TaskAdapter(d_model=16, rank=4, alpha=8.0, task_id=0)
    x = torch.randn(1, 4, 16)

    # Give A and B non-trivial values so the output differs from identity
    with torch.no_grad():
        adapter.A.normal_(0, 0.1)
        adapter.B.normal_(0, 0.1)

    out = adapter(x)
    # Output should not be identical to input once weights are non-trivial
    assert not torch.allclose(out, x), "Adapter output should differ from input when weights are non-zero"


# ---------------------------------------------------------------------------
# 4. Test MultiTaskAdapterLayer.add_task registers adapter
# ---------------------------------------------------------------------------

def test_multi_task_adapter_layer_add_task_registers(adapter_layer):
    assert 0 not in adapter_layer.adapters
    adapter_layer.add_task(0)
    assert 0 in adapter_layer.adapters
    assert isinstance(adapter_layer.adapters[0], TaskAdapter)


# ---------------------------------------------------------------------------
# 5. Test MultiTaskAdapterLayer.forward routes to correct adapter
# ---------------------------------------------------------------------------

def test_multi_task_adapter_layer_forward_routes(adapter_layer):
    adapter_layer.add_task(0)
    adapter_layer.add_task(1)

    # Give task-0 adapter non-trivial weights
    with torch.no_grad():
        adapter_layer.adapters[0].A.fill_(0.1)
        adapter_layer.adapters[0].B.fill_(0.1)
        adapter_layer.adapters[1].A.fill_(0.0)
        adapter_layer.adapters[1].B.fill_(0.0)

    x = torch.ones(1, 4, 64)
    out0 = adapter_layer(x, task_id=0)
    out1 = adapter_layer(x, task_id=1)

    # Task-0 has non-trivial adapter; task-1 is near identity
    assert not torch.allclose(out0, out1), "Different adapters should produce different outputs"


# ---------------------------------------------------------------------------
# 6. Test MultiTaskAdapterLayer unknown task → identity
# ---------------------------------------------------------------------------

def test_multi_task_adapter_layer_unknown_task_identity(adapter_layer):
    x = torch.randn(2, 8, 64)
    out = adapter_layer(x, task_id=99)
    assert torch.equal(out, x), "Unknown task_id should return input unchanged"


# ---------------------------------------------------------------------------
# 7. Test compute_task_overlap returns float in [0, 1]
# ---------------------------------------------------------------------------

def test_compute_task_overlap_range():
    torch.manual_seed(1)
    a1 = TaskAdapter(d_model=32, rank=4, alpha=8.0, task_id=0)
    a2 = TaskAdapter(d_model=32, rank=4, alpha=8.0, task_id=1)
    # Give non-trivial weights
    with torch.no_grad():
        a1.A.normal_(0, 0.1)
        a1.B.normal_(0, 0.1)
        a2.A.normal_(0, 0.1)
        a2.B.normal_(0, 0.1)
    overlap = compute_task_overlap(a1, a2)
    assert isinstance(overlap, float)
    assert 0.0 <= overlap <= 1.0, f"Overlap {overlap} out of [0, 1]"


# ---------------------------------------------------------------------------
# 8. Test compute_task_overlap same adapter → ~1.0
# ---------------------------------------------------------------------------

def test_compute_task_overlap_same_adapter():
    torch.manual_seed(2)
    a = TaskAdapter(d_model=32, rank=4, alpha=8.0, task_id=0)
    with torch.no_grad():
        a.A.normal_(0, 0.5)
        a.B.normal_(0, 0.5)
    overlap = compute_task_overlap(a, a)
    assert abs(overlap - 1.0) < 1e-5, f"Same adapter overlap should be ~1.0, got {overlap}"


# ---------------------------------------------------------------------------
# 9. Test freeze_adapters freezes all but current
# ---------------------------------------------------------------------------

def test_freeze_adapters_freezes_all_but_current(adapter_layer):
    for tid in [0, 1, 2]:
        adapter_layer.add_task(tid)

    freeze_adapters(adapter_layer, except_task_id=1)

    for tid, adapter in adapter_layer.adapters.items():
        for param in adapter.parameters():
            if tid == 1:
                assert param.requires_grad, f"Task {tid} adapter should remain trainable"
            else:
                assert not param.requires_grad, f"Task {tid} adapter should be frozen"


# ---------------------------------------------------------------------------
# 10. Test compute_overlap_penalty returns scalar
# ---------------------------------------------------------------------------

def test_compute_overlap_penalty_returns_scalar():
    torch.manual_seed(3)
    adapters = {}
    for tid in [0, 1, 2]:
        a = TaskAdapter(d_model=16, rank=4, alpha=8.0, task_id=tid)
        with torch.no_grad():
            a.A.normal_(0, 0.1)
            a.B.normal_(0, 0.1)
        adapters[tid] = a

    penalty = compute_overlap_penalty(adapters)
    assert penalty.dim() == 0, "Penalty should be a scalar tensor"
    assert penalty.item() >= 0.0, "Penalty should be non-negative"


# ---------------------------------------------------------------------------
# 11. Test AdapterContinualTrainer.train_step returns correct keys
# ---------------------------------------------------------------------------

def test_trainer_train_step_returns_correct_keys(small_model):
    config = AdapterContinualConfig(adapter_rank=4, adapter_alpha=8.0)
    trainer = AdapterContinualTrainer(small_model, config)
    trainer.start_task(task_id=0)

    input_ids = torch.randint(0, 256, (2, 16))
    labels = torch.randint(0, 256, (2, 16))

    result = trainer.train_step(input_ids, labels, task_id=0)

    assert "loss" in result, "Result must contain 'loss'"
    assert "task_id" in result, "Result must contain 'task_id'"
    assert "overlap_penalty" in result, "Result must contain 'overlap_penalty'"
    assert result["task_id"] == 0
    assert isinstance(result["loss"], float)
    assert isinstance(result["overlap_penalty"], float)


# ---------------------------------------------------------------------------
# 12. Test AdapterContinualTrainer.evaluate_forgetting with no history → 0.0
# ---------------------------------------------------------------------------

def test_trainer_evaluate_forgetting_no_history(small_model):
    config = AdapterContinualConfig()
    trainer = AdapterContinualTrainer(small_model, config)
    # No tasks trained yet
    result = trainer.evaluate_forgetting({0: 1.5, 1: 2.0})
    assert result == 0.0, f"Expected 0.0 with no history, got {result}"


# ---------------------------------------------------------------------------
# Bonus: evaluate_forgetting with task history
# ---------------------------------------------------------------------------

def test_trainer_evaluate_forgetting_detects_increase(small_model):
    config = AdapterContinualConfig(adapter_rank=4)
    trainer = AdapterContinualTrainer(small_model, config)
    trainer.start_task(task_id=0)

    input_ids = torch.randint(0, 256, (2, 8))
    labels = torch.randint(0, 256, (2, 8))
    trainer.train_step(input_ids, labels, task_id=0)

    # Simulate that task-0 loss increased after learning task-1
    initial = trainer._initial_losses[0]
    forgetting = trainer.evaluate_forgetting({0: initial + 1.0})
    assert forgetting > 0.0, "Should detect forgetting when loss increases"
