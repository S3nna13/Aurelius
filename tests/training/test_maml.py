"""Tests for MAML (Model-Agnostic Meta-Learning) module."""
from __future__ import annotations

import torch
import pytest
from torch import Tensor

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.maml import (
    MAMLConfig,
    MAMLTrainer,
    Task,
    apply_adapted_params,
    compute_task_loss,
    inner_loop,
    restore_params,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB = 256
SEQ_LEN = 4
BATCH = 1


@pytest.fixture(scope="module")
def small_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def model(small_cfg: AureliusConfig) -> AureliusTransformer:
    torch.manual_seed(42)
    return AureliusTransformer(small_cfg)


@pytest.fixture(scope="module")
def maml_cfg() -> MAMLConfig:
    return MAMLConfig(n_inner_steps=1, inner_lr=0.01)


@pytest.fixture(scope="module")
def trainer(model: AureliusTransformer, maml_cfg: MAMLConfig) -> MAMLTrainer:
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=maml_cfg.meta_lr)
    return MAMLTrainer(model, meta_optimizer, maml_cfg)


@pytest.fixture
def input_ids() -> Tensor:
    torch.manual_seed(0)
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN))


def _make_task(seed: int = 0) -> Task:
    torch.manual_seed(seed)
    support = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    query = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    return Task(support_ids=support, query_ids=query, task_id=f"task_{seed}")


def _make_tasks(n: int = 2) -> list[Task]:
    return [_make_task(seed=i) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. MAMLConfig defaults
# ---------------------------------------------------------------------------

def test_maml_config_defaults():
    cfg = MAMLConfig()
    assert cfg.n_inner_steps == 5
    assert cfg.inner_lr == 0.01
    assert cfg.meta_lr == 1e-3
    assert cfg.n_tasks == 4
    assert cfg.first_order is True
    assert cfg.task_batch_size == 8


# ---------------------------------------------------------------------------
# 2. Task fields
# ---------------------------------------------------------------------------

def test_task_fields():
    support = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    query = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    task = Task(support_ids=support, query_ids=query, task_id="t1")
    assert task.support_ids is support
    assert task.query_ids is query
    assert task.task_id == "t1"


def test_task_default_task_id():
    support = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    query = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    task = Task(support_ids=support, query_ids=query)
    assert task.task_id == ""


# ---------------------------------------------------------------------------
# 3. compute_task_loss returns scalar tensor
# ---------------------------------------------------------------------------

def test_compute_task_loss_returns_scalar(model, input_ids):
    loss = compute_task_loss(model, input_ids)
    assert isinstance(loss, Tensor)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 4. compute_task_loss is positive
# ---------------------------------------------------------------------------

def test_compute_task_loss_is_positive(model, input_ids):
    loss = compute_task_loss(model, input_ids)
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# 5. inner_loop returns dict of tensors
# ---------------------------------------------------------------------------

def test_inner_loop_returns_dict_of_tensors(model, maml_cfg):
    task = _make_task(0)
    adapted = inner_loop(model, task, maml_cfg)
    assert isinstance(adapted, dict)
    for name, val in adapted.items():
        assert isinstance(val, Tensor), f"Value for '{name}' is not a Tensor"


# ---------------------------------------------------------------------------
# 6. inner_loop adapted params have same keys as model params
# ---------------------------------------------------------------------------

def test_inner_loop_keys_match_model_params(model, maml_cfg):
    expected_keys = {name for name, p in model.named_parameters() if p.requires_grad}
    task = _make_task(1)
    adapted = inner_loop(model, task, maml_cfg)
    assert set(adapted.keys()) == expected_keys


# ---------------------------------------------------------------------------
# 7. inner_loop params differ from original after adaptation
# ---------------------------------------------------------------------------

def test_inner_loop_params_differ_from_original(model, maml_cfg):
    original = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}
    task = _make_task(2)
    adapted = inner_loop(model, task, maml_cfg)
    any_changed = any(
        not torch.equal(original[name], adapted[name])
        for name in original
    )
    assert any_changed, "No parameter changed after inner_loop — gradients may be zero"


# ---------------------------------------------------------------------------
# 8. apply_adapted_params + restore_params round-trip
# ---------------------------------------------------------------------------

def test_apply_and_restore_params_round_trip(model):
    original = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}

    # Build fake adapted params (add 1 to each)
    fake_adapted = {name: val + 1.0 for name, val in original.items()}

    apply_adapted_params(model, fake_adapted)
    # Verify fake params are applied
    for name, param in model.named_parameters():
        if name in fake_adapted:
            assert torch.allclose(param.data, fake_adapted[name]), \
                f"apply_adapted_params did not set '{name}' correctly"

    restore_params(model, original)
    # Verify original params are restored
    for name, param in model.named_parameters():
        if name in original:
            assert torch.equal(param.data, original[name]), \
                f"restore_params did not restore '{name}' correctly"


# ---------------------------------------------------------------------------
# 9. MAMLTrainer.meta_train_step returns required keys
# ---------------------------------------------------------------------------

def test_meta_train_step_returns_required_keys(trainer):
    tasks = _make_tasks(n=2)
    result = trainer.meta_train_step(tasks)
    assert "meta_loss" in result
    assert "mean_task_loss" in result
    assert "n_tasks" in result


# ---------------------------------------------------------------------------
# 10. MAMLTrainer.meta_train_step meta_loss is finite
# ---------------------------------------------------------------------------

def test_meta_train_step_meta_loss_is_finite(trainer):
    tasks = _make_tasks(n=2)
    result = trainer.meta_train_step(tasks)
    assert isinstance(result["meta_loss"], float)
    assert torch.isfinite(torch.tensor(result["meta_loss"])), \
        f"meta_loss is not finite: {result['meta_loss']}"


# ---------------------------------------------------------------------------
# 11. MAMLTrainer.meta_train_step n_tasks matches input
# ---------------------------------------------------------------------------

def test_meta_train_step_n_tasks_matches_input(trainer):
    tasks = _make_tasks(n=3)
    result = trainer.meta_train_step(tasks)
    assert result["n_tasks"] == 3


# ---------------------------------------------------------------------------
# 12. MAMLTrainer.adapt returns dict of tensors
# ---------------------------------------------------------------------------

def test_adapt_returns_dict_of_tensors(trainer, input_ids):
    adapted = trainer.adapt(input_ids)
    assert isinstance(adapted, dict)
    for name, val in adapted.items():
        assert isinstance(val, Tensor), f"Value for '{name}' is not a Tensor"


# ---------------------------------------------------------------------------
# 13. MAMLTrainer.evaluate_adapted returns float
# ---------------------------------------------------------------------------

def test_evaluate_adapted_returns_float(trainer, input_ids):
    adapted = trainer.adapt(input_ids)
    result = trainer.evaluate_adapted(adapted, input_ids)
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert torch.isfinite(torch.tensor(result)), f"evaluate_adapted returned non-finite: {result}"


# ---------------------------------------------------------------------------
# 14. inner_loop with n_inner_steps=0 returns original params unchanged
# ---------------------------------------------------------------------------

def test_inner_loop_zero_steps_returns_original_params(model):
    cfg_zero = MAMLConfig(n_inner_steps=0, inner_lr=0.01)
    original = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}
    task = _make_task(7)
    adapted = inner_loop(model, task, cfg_zero)
    for name in original:
        assert torch.equal(original[name], adapted[name]), \
            f"Param '{name}' changed with n_inner_steps=0"
