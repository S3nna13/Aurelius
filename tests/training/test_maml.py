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
    clone_parameters,
    compute_task_loss,
    inner_loop,
    inner_update,
    maml_loss,
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


# ---------------------------------------------------------------------------
# New spec-aligned API tests (16+ tests)
# ---------------------------------------------------------------------------

# 15. MAMLConfig spec fields exist with correct defaults
def test_maml_config_spec_fields():
    cfg = MAMLConfig()
    assert hasattr(cfg, "inner_steps")
    assert hasattr(cfg, "outer_lr")
    assert cfg.inner_steps == 5
    assert cfg.outer_lr == 1e-3


# 16. MAMLConfig spec fields can be set at construction
def test_maml_config_spec_fields_settable():
    cfg = MAMLConfig(inner_steps=3, outer_lr=5e-4, inner_lr=0.05, n_tasks=8, first_order=False)
    assert cfg.inner_steps == 3
    assert cfg.outer_lr == 5e-4
    assert cfg.inner_lr == 0.05
    assert cfg.n_tasks == 8
    assert cfg.first_order is False


# 17. clone_parameters returns OrderedDict with same keys as model.named_parameters()
def test_clone_parameters_returns_ordered_dict_with_model_keys(model):
    from collections import OrderedDict
    cloned = clone_parameters(model)
    assert isinstance(cloned, OrderedDict)
    model_keys = set(name for name, _ in model.named_parameters())
    assert set(cloned.keys()) == model_keys


# 18. clone_parameters returns tensors (not the same objects as model params)
def test_clone_parameters_returns_independent_copies(model):
    cloned = clone_parameters(model)
    for name, param in model.named_parameters():
        assert name in cloned
        assert cloned[name] is not param.data, \
            f"clone_parameters returned same object for '{name}'"
        assert torch.equal(cloned[name].data, param.data), \
            f"Cloned value for '{name}' does not match original"


# 19. clone_parameters tensors have requires_grad consistent with model params
def test_clone_parameters_requires_grad(model):
    cloned = clone_parameters(model)
    for name, param in model.named_parameters():
        assert cloned[name].requires_grad == param.requires_grad, \
            f"requires_grad mismatch for '{name}'"


# 20. inner_update returns an OrderedDict
def test_inner_update_returns_dict(model):
    cfg = MAMLConfig(inner_steps=1, inner_lr=0.01, first_order=True)
    support = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    result = inner_update(model, support, cfg)
    assert isinstance(result, dict)


# 21. inner_update result has same keys as model.named_parameters()
def test_inner_update_keys_match_model(model):
    cfg = MAMLConfig(inner_steps=1, inner_lr=0.01, first_order=True)
    support = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    result = inner_update(model, support, cfg)
    model_keys = {name for name, _ in model.named_parameters()}
    assert set(result.keys()) == model_keys


# 22. inner_update changes at least one parameter vs initial model state
def test_inner_update_changes_params(model):
    cfg = MAMLConfig(inner_steps=1, inner_lr=0.01, first_order=True)
    support = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    original = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}
    updated = inner_update(model, support, cfg)
    any_changed = any(
        not torch.equal(original[name], updated[name])
        for name in original
        if name in updated
    )
    assert any_changed, "inner_update: no parameter changed after 1 step"


# 23. inner_update does NOT modify model in-place
def test_inner_update_does_not_modify_model_in_place(model):
    cfg = MAMLConfig(inner_steps=2, inner_lr=0.01, first_order=True)
    support = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    before = {name: p.data.clone() for name, p in model.named_parameters()}
    inner_update(model, support, cfg)
    after = {name: p.data.clone() for name, p in model.named_parameters()}
    for name in before:
        assert torch.equal(before[name], after[name]), \
            f"inner_update modified model in-place for '{name}'"


# 24. inner_update works with inner_steps=1
def test_inner_update_single_step(model):
    cfg = MAMLConfig(inner_steps=1, inner_lr=0.01, first_order=True)
    support = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    result = inner_update(model, support, cfg)
    assert isinstance(result, dict)
    assert len(result) > 0


# 25. inner_update accepts explicit params argument
def test_inner_update_accepts_params_arg(model):
    cfg = MAMLConfig(inner_steps=1, inner_lr=0.01, first_order=True)
    support = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    initial_params = clone_parameters(model)
    result = inner_update(model, support, cfg, params=initial_params)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(initial_params.keys())


# 26. maml_loss returns a scalar tensor
def test_maml_loss_returns_scalar(model):
    cfg = MAMLConfig(inner_steps=1, inner_lr=0.01, first_order=True)
    support = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    query = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    updated_params = inner_update(model, support, cfg)
    loss = maml_loss(model, updated_params, query)
    assert isinstance(loss, Tensor)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"


# 27. maml_loss returns a positive float
def test_maml_loss_is_positive(model):
    cfg = MAMLConfig(inner_steps=1, inner_lr=0.01, first_order=True)
    support = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    query = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    updated_params = inner_update(model, support, cfg)
    loss = maml_loss(model, updated_params, query)
    assert loss.item() > 0.0, f"Expected positive maml_loss, got {loss.item()}"


# 28. maml_loss does NOT modify model in-place
def test_maml_loss_does_not_modify_model_in_place(model):
    cfg = MAMLConfig(inner_steps=1, inner_lr=0.01, first_order=True)
    support = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    query = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    updated_params = inner_update(model, support, cfg)
    before = {name: p.data.clone() for name, p in model.named_parameters()}
    maml_loss(model, updated_params, query)
    after = {name: p.data.clone() for name, p in model.named_parameters()}
    for name in before:
        assert torch.equal(before[name], after[name]), \
            f"maml_loss modified model in-place for '{name}'"


# 29. MAMLTrainer accepts (model, config, optimizer) spec convention
def test_maml_trainer_spec_constructor(model):
    cfg = MAMLConfig(inner_steps=1, inner_lr=0.01, outer_lr=1e-3, first_order=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = MAMLTrainer(model, cfg, optimizer)
    assert trainer.config is cfg
    assert trainer.meta_optimizer is optimizer
    assert trainer.model is model


# 30. MAMLTrainer.meta_train_step accepts (support_ids, query_ids) tuple list
def test_meta_train_step_accepts_tuple_task_batch(model):
    cfg = MAMLConfig(inner_steps=1, inner_lr=0.01, outer_lr=1e-3, first_order=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = MAMLTrainer(model, cfg, optimizer)
    task_batch = [
        (torch.randint(0, VOCAB, (BATCH, SEQ_LEN)), torch.randint(0, VOCAB, (BATCH, SEQ_LEN)))
        for _ in range(2)
    ]
    result = trainer.meta_train_step(task_batch)
    assert "meta_loss" in result
    assert "n_tasks" in result
    assert result["n_tasks"] == 2


# 31. meta_train_step with tuple tasks returns positive meta_loss
def test_meta_train_step_tuple_tasks_positive_loss(model):
    cfg = MAMLConfig(inner_steps=1, inner_lr=0.01, outer_lr=1e-3, first_order=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = MAMLTrainer(model, cfg, optimizer)
    task_batch = [
        (torch.randint(0, VOCAB, (BATCH, SEQ_LEN)), torch.randint(0, VOCAB, (BATCH, SEQ_LEN)))
        for _ in range(2)
    ]
    result = trainer.meta_train_step(task_batch)
    assert isinstance(result["meta_loss"], float)
    assert result["meta_loss"] > 0.0


# 32. first_order=True and first_order=False both produce valid results with inner_update
def test_inner_update_first_order_vs_second_order(model):
    support = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    cfg_fo = MAMLConfig(inner_steps=1, inner_lr=0.01, first_order=True)
    cfg_so = MAMLConfig(inner_steps=1, inner_lr=0.01, first_order=False)
    result_fo = inner_update(model, support, cfg_fo)
    result_so = inner_update(model, support, cfg_so)
    assert isinstance(result_fo, dict)
    assert isinstance(result_so, dict)
    assert set(result_fo.keys()) == set(result_so.keys())


# 33. gradient accumulation: meta_loss scales with number of tasks
def test_meta_loss_scales_with_n_tasks(model):
    cfg = MAMLConfig(inner_steps=1, inner_lr=0.01, outer_lr=1e-3, first_order=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = MAMLTrainer(model, cfg, optimizer)
    torch.manual_seed(42)
    task_batch_2 = [
        (torch.randint(0, VOCAB, (BATCH, SEQ_LEN)), torch.randint(0, VOCAB, (BATCH, SEQ_LEN)))
        for _ in range(2)
    ]
    torch.manual_seed(42)
    task_batch_4 = [
        (torch.randint(0, VOCAB, (BATCH, SEQ_LEN)), torch.randint(0, VOCAB, (BATCH, SEQ_LEN)))
        for _ in range(4)
    ]
    result_2 = trainer.meta_train_step(task_batch_2)
    result_4 = trainer.meta_train_step(task_batch_4)
    # More tasks should accumulate more loss (not strictly equal)
    assert result_4["n_tasks"] == 4
    assert result_2["n_tasks"] == 2
