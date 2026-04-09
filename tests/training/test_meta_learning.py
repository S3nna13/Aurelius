"""Tests for MAML / Reptile meta-learning module."""
from __future__ import annotations

import torch
import pytest
from torch import nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.meta_learning import (
    MetaLearningConfig,
    MetaLearner,
    compute_task_loss,
    fomaml_meta_gradient,
    inner_loop_update,
    reptile_update,
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


@pytest.fixture
def model(small_cfg: AureliusConfig) -> AureliusTransformer:
    torch.manual_seed(42)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def support_ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN))


@pytest.fixture
def support_labels(support_ids: torch.Tensor) -> torch.Tensor:
    return support_ids.clone()


@pytest.fixture
def meta_learner(model: AureliusTransformer) -> MetaLearner:
    cfg = MetaLearningConfig(algorithm="reptile", n_inner_steps=2, inner_lr=0.01, outer_lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    return MetaLearner(model, cfg, optimizer)


def _make_tasks(n: int = 2) -> list[dict]:
    tasks = []
    for _ in range(n):
        ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        tasks.append({
            "support_ids": ids,
            "support_labels": ids.clone(),
            "query_ids": ids,
            "query_labels": ids.clone(),
        })
    return tasks


# ---------------------------------------------------------------------------
# MetaLearningConfig tests
# ---------------------------------------------------------------------------

def test_config_defaults():
    """MetaLearningConfig should have correct default values."""
    cfg = MetaLearningConfig()
    assert cfg.algorithm == "reptile"
    assert cfg.n_inner_steps == 5
    assert cfg.inner_lr == 0.01
    assert cfg.outer_lr == 0.001
    assert cfg.n_tasks_per_batch == 4
    assert cfg.first_order is True


# ---------------------------------------------------------------------------
# compute_task_loss tests
# ---------------------------------------------------------------------------

def test_compute_task_loss_scalar(model, support_ids, support_labels):
    """compute_task_loss must return a scalar (0-dim) tensor."""
    loss = compute_task_loss(model, support_ids, support_labels)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"


def test_compute_task_loss_finite(model, support_ids, support_labels):
    """compute_task_loss must return a finite (non-nan, non-inf) value."""
    loss = compute_task_loss(model, support_ids, support_labels)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# inner_loop_update tests
# ---------------------------------------------------------------------------

def test_inner_loop_update_returns_dict(model, support_ids, support_labels):
    """inner_loop_update must return a dict keyed by parameter names."""
    param_names = {name for name, p in model.named_parameters() if p.requires_grad}
    adapted = inner_loop_update(model, support_ids, support_labels, inner_lr=0.01, n_steps=1)
    assert isinstance(adapted, dict)
    assert set(adapted.keys()) == param_names


def test_inner_loop_update_params_differ(model, support_ids, support_labels):
    """Adapted params must differ from original params (gradient was applied)."""
    original = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}
    adapted = inner_loop_update(model, support_ids, support_labels, inner_lr=0.01, n_steps=1)

    any_changed = any(
        not torch.equal(original[name], adapted[name])
        for name in original
    )
    assert any_changed, "No parameter changed after inner loop — gradients may be zero"


# ---------------------------------------------------------------------------
# reptile_update tests
# ---------------------------------------------------------------------------

def _make_param_dicts(seed_orig: int = 0, seed_adapt: int = 1):
    torch.manual_seed(seed_orig)
    original = {"w": torch.randn(4, 4), "b": torch.randn(4)}
    torch.manual_seed(seed_adapt)
    adapted = {"w": torch.randn(4, 4), "b": torch.randn(4)}
    return original, adapted


def test_reptile_update_moves_toward_adapted():
    """reptile_update should move params toward adapted params."""
    original, adapted = _make_param_dicts()
    outer_lr = 0.5
    updated = reptile_update(original, adapted, outer_lr)

    for name in original:
        # distance after update should be smaller than before
        dist_before = (adapted[name] - original[name]).norm().item()
        dist_after = (adapted[name] - updated[name]).norm().item()
        assert dist_after < dist_before + 1e-6, (
            f"Param '{name}': distance didn't decrease (before={dist_before:.4f}, after={dist_after:.4f})"
        )


def test_reptile_update_outer_lr_zero():
    """outer_lr=0 should produce no change."""
    original, adapted = _make_param_dicts()
    updated = reptile_update(original, adapted, outer_lr=0.0)
    for name in original:
        assert torch.allclose(updated[name], original[name]), (
            f"Param '{name}' changed with outer_lr=0"
        )


def test_reptile_update_outer_lr_one():
    """outer_lr=1 should equal adapted params."""
    original, adapted = _make_param_dicts()
    updated = reptile_update(original, adapted, outer_lr=1.0)
    for name in original:
        assert torch.allclose(updated[name], adapted[name]), (
            f"Param '{name}': updated != adapted with outer_lr=1"
        )


# ---------------------------------------------------------------------------
# MetaLearner.meta_step tests
# ---------------------------------------------------------------------------

def test_meta_step_returns_correct_keys(meta_learner):
    """meta_step must return dict with 'meta_loss', 'n_tasks', 'mean_inner_loss'."""
    tasks = _make_tasks(n=2)
    result = meta_learner.meta_step(tasks)
    assert "meta_loss" in result
    assert "n_tasks" in result
    assert "mean_inner_loss" in result


def test_meta_step_n_tasks_matches_input(meta_learner):
    """meta_step n_tasks must equal the number of tasks passed."""
    tasks = _make_tasks(n=3)
    result = meta_learner.meta_step(tasks)
    assert result["n_tasks"] == 3


# ---------------------------------------------------------------------------
# MetaLearner.adapt tests
# ---------------------------------------------------------------------------

def test_adapt_returns_nn_module(meta_learner, support_ids, support_labels):
    """adapt must return an nn.Module."""
    adapted_model = meta_learner.adapt(support_ids, support_labels)
    assert isinstance(adapted_model, nn.Module)


def test_adapt_changed_params_from_original(meta_learner, support_ids, support_labels):
    """Adapted model weights must differ from original model weights."""
    original_params = {
        name: p.data.clone()
        for name, p in meta_learner.model.named_parameters()
        if p.requires_grad
    }

    adapted_model = meta_learner.adapt(support_ids, support_labels)

    any_changed = any(
        not torch.equal(original_params[name], p.data)
        for name, p in adapted_model.named_parameters()
        if name in original_params
    )
    assert any_changed, "adapt() returned a model with identical weights — inner loop may not have run"
