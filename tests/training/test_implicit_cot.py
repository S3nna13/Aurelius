"""Tests for src/training/implicit_cot.py — Implicit Chain-of-Thought training."""
from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn

from aurelius.training.implicit_cot import (
    ImplicitCoTConfig,
    ImplicitCoTLoss,
    ImplicitCoTTrainer,
    ReasoningStateProjector,
)


# ---------------------------------------------------------------------------
# Tiny language model used throughout the tests
# ---------------------------------------------------------------------------


class _TinyLM(nn.Module):
    """Minimal model that returns (logits, hidden) for testing."""

    def __init__(self, V: int = 16, d: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.proj = nn.Linear(d, V)

    def forward(self, x: torch.Tensor):  # (B, T) → ((B, T, V), (B, T, d))
        h = self.embed(x)
        return self.proj(h), h


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B, T, V, D = 2, 8, 16, 32  # batch, seq-len, vocab, hidden-dim


@pytest.fixture()
def config() -> ImplicitCoTConfig:
    return ImplicitCoTConfig(
        n_reasoning_steps=4,
        hidden_dim=D,
        distill_weight=0.5,
        temperature=2.0,
    )


@pytest.fixture()
def loss_fn(config: ImplicitCoTConfig) -> ImplicitCoTLoss:
    return ImplicitCoTLoss(config)


@pytest.fixture()
def student() -> _TinyLM:
    torch.manual_seed(0)
    return _TinyLM(V=V, d=D)


@pytest.fixture()
def teacher() -> _TinyLM:
    torch.manual_seed(1)
    return _TinyLM(V=V, d=D)


@pytest.fixture()
def trainer(student, teacher, loss_fn) -> ImplicitCoTTrainer:
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    t = ImplicitCoTTrainer(student, teacher, opt, loss_fn)
    t.freeze_teacher()
    return t


@pytest.fixture()
def input_ids() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randint(0, V, (B, T))


@pytest.fixture()
def labels(input_ids: torch.Tensor) -> torch.Tensor:
    return input_ids.clone()


# ---------------------------------------------------------------------------
# 1. ImplicitCoTConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = ImplicitCoTConfig()
    assert cfg.n_reasoning_steps == 4
    assert cfg.hidden_dim == 512
    assert cfg.distill_weight == 0.5
    assert cfg.temperature == 2.0


# ---------------------------------------------------------------------------
# 2. ReasoningStateProjector output shape
# ---------------------------------------------------------------------------


def test_projector_output_shape():
    proj = ReasoningStateProjector(student_dim=D, teacher_dim=64)
    x = torch.randn(B, T, D)
    out = proj(x)
    assert out.shape == (B, T, 64), f"Expected (B,T,64) got {out.shape}"


# ---------------------------------------------------------------------------
# 3. ReasoningStateProjector gradient flows
# ---------------------------------------------------------------------------


def test_projector_gradient_flows():
    proj = ReasoningStateProjector(student_dim=D, teacher_dim=64)
    x = torch.randn(B, T, D, requires_grad=True)
    out = proj(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient did not flow to input"
    assert proj.proj.weight.grad is not None, "Gradient did not reach proj weight"


# ---------------------------------------------------------------------------
# 4. ImplicitCoTLoss.task_loss — scalar and finite
# ---------------------------------------------------------------------------


def test_task_loss_scalar_and_finite(loss_fn):
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    loss = loss_fn.task_loss(logits, labels)
    assert loss.ndim == 0, "task_loss should be a scalar"
    assert math.isfinite(loss.item()), "task_loss is not finite"


# ---------------------------------------------------------------------------
# 5. ImplicitCoTLoss.task_loss handles -100 labels
# ---------------------------------------------------------------------------


def test_task_loss_handles_ignore_index(loss_fn):
    logits = torch.randn(B, T, V)
    labels = torch.full((B, T), -100, dtype=torch.long)
    # With all labels masked the CE loss should be 0 (no active tokens)
    loss = loss_fn.task_loss(logits, labels)
    assert loss.ndim == 0
    # When every token is masked the implementation returns 0.0 (not NaN)
    assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# 6. ImplicitCoTLoss.distillation_loss — scalar and finite
# ---------------------------------------------------------------------------


def test_distillation_loss_scalar_and_finite(loss_fn):
    s_hidden = torch.randn(B, T, D)
    t_hidden = torch.randn(B, T, D)
    loss = loss_fn.distillation_loss(s_hidden, t_hidden)
    assert loss.ndim == 0, "distillation_loss should be scalar"
    assert math.isfinite(loss.item()), "distillation_loss is not finite"


# ---------------------------------------------------------------------------
# 7. ImplicitCoTLoss.distillation_loss — mismatched dims handled via interp
# ---------------------------------------------------------------------------


def test_distillation_loss_mismatched_dims(loss_fn):
    s_hidden = torch.randn(B, T, D)       # student dim = 32
    t_hidden = torch.randn(B, T, D * 2)   # teacher dim = 64
    loss = loss_fn.distillation_loss(s_hidden, t_hidden)
    assert loss.ndim == 0
    assert math.isfinite(loss.item())


# ---------------------------------------------------------------------------
# 8. ImplicitCoTLoss.forward — correct dict keys
# ---------------------------------------------------------------------------


def test_forward_returns_correct_keys(loss_fn):
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    s_hidden = torch.randn(B, T, D)
    t_hidden = torch.randn(B, T, D)
    _, metrics = loss_fn(logits, labels, s_hidden, t_hidden)
    assert set(metrics.keys()) == {"task_loss", "distill_loss", "total_loss"}


# ---------------------------------------------------------------------------
# 9. ImplicitCoTLoss total is correct weighted combination
# ---------------------------------------------------------------------------


def test_total_is_weighted_combination(config):
    config.distill_weight = 0.3
    loss_fn = ImplicitCoTLoss(config)

    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    s_hidden = torch.randn(B, T, D)
    t_hidden = torch.randn(B, T, D)

    total, metrics = loss_fn(logits, labels, s_hidden, t_hidden)

    expected = 0.7 * metrics["task_loss"] + 0.3 * metrics["distill_loss"]
    assert torch.allclose(total, expected, atol=1e-6), (
        f"total={total.item():.6f} expected={expected.item():.6f}"
    )


# ---------------------------------------------------------------------------
# 10. ImplicitCoTTrainer.freeze_teacher — all teacher params frozen
# ---------------------------------------------------------------------------


def test_freeze_teacher_freezes_all_params(trainer):
    for name, param in trainer.teacher.named_parameters():
        assert not param.requires_grad, f"param {name} should be frozen"


# ---------------------------------------------------------------------------
# 11. ImplicitCoTTrainer.train_step — returns correct keys
# ---------------------------------------------------------------------------


def test_train_step_returns_correct_keys(trainer, input_ids, labels):
    result = trainer.train_step(input_ids, labels)
    assert set(result.keys()) == {"task_loss", "distill_loss", "total_loss"}


# ---------------------------------------------------------------------------
# 12. ImplicitCoTTrainer.train_step — loss values are finite
# ---------------------------------------------------------------------------


def test_train_step_loss_is_finite(trainer, input_ids, labels):
    result = trainer.train_step(input_ids, labels)
    for key, val in result.items():
        assert math.isfinite(val), f"{key}={val} is not finite"


# ---------------------------------------------------------------------------
# 13. ImplicitCoTTrainer.train_step — student weights update
# ---------------------------------------------------------------------------


def test_train_step_updates_student_weights(trainer, input_ids, labels):
    # Snapshot student weights before a step
    before = {
        name: param.data.clone()
        for name, param in trainer.student.named_parameters()
    }

    trainer.train_step(input_ids, labels)

    changed = False
    for name, param in trainer.student.named_parameters():
        if not torch.equal(param.data, before[name]):
            changed = True
            break

    assert changed, "Student weights did not change after train_step"
