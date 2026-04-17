"""Tests for knowledge_distillation_v2.py.

Uses tiny configurations (V=16, d=32, B=2, T=6) so tests run fast.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from aurelius.training.knowledge_distillation_v2 import (
    DistillationConfig,
    SoftTargetLoss,
    DistillationLoss,
    DistillationTrainer,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

V = 16   # vocab size
D = 32   # embedding dim
B = 2    # batch size
T = 6    # sequence length


class _TinyLM(nn.Module):
    def __init__(self, V=16, d=32):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.proj = nn.Linear(d, V)

    def forward(self, x):
        return self.proj(self.embed(x).float())


def _random_logits(requires_grad: bool = False) -> torch.Tensor:
    t = torch.randn(B, T, V)
    if requires_grad:
        t.requires_grad_(True)
    return t


def _random_labels(ignore_first: bool = False) -> torch.LongTensor:
    labels = torch.randint(0, V, (B, T))
    if ignore_first:
        labels[:, 0] = -100
    return labels


# ---------------------------------------------------------------------------
# DistillationConfig defaults
# ---------------------------------------------------------------------------

class TestDistillationConfigDefaults:
    def test_temperature_default(self):
        cfg = DistillationConfig()
        assert cfg.temperature == 4.0

    def test_alpha_default(self):
        cfg = DistillationConfig()
        assert cfg.alpha == 0.5

    def test_loss_type_default(self):
        cfg = DistillationConfig()
        assert cfg.loss_type == "forward_kl"


# ---------------------------------------------------------------------------
# SoftTargetLoss.soft_probs
# ---------------------------------------------------------------------------

class TestSoftProbs:
    def test_sums_to_one(self):
        loss = SoftTargetLoss(temperature=4.0)
        logits = _random_logits()
        probs = loss.soft_probs(logits)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_higher_temperature_more_uniform(self):
        """Higher temperature should yield a distribution closer to uniform (lower max prob)."""
        torch.manual_seed(0)
        logits = _random_logits()
        loss_low = SoftTargetLoss(temperature=1.0)
        loss_high = SoftTargetLoss(temperature=10.0)
        max_low = loss_low.soft_probs(logits).max().item()
        max_high = loss_high.soft_probs(logits).max().item()
        assert max_high < max_low, (
            f"Expected high-T max prob ({max_high:.4f}) < low-T max prob ({max_low:.4f})"
        )


# ---------------------------------------------------------------------------
# Individual KL / JSD / MSE losses
# ---------------------------------------------------------------------------

class TestSoftTargetLossVariants:
    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.loss = SoftTargetLoss(temperature=2.0)
        self.s = _random_logits()
        self.t = _random_logits()

    def test_forward_kl_scalar(self):
        result = self.loss.forward_kl(self.s, self.t)
        assert result.shape == torch.Size([])

    def test_forward_kl_finite(self):
        result = self.loss.forward_kl(self.s, self.t)
        assert torch.isfinite(result)

    def test_reverse_kl_scalar(self):
        result = self.loss.reverse_kl(self.s, self.t)
        assert result.shape == torch.Size([])

    def test_reverse_kl_finite(self):
        result = self.loss.reverse_kl(self.s, self.t)
        assert torch.isfinite(result)

    def test_jsd_scalar(self):
        result = self.loss.jsd(self.s, self.t)
        assert result.shape == torch.Size([])

    def test_jsd_finite(self):
        result = self.loss.jsd(self.s, self.t)
        assert torch.isfinite(result)

    def test_jsd_symmetric(self):
        """JSD(s, t) should equal JSD(t, s)."""
        jsd_st = self.loss.jsd(self.s, self.t)
        jsd_ts = self.loss.jsd(self.t, self.s)
        assert torch.allclose(jsd_st, jsd_ts, atol=1e-5), (
            f"JSD not symmetric: {jsd_st.item():.6f} vs {jsd_ts.item():.6f}"
        )

    def test_mse_loss_scalar(self):
        result = self.loss.mse_loss(self.s, self.t)
        assert result.shape == torch.Size([])

    def test_mse_loss_non_negative(self):
        result = self.loss.mse_loss(self.s, self.t)
        assert result.item() >= 0.0


# ---------------------------------------------------------------------------
# DistillationLoss
# ---------------------------------------------------------------------------

class TestDistillationLoss:
    def test_task_loss_handles_ignore_index(self):
        """task_loss should not crash with -100 labels and still return a finite scalar."""
        cfg = DistillationConfig()
        dl = DistillationLoss(cfg)
        logits = _random_logits()
        labels = _random_labels(ignore_first=True)
        result = dl.task_loss(logits, labels)
        assert result.shape == torch.Size([])
        assert torch.isfinite(result)

    def test_forward_returns_correct_keys(self):
        cfg = DistillationConfig()
        dl = DistillationLoss(cfg)
        s = _random_logits()
        t = _random_logits()
        labels = _random_labels()
        total, metrics = dl(s, t, labels)
        assert "task_loss" in metrics
        assert "distill_loss" in metrics
        assert "total_loss" in metrics

    def test_gradient_flows_through_student(self):
        """Gradients must flow back to student logits."""
        cfg = DistillationConfig()
        dl = DistillationLoss(cfg)
        s = _random_logits(requires_grad=True)
        t = _random_logits()
        labels = _random_labels()
        total, _ = dl(s, t, labels)
        total.backward()
        assert s.grad is not None
        assert not torch.all(s.grad == 0)

    def test_distill_loss_zero_when_identical(self):
        """forward_kl, reverse_kl, jsd and mse should all be ~0 when student == teacher."""
        for loss_type in ("forward_kl", "reverse_kl", "jsd", "mse"):
            cfg = DistillationConfig(loss_type=loss_type)
            dl = DistillationLoss(cfg)
            logits = _random_logits()
            result = dl.distill_loss(logits, logits)
            assert result.item() < 1e-4, (
                f"loss_type={loss_type}: expected ~0 for identical logits, got {result.item()}"
            )


# ---------------------------------------------------------------------------
# DistillationTrainer
# ---------------------------------------------------------------------------

class TestDistillationTrainer:
    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(7)
        self.student = _TinyLM(V=V, d=D)
        self.teacher = _TinyLM(V=V, d=D)
        cfg = DistillationConfig(loss_type="forward_kl")
        loss_fn = DistillationLoss(cfg)
        optimizer = torch.optim.SGD(self.student.parameters(), lr=1e-3)
        self.trainer = DistillationTrainer(
            self.student, self.teacher, optimizer, loss_fn
        )

    def test_freeze_teacher_all_params(self):
        """After freeze_teacher(), every teacher parameter must have requires_grad=False."""
        for name, param in self.teacher.named_parameters():
            assert not param.requires_grad, f"Teacher param '{name}' still requires grad"

    def test_train_step_returns_correct_keys(self):
        input_ids = torch.randint(0, V, (B, T))
        labels = torch.randint(0, V, (B, T))
        metrics = self.trainer.train_step(input_ids, labels)
        assert "task_loss" in metrics
        assert "distill_loss" in metrics
        assert "total_loss" in metrics
