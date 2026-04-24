"""Tests for ContextDistillationTrainer (arXiv:2209.15189)."""
import math
import pytest
import torch
import torch.nn as nn

from src.training.context_distillation_trainer import (
    ContextDistillationTrainer,
    ContextDistillConfig,
)


def _make_pair(in_dim=8, out_dim=8):
    torch.manual_seed(42)
    student = nn.Linear(in_dim, out_dim)
    torch.manual_seed(7)
    teacher = nn.Linear(in_dim, out_dim)
    return student, teacher


def _logits(batch=2, seq=4, vocab=8):
    return torch.randn(batch, seq, vocab)


class TestContextDistillConfig:
    def test_defaults(self):
        cfg = ContextDistillConfig()
        assert cfg.alpha == 0.5
        assert cfg.temperature == 2.0
        assert cfg.context_prefix == ""
        assert cfg.max_seq_len == 2048

    def test_custom(self):
        cfg = ContextDistillConfig(alpha=0.3, temperature=4.0, max_seq_len=512)
        assert cfg.alpha == 0.3
        assert cfg.temperature == 4.0
        assert cfg.max_seq_len == 512


class TestDistillLoss:
    def setup_method(self):
        student, teacher = _make_pair()
        self.trainer = ContextDistillationTrainer(student, teacher)

    def test_returns_scalar(self):
        s, t = _logits(), _logits()
        loss = self.trainer.distill_loss(s, t)
        assert loss.ndim == 0

    def test_nonnegative(self):
        s, t = _logits(), _logits()
        loss = self.trainer.distill_loss(s, t)
        assert loss.item() >= 0.0

    def test_identical_logits_near_zero(self):
        logits = _logits()
        loss = self.trainer.distill_loss(logits, logits.clone())
        assert loss.item() < 1e-4

    def test_finite(self):
        s, t = _logits(), _logits()
        loss = self.trainer.distill_loss(s, t)
        assert math.isfinite(loss.item())

    def test_with_labels_returns_scalar(self):
        s, t = _logits(), _logits()
        labels = torch.randint(0, 8, (2, 4))
        loss = self.trainer.distill_loss(s, t, labels)
        assert loss.ndim == 0

    def test_alpha_zero_no_kl_when_identical(self):
        cfg = ContextDistillConfig(alpha=0.0)
        student, teacher = _make_pair()
        trainer = ContextDistillationTrainer(student, teacher, cfg)
        logits = _logits()
        labels = torch.randint(0, 8, (2, 4))
        loss = trainer.distill_loss(logits, logits.clone(), labels)
        assert loss.item() >= 0.0

    def test_alpha_one_kl_dominates(self):
        cfg = ContextDistillConfig(alpha=1.0)
        student, teacher = _make_pair()
        trainer = ContextDistillationTrainer(student, teacher, cfg)
        s, t = _logits(), _logits()
        labels = torch.randint(0, 8, (2, 4))
        loss_with_labels = trainer.distill_loss(s, t, labels)
        loss_no_labels = trainer.distill_loss(s, t)
        assert abs(loss_with_labels.item() - loss_no_labels.item()) < 1e-4

    def test_temperature_effect(self):
        cfg_low = ContextDistillConfig(temperature=1.0)
        cfg_high = ContextDistillConfig(temperature=10.0)
        s, t = _make_pair()
        t1 = ContextDistillationTrainer(s, t, cfg_low)
        t2 = ContextDistillationTrainer(s, t, cfg_high)
        logits_s, logits_t = _logits(), _logits()
        loss_low = t1.distill_loss(logits_s, logits_t)
        loss_high = t2.distill_loss(logits_s, logits_t)
        assert loss_low.item() != loss_high.item()

    def test_ignore_index_in_labels(self):
        s, t = _logits(), _logits()
        labels = torch.full((2, 4), -100, dtype=torch.long)
        loss = self.trainer.distill_loss(s, t, labels)
        assert math.isfinite(loss.item())


class TestTrainStep:
    def test_returns_dict_keys(self):
        student, teacher = _make_pair()
        trainer = ContextDistillationTrainer(student, teacher)
        input_ids = torch.randint(0, 8, (1, 4))
        result = trainer.train_step(input_ids)
        assert "distill_loss" in result
        assert "ce_loss" in result
        assert "total_loss" in result

    def test_distill_loss_nonnegative(self):
        student, teacher = _make_pair()
        trainer = ContextDistillationTrainer(student, teacher)
        input_ids = torch.randint(0, 8, (1, 4))
        result = trainer.train_step(input_ids)
        assert result["distill_loss"] >= 0.0

    def test_total_equals_distill_when_no_labels(self):
        student, teacher = _make_pair()
        trainer = ContextDistillationTrainer(student, teacher)
        input_ids = torch.randint(0, 8, (1, 4))
        result = trainer.train_step(input_ids)
        assert abs(result["total_loss"] - result["distill_loss"]) < 1e-6

    def test_with_context_ids(self):
        student, teacher = _make_pair()
        trainer = ContextDistillationTrainer(student, teacher)
        input_ids = torch.randint(0, 8, (1, 4))
        context_ids = torch.randint(0, 8, (1, 4))
        result = trainer.train_step(input_ids, context_ids=context_ids)
        assert "distill_loss" in result

    def test_finite_loss(self):
        student, teacher = _make_pair()
        trainer = ContextDistillationTrainer(student, teacher)
        input_ids = torch.randint(0, 8, (1, 4))
        result = trainer.train_step(input_ids)
        assert math.isfinite(result["total_loss"])


class TestEvaluateTransfer:
    def test_returns_dict_keys(self):
        student, teacher = _make_pair()
        trainer = ContextDistillationTrainer(student, teacher)
        eval_ids = torch.randint(0, 8, (1, 4))
        result = trainer.evaluate_transfer(eval_ids)
        assert "mean_kl" in result
        assert "max_kl" in result
        assert "student_perplexity" in result

    def test_mean_kl_nonnegative(self):
        student, teacher = _make_pair()
        trainer = ContextDistillationTrainer(student, teacher)
        eval_ids = torch.randint(0, 8, (1, 4))
        result = trainer.evaluate_transfer(eval_ids)
        assert result["mean_kl"] >= 0.0

    def test_max_kl_gte_mean_kl(self):
        student, teacher = _make_pair()
        trainer = ContextDistillationTrainer(student, teacher)
        eval_ids = torch.randint(0, 8, (1, 4))
        result = trainer.evaluate_transfer(eval_ids)
        assert result["max_kl"] >= result["mean_kl"] - 1e-6

    def test_perplexity_positive(self):
        student, teacher = _make_pair()
        trainer = ContextDistillationTrainer(student, teacher)
        eval_ids = torch.randint(0, 8, (1, 4))
        result = trainer.evaluate_transfer(eval_ids)
        assert result["student_perplexity"] > 0.0

    def test_all_finite(self):
        student, teacher = _make_pair()
        trainer = ContextDistillationTrainer(student, teacher)
        eval_ids = torch.randint(0, 8, (1, 4))
        result = trainer.evaluate_transfer(eval_ids)
        assert math.isfinite(result["mean_kl"])
        assert math.isfinite(result["max_kl"])
        assert math.isfinite(result["student_perplexity"])

    def test_default_config_used_when_none(self):
        student, teacher = _make_pair()
        trainer = ContextDistillationTrainer(student, teacher, config=None)
        assert trainer.config.alpha == 0.5
