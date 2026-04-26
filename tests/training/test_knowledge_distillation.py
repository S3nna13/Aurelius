"""Tests for knowledge_distillation.py.

Uses tiny configurations (VOCAB=16, D=8, B=2, T=6) so tests run fast.
Small nn.Module-based student/teacher models ensure differentiable training.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.training.knowledge_distillation import (
    KDConfig,
    KnowledgeDistillationTrainer,
    combined_kd_loss,
    forward_kl_loss,
    layer_wise_distillation_loss,
    mse_distillation_loss,
    reverse_kl_loss,
    soft_cross_entropy,
)

# ---------------------------------------------------------------------------
# Shared tiny dimensions
# ---------------------------------------------------------------------------
VOCAB = 16
D = 8
B = 2
T = 6


# ---------------------------------------------------------------------------
# Tiny model helpers
# ---------------------------------------------------------------------------


class TinyLM(nn.Module):
    """Minimal token-level language model: Embedding -> Linear -> logits."""

    def __init__(self, vocab: int = VOCAB, d: int = D) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.proj = nn.Linear(d, vocab)

    def forward(self, token_ids: Tensor) -> Tensor:
        """(B, T) -> (B, T, V)"""
        return self.proj(self.embed(token_ids))


def make_student() -> TinyLM:
    torch.manual_seed(0)
    return TinyLM()


def make_teacher() -> TinyLM:
    torch.manual_seed(1)
    m = TinyLM()
    # Put teacher in inference mode (no grad tracking)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def make_student_fn(model: TinyLM):
    """Wrap a TinyLM as a plain callable."""

    def fn(token_ids: Tensor) -> Tensor:
        return model(token_ids)

    return fn


def make_teacher_fn(model: TinyLM):
    def fn(token_ids: Tensor) -> Tensor:
        return model(token_ids)

    return fn


# ---------------------------------------------------------------------------
# 1. KDConfig defaults
# ---------------------------------------------------------------------------


def test_kdconfig_defaults():
    cfg = KDConfig()
    assert cfg.temperature == 4.0
    assert cfg.alpha == 0.5
    assert cfg.kd_loss_type == "forward_kl"


# ---------------------------------------------------------------------------
# 2. soft_cross_entropy returns a scalar
# ---------------------------------------------------------------------------


def test_soft_cross_entropy_scalar():
    s = torch.randn(B, T, VOCAB)
    t = torch.randn(B, T, VOCAB)
    loss = soft_cross_entropy(s, t, temperature=4.0)
    assert loss.ndim == 0, "Expected scalar (0-dim tensor)"


# ---------------------------------------------------------------------------
# 3. soft_cross_entropy >= 0
# ---------------------------------------------------------------------------


def test_soft_cross_entropy_nonnegative():
    torch.manual_seed(42)
    s = torch.randn(B, T, VOCAB)
    t = torch.randn(B, T, VOCAB)
    loss = soft_cross_entropy(s, t, temperature=4.0)
    assert loss.item() >= -1e-6, f"Expected non-negative loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# 4. forward_kl_loss >= 0
# ---------------------------------------------------------------------------


def test_forward_kl_nonnegative():
    torch.manual_seed(7)
    s = torch.randn(B, T, VOCAB)
    t = torch.randn(B, T, VOCAB)
    loss = forward_kl_loss(s, t, temperature=4.0)
    assert loss.item() >= -1e-6, f"Expected non-negative loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# 5. reverse_kl_loss >= 0
# ---------------------------------------------------------------------------


def test_reverse_kl_nonnegative():
    torch.manual_seed(13)
    s = torch.randn(B, T, VOCAB)
    t = torch.randn(B, T, VOCAB)
    loss = reverse_kl_loss(s, t, temperature=4.0)
    assert loss.item() >= -1e-6, f"Expected non-negative loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# 6. mse_distillation_loss >= 0
# ---------------------------------------------------------------------------


def test_mse_distillation_nonnegative():
    torch.manual_seed(99)
    s = torch.randn(B, T, VOCAB)
    t = torch.randn(B, T, VOCAB)
    loss = mse_distillation_loss(s, t)
    assert loss.item() >= 0.0, f"Expected non-negative loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# 7. combined_kd_loss returns (total, kd, ce) — 3 tensors
# ---------------------------------------------------------------------------


def test_combined_kd_loss_returns_three_tensors():
    s = torch.randn(B, T, VOCAB)
    t = torch.randn(B, T, VOCAB)
    labels = torch.randint(0, VOCAB, (B, T))
    cfg = KDConfig()
    result = combined_kd_loss(s, t, labels, cfg)
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 3, "Result should have exactly 3 elements"
    total, kd, ce = result
    assert isinstance(total, torch.Tensor)
    assert isinstance(kd, torch.Tensor)
    assert isinstance(ce, torch.Tensor)


# ---------------------------------------------------------------------------
# 8. total = alpha * kd + (1 - alpha) * ce
# ---------------------------------------------------------------------------


def test_combined_kd_loss_total_formula():
    torch.manual_seed(5)
    s = torch.randn(B, T, VOCAB)
    t = torch.randn(B, T, VOCAB)
    labels = torch.randint(0, VOCAB, (B, T))
    alpha = 0.3
    cfg = KDConfig(alpha=alpha)
    total, kd, ce = combined_kd_loss(s, t, labels, cfg)
    expected = alpha * kd.item() + (1.0 - alpha) * ce.item()
    assert abs(total.item() - expected) < 1e-5, f"total={total.item():.6f}, expected={expected:.6f}"


# ---------------------------------------------------------------------------
# 9. train_step returns correct keys
# ---------------------------------------------------------------------------


def test_train_step_keys():
    student = make_student()
    teacher = make_teacher()
    optimizer = torch.optim.SGD(student.parameters(), lr=1e-3)
    cfg = KDConfig()

    trainer = KnowledgeDistillationTrainer(
        student_fn=make_student_fn(student),
        teacher_fn=make_teacher_fn(teacher),
        optimizer=optimizer,
        config=cfg,
    )
    token_ids = torch.randint(0, VOCAB, (B, T))
    labels = torch.randint(0, VOCAB, (B, T))
    result = trainer.train_step(token_ids, labels)

    assert set(result.keys()) == {"total_loss", "kd_loss", "ce_loss"}, (
        f"Unexpected keys: {result.keys()}"
    )


# ---------------------------------------------------------------------------
# 10. train_step loss is finite
# ---------------------------------------------------------------------------


def test_train_step_loss_finite():
    student = make_student()
    teacher = make_teacher()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    cfg = KDConfig()

    trainer = KnowledgeDistillationTrainer(
        student_fn=make_student_fn(student),
        teacher_fn=make_teacher_fn(teacher),
        optimizer=optimizer,
        config=cfg,
    )
    token_ids = torch.randint(0, VOCAB, (B, T))
    labels = torch.randint(0, VOCAB, (B, T))
    result = trainer.train_step(token_ids, labels)

    assert math.isfinite(result["total_loss"]), "total_loss should be finite"
    assert math.isfinite(result["kd_loss"]), "kd_loss should be finite"
    assert math.isfinite(result["ce_loss"]), "ce_loss should be finite"


# ---------------------------------------------------------------------------
# 11. evaluate runs under no_grad (student params unchanged)
# ---------------------------------------------------------------------------


def test_evaluate_no_grad():
    student = make_student()
    teacher = make_teacher()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    cfg = KDConfig()

    trainer = KnowledgeDistillationTrainer(
        student_fn=make_student_fn(student),
        teacher_fn=make_teacher_fn(teacher),
        optimizer=optimizer,
        config=cfg,
    )
    token_ids = torch.randint(0, VOCAB, (B, T))
    labels = torch.randint(0, VOCAB, (B, T))

    # Record student weights before evaluate
    before = {n: p.detach().clone() for n, p in student.named_parameters()}
    result = trainer.evaluate(token_ids, labels)
    after = {n: p.detach().clone() for n, p in student.named_parameters()}

    assert set(result.keys()) == {"total_loss", "kd_loss", "ce_loss"}
    for n in before:
        assert torch.allclose(before[n], after[n]), (
            f"Student parameter '{n}' changed during evaluate()"
        )


# ---------------------------------------------------------------------------
# 12. layer_wise_distillation_loss shape is () (scalar)
# ---------------------------------------------------------------------------


def test_layer_wise_distillation_loss_scalar():
    s = torch.randn(B, T, D)
    t = torch.randn(B, T, D)
    loss = layer_wise_distillation_loss(s, t)
    assert loss.shape == torch.Size([]), "Expected scalar (0-dim tensor)"


# ---------------------------------------------------------------------------
# 13. temperature=1 gives regular (un-softened) KD
# ---------------------------------------------------------------------------


def test_temperature_one_matches_manual_kl():
    """At T=1, soft_cross_entropy should match plain KL(teacher||student)."""
    torch.manual_seed(42)
    s = torch.randn(B, T, VOCAB)
    t = torch.randn(B, T, VOCAB)
    loss_t1 = soft_cross_entropy(s, t, temperature=1.0)

    # Compute manually (T^2 = 1, so no scaling difference)
    V = VOCAB
    s_log = F.log_softmax(s, dim=-1)
    t_prob = F.softmax(t, dim=-1)
    manual = F.kl_div(s_log.reshape(-1, V), t_prob.reshape(-1, V), reduction="batchmean")

    assert abs(loss_t1.item() - manual.item()) < 1e-5, (
        f"T=1 mismatch: {loss_t1.item()} vs {manual.item()}"
    )


# ---------------------------------------------------------------------------
# 14. alpha=1.0 makes total_loss == kd_loss (CE weight is 0)
# ---------------------------------------------------------------------------


def test_alpha_one_ignores_ce():
    torch.manual_seed(77)
    s = torch.randn(B, T, VOCAB)
    t = torch.randn(B, T, VOCAB)
    labels = torch.randint(0, VOCAB, (B, T))
    cfg = KDConfig(alpha=1.0)
    total, kd, ce = combined_kd_loss(s, t, labels, cfg)
    assert abs(total.item() - kd.item()) < 1e-5, (
        f"With alpha=1.0: total={total.item()}, kd={kd.item()}, ce={ce.item()}"
    )


# ---------------------------------------------------------------------------
# 15. layer_wise_distillation_loss raises ValueError on shape mismatch
# ---------------------------------------------------------------------------


def test_layer_wise_distillation_loss_shape_mismatch_raises():
    s = torch.randn(B, T, D)
    t = torch.randn(B, T, D * 2)  # different hidden dim
    with pytest.raises(ValueError, match="Shape mismatch"):
        layer_wise_distillation_loss(s, t)


# ---------------------------------------------------------------------------
# 16. mse_distillation_loss is 0 when inputs are identical
# ---------------------------------------------------------------------------


def test_mse_distillation_loss_zero_on_equal():
    x = torch.randn(B, T, VOCAB)
    loss = mse_distillation_loss(x, x.clone())
    assert loss.item() < 1e-6, f"Expected ~0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 17. reverse_kl != forward_kl for asymmetric distributions
# ---------------------------------------------------------------------------


def test_forward_reverse_kl_differ():
    torch.manual_seed(200)
    s2 = torch.randn(B, T, VOCAB) * 3.0
    t2 = torch.randn(B, T, VOCAB) * 0.5
    fwd2 = forward_kl_loss(s2, t2, temperature=1.0)
    rev2 = reverse_kl_loss(s2, t2, temperature=1.0)
    diff = abs(fwd2.item() - rev2.item())
    assert diff > 1e-4, (
        f"forward_kl={fwd2.item():.6f} and reverse_kl={rev2.item():.6f} "
        "are suspiciously equal — KL divergence should be asymmetric"
    )


# ---------------------------------------------------------------------------
# 18. combined_kd_loss works with "mse" and "reverse_kl" types
# ---------------------------------------------------------------------------


def test_combined_kd_loss_mse_type():
    s = torch.randn(B, T, VOCAB)
    t = torch.randn(B, T, VOCAB)
    labels = torch.randint(0, VOCAB, (B, T))
    cfg = KDConfig(kd_loss_type="mse")
    total, kd, ce = combined_kd_loss(s, t, labels, cfg)
    assert total.ndim == 0
    assert kd.item() >= 0.0


def test_combined_kd_loss_reverse_kl_type():
    s = torch.randn(B, T, VOCAB)
    t = torch.randn(B, T, VOCAB)
    labels = torch.randint(0, VOCAB, (B, T))
    cfg = KDConfig(kd_loss_type="reverse_kl")
    total, kd, ce = combined_kd_loss(s, t, labels, cfg)
    assert total.ndim == 0
    assert kd.item() >= -1e-6
