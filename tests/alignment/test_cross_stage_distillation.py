"""Unit tests for CrossStageDistillation — GLM-5 §5.4 on-policy KL regularizer.

Tiny test config: B=2, T=8, V=32 (vocab size).
"""

import pytest
import torch

from src.alignment.cross_stage_distillation import CrossStageDistillation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T, V = 2, 8, 32


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(42)


@pytest.fixture
def random_logits(rng):
    student = torch.randn(B, T, V, generator=rng)
    teacher = torch.randn(B, T, V, generator=rng)
    return student, teacher


@pytest.fixture
def rl_loss():
    return torch.tensor(1.5)


# ---------------------------------------------------------------------------
# Test 1: alpha=0 → output equals rl_loss (KL term zeroed)
# ---------------------------------------------------------------------------


def test_alpha_zero_passthrough(random_logits, rl_loss):
    csd = CrossStageDistillation(alpha=0.0)
    student, teacher = random_logits
    out = csd.loss(rl_loss, student, teacher)
    assert torch.allclose(out, rl_loss), f"With alpha=0 expected output == rl_loss, got {out}"


# ---------------------------------------------------------------------------
# Test 2: KL is non-negative → total loss >= rl_loss when alpha > 0
# ---------------------------------------------------------------------------


def test_kl_nonnegative_total_loss_ge_rl(random_logits, rl_loss):
    csd = CrossStageDistillation(alpha=0.5)
    student, teacher = random_logits
    out = csd.loss(rl_loss, student, teacher)
    assert out.item() >= rl_loss.item() - 1e-6, (
        f"Expected total loss >= rl_loss, got {out} < {rl_loss}"
    )


# ---------------------------------------------------------------------------
# Test 3: student == teacher (identical logits) → KL ≈ 0 → loss ≈ rl_loss
# ---------------------------------------------------------------------------


def test_identical_logits_kl_zero(rl_loss):
    csd = CrossStageDistillation(alpha=1.0)
    logits = torch.randn(B, T, V)
    out = csd.loss(rl_loss, logits, logits.clone())
    assert torch.allclose(out, rl_loss, atol=1e-5), (
        f"Identical student/teacher should yield loss ≈ rl_loss, got {out}"
    )


# ---------------------------------------------------------------------------
# Test 4: gradient flows through student_logits, not teacher
# ---------------------------------------------------------------------------


def test_gradient_flows_through_student(rl_loss):
    csd = CrossStageDistillation(alpha=0.2)
    student = torch.randn(B, T, V, requires_grad=True)
    teacher = torch.randn(B, T, V)  # no requires_grad

    out = csd.loss(rl_loss, student, teacher)
    out.backward()

    assert student.grad is not None, "Gradient must flow through student_logits"
    assert not torch.all(student.grad == 0), "student.grad should not be all zeros"


# ---------------------------------------------------------------------------
# Test 5: teacher_logits detached → teacher grad is None after backward
# ---------------------------------------------------------------------------


def test_teacher_grad_none_after_backward(rl_loss):
    csd = CrossStageDistillation(alpha=0.2)
    student = torch.randn(B, T, V, requires_grad=True)
    teacher = torch.randn(B, T, V, requires_grad=True)

    out = csd.loss(rl_loss, student, teacher)
    out.backward()

    assert teacher.grad is None, "Teacher logits are detached — teacher.grad must remain None"


# ---------------------------------------------------------------------------
# Test 6: attention_mask all zeros → KL contribution zeroed out
# ---------------------------------------------------------------------------


def test_all_zero_mask_zeroes_kl(random_logits, rl_loss):
    csd = CrossStageDistillation(alpha=1.0)
    student, teacher = random_logits
    mask = torch.zeros(B, T)
    out = csd.loss(rl_loss, student, teacher, attention_mask=mask)
    # With all-zero mask: kl.sum() / (0 + 1e-8) → effectively 0
    assert torch.allclose(out, rl_loss, atol=1e-4), (
        f"All-zero mask should zero KL contribution; got {out} vs {rl_loss}"
    )


# ---------------------------------------------------------------------------
# Test 7: attention_mask all ones → same result as no mask
# ---------------------------------------------------------------------------


def test_all_ones_mask_same_as_no_mask(random_logits, rl_loss):
    csd = CrossStageDistillation(alpha=0.3)
    student, teacher = random_logits
    mask = torch.ones(B, T)

    out_masked = csd.loss(rl_loss, student, teacher, attention_mask=mask)
    out_no_mask = csd.loss(rl_loss, student, teacher, attention_mask=None)

    assert torch.allclose(out_masked, out_no_mask, atol=1e-5), (
        f"All-ones mask should give same result as no mask: {out_masked} vs {out_no_mask}"
    )


# ---------------------------------------------------------------------------
# Test 8: finite output on random logits (no NaN / Inf)
# ---------------------------------------------------------------------------


def test_finite_output_random_logits(rl_loss):
    csd = CrossStageDistillation(alpha=0.1)
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        student = torch.randn(B, T, V, generator=g)
        teacher = torch.randn(B, T, V, generator=g)
        out = csd.loss(rl_loss, student, teacher)
        assert torch.isfinite(out), f"Expected finite output, got {out} (seed={seed})"


# ---------------------------------------------------------------------------
# Test 9: alpha=1.0 → loss > rl_loss when student != teacher
# ---------------------------------------------------------------------------


def test_alpha_one_increases_loss(random_logits, rl_loss):
    csd = CrossStageDistillation(alpha=1.0)
    student, teacher = random_logits
    # Ensure student and teacher are meaningfully different
    teacher = teacher + 5.0
    out = csd.loss(rl_loss, student, teacher)
    assert out.item() > rl_loss.item(), (
        f"alpha=1.0 with different student/teacher should give loss > rl_loss; got {out}"
    )


# ---------------------------------------------------------------------------
# Test 10: doubling alpha doubles the KL contribution
# ---------------------------------------------------------------------------


def test_doubling_alpha_doubles_kl_contribution(random_logits, rl_loss):
    student, teacher = random_logits

    csd1 = CrossStageDistillation(alpha=0.2)
    csd2 = CrossStageDistillation(alpha=0.4)

    out1 = csd1.loss(rl_loss, student, teacher)
    out2 = csd2.loss(rl_loss, student, teacher)

    kl_contrib1 = (out1 - rl_loss).item()
    kl_contrib2 = (out2 - rl_loss).item()

    assert abs(kl_contrib2 - 2.0 * kl_contrib1) < 1e-4, (
        f"Doubling alpha should double KL contribution: 2×{kl_contrib1:.6f} vs {kl_contrib2:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 11: batch_size=1, seq_len=1 edge case
# ---------------------------------------------------------------------------


def test_single_token_edge_case():
    csd = CrossStageDistillation(alpha=0.1)
    rl_loss = torch.tensor(0.5)
    student = torch.randn(1, 1, V)
    teacher = torch.randn(1, 1, V)
    out = csd.loss(rl_loss, student, teacher)
    assert torch.isfinite(out), f"Single-token edge case should produce finite output: {out}"
    assert out.shape == torch.Size([]), f"Output must be scalar, got shape {out.shape}"


# ---------------------------------------------------------------------------
# Test 12: output is scalar (shape == [])
# ---------------------------------------------------------------------------


def test_output_is_scalar(random_logits, rl_loss):
    csd = CrossStageDistillation(alpha=0.1)
    student, teacher = random_logits
    out = csd.loss(rl_loss, student, teacher)
    assert out.shape == torch.Size([]), f"Output must be a scalar tensor, got shape {out.shape}"


# ---------------------------------------------------------------------------
# Bonus Test 13: partial attention mask (mix of 0s and 1s)
# ---------------------------------------------------------------------------


def test_partial_attention_mask(random_logits, rl_loss):
    """Partial mask should give a result strictly between the all-zero and all-ones cases."""
    csd = CrossStageDistillation(alpha=0.5)
    student, teacher = random_logits
    # Make student and teacher clearly different
    teacher = teacher + 3.0

    out_zero = csd.loss(rl_loss, student, teacher, attention_mask=torch.zeros(B, T))
    out_ones = csd.loss(rl_loss, student, teacher, attention_mask=torch.ones(B, T))

    # Mask half the tokens
    mask = torch.zeros(B, T)
    mask[:, : T // 2] = 1.0
    out_partial = csd.loss(rl_loss, student, teacher, attention_mask=mask)

    assert out_zero.item() <= out_partial.item() + 1e-5
    assert out_partial.item() <= out_ones.item() + 1e-5
