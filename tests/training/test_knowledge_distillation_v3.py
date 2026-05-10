"""Tests for knowledge_distillation_v3.py.

Uses tiny configurations:
  student_dim=8, teacher_dim=16, T=4.0, alpha=0.7,
  seq_len=8, batch=2, vocab=16.

Every test performs actual forward (and where relevant, backward) passes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from aurelius.training.knowledge_distillation_v3 import (
    AttentionTransferLoss,
    CompressionBenchmark,
    DistilTrainer,
    FeatDistilLoss,
    PKDLoss,
    SoftTargetLoss,
)
from torch import Tensor

# ---------------------------------------------------------------------------
# Tiny config constants
# ---------------------------------------------------------------------------

BATCH = 2
SEQ_LEN = 8
VOCAB = 16
STUDENT_DIM = 8
TEACHER_DIM = 16
TEMPERATURE = 4.0
ALPHA = 0.7
N_HEADS_S = 2  # student attention heads
N_HEADS_T = 4  # teacher attention heads


# ---------------------------------------------------------------------------
# Minimal stub models for DistilTrainer tests
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Minimal LM-like model returning (logits, hidden_states, attn_maps)."""

    def __init__(self, d_model: int, n_heads: int, vocab: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.proj = nn.Linear(d_model, vocab)
        self.d_model = d_model
        self.n_heads = n_heads

    def forward(self, input_ids: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        B, T = input_ids.shape
        h = self.embed(input_ids)  # (B, T, d_model)
        logits = self.proj(h)  # (B, T, vocab)
        hidden_states = [h]
        # Attention maps built from a learnable parameter so grads flow
        raw = torch.ones(B, self.n_heads, T, T, device=h.device, dtype=h.dtype)
        attn = raw / T
        attn_maps = [attn]
        return logits, hidden_states, attn_maps


def make_student() -> TinyModel:
    return TinyModel(d_model=STUDENT_DIM, n_heads=N_HEADS_S, vocab=VOCAB)


def make_teacher() -> TinyModel:
    return TinyModel(d_model=TEACHER_DIM, n_heads=N_HEADS_T, vocab=VOCAB)


# ---------------------------------------------------------------------------
# Test 1: SoftTargetLoss -- all outputs finite, backward runs
# ---------------------------------------------------------------------------


def test_soft_target_loss_finite():
    loss_fn = SoftTargetLoss(temperature=TEMPERATURE, alpha=ALPHA)
    # requires_grad=True so backward is possible
    student_logits = torch.randn(BATCH, VOCAB, requires_grad=True)
    teacher_logits = torch.randn(BATCH, VOCAB)
    labels = torch.randint(0, VOCAB, (BATCH,))

    total_loss, kd_loss, ce_loss = loss_fn(student_logits, teacher_logits, labels)

    # Forward pass confirmed; check finiteness
    assert torch.isfinite(total_loss), "total_loss is not finite"
    assert torch.isfinite(kd_loss), "kd_loss is not finite"
    assert torch.isfinite(ce_loss), "ce_loss is not finite"

    # Backward pass
    total_loss.backward()
    assert student_logits.grad is not None, "Expected grad on student_logits"


# ---------------------------------------------------------------------------
# Test 2: SoftTargetLoss -- ce_loss matches standard cross-entropy
# ---------------------------------------------------------------------------


def test_soft_target_ce_matches_standard():
    loss_fn = SoftTargetLoss(temperature=TEMPERATURE, alpha=ALPHA)
    student_logits = torch.randn(BATCH, VOCAB)
    teacher_logits = torch.randn(BATCH, VOCAB)
    labels = torch.randint(0, VOCAB, (BATCH,))

    _, _, ce_loss = loss_fn(student_logits, teacher_logits, labels)
    expected_ce = F.cross_entropy(student_logits, labels)

    assert torch.allclose(ce_loss, expected_ce, atol=1e-6), (
        f"ce_loss {ce_loss.item():.6f} != expected {expected_ce.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: SoftTargetLoss -- T=1 => kd_loss approximately equals standard KL
# ---------------------------------------------------------------------------


def test_soft_target_t1_equals_standard_kl():
    loss_fn = SoftTargetLoss(temperature=1.0, alpha=ALPHA)
    student_logits = torch.randn(BATCH, VOCAB)
    teacher_logits = torch.randn(BATCH, VOCAB)
    labels = torch.randint(0, VOCAB, (BATCH,))

    _, kd_loss, _ = loss_fn(student_logits, teacher_logits, labels)

    # Standard KL at T=1 (T^2 factor = 1.0, no scaling effect)
    student_lp = F.log_softmax(student_logits, dim=-1)
    teacher_p = F.softmax(teacher_logits, dim=-1)
    expected_kl = F.kl_div(student_lp, teacher_p, reduction="batchmean")

    assert torch.allclose(kd_loss, expected_kl, atol=1e-5), (
        f"T=1 kd_loss {kd_loss.item():.6f} != standard KL {expected_kl.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 4: SoftTargetLoss -- alpha=0 => total_loss == ce_loss
# ---------------------------------------------------------------------------


def test_soft_target_alpha_zero():
    loss_fn = SoftTargetLoss(temperature=TEMPERATURE, alpha=0.0)
    student_logits = torch.randn(BATCH, VOCAB, requires_grad=True)
    teacher_logits = torch.randn(BATCH, VOCAB)
    labels = torch.randint(0, VOCAB, (BATCH,))

    total_loss, _, ce_loss = loss_fn(student_logits, teacher_logits, labels)

    assert torch.allclose(total_loss, ce_loss, atol=1e-6), (
        f"alpha=0 total_loss {total_loss.item():.6f} != ce_loss {ce_loss.item():.6f}"
    )
    total_loss.backward()
    assert student_logits.grad is not None, "Expected grad on student_logits"


# ---------------------------------------------------------------------------
# Test 5: SoftTargetLoss -- alpha=1 => total_loss == kd_loss
# ---------------------------------------------------------------------------


def test_soft_target_alpha_one():
    loss_fn = SoftTargetLoss(temperature=TEMPERATURE, alpha=1.0)
    student_logits = torch.randn(BATCH, VOCAB, requires_grad=True)
    teacher_logits = torch.randn(BATCH, VOCAB)
    labels = torch.randint(0, VOCAB, (BATCH,))

    total_loss, kd_loss, _ = loss_fn(student_logits, teacher_logits, labels)

    assert torch.allclose(total_loss, kd_loss, atol=1e-6), (
        f"alpha=1 total_loss {total_loss.item():.6f} != kd_loss {kd_loss.item():.6f}"
    )
    total_loss.backward()
    assert student_logits.grad is not None, "Expected grad on student_logits"


# ---------------------------------------------------------------------------
# Test 6: FeatDistilLoss -- scalar, finite, grad flows to student but not teacher
# ---------------------------------------------------------------------------


def test_feat_distil_loss_finite_and_grad():
    feat_fn = FeatDistilLoss(student_dim=STUDENT_DIM, teacher_dim=TEACHER_DIM)

    student_hidden = torch.randn(BATCH, SEQ_LEN, STUDENT_DIM, requires_grad=True)
    teacher_hidden = torch.randn(BATCH, SEQ_LEN, TEACHER_DIM, requires_grad=True)

    loss = feat_fn(student_hidden, teacher_hidden)

    # Forward pass confirmed; check scalar and finite
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), "FeatDistilLoss is not finite"

    # Backward pass
    loss.backward()

    # Gradient flows to student hidden
    assert student_hidden.grad is not None, "No grad on student_hidden"
    assert torch.any(student_hidden.grad != 0), "student_hidden grad is all zeros"

    # Teacher hidden should have NO gradient (detached inside forward)
    assert teacher_hidden.grad is None, "teacher_hidden should have no grad"


# ---------------------------------------------------------------------------
# Test 7: FeatDistilLoss.layer_mapping -- correct length and valid indices
# ---------------------------------------------------------------------------


def test_feat_distil_layer_mapping():
    feat_fn = FeatDistilLoss(student_dim=STUDENT_DIM, teacher_dim=TEACHER_DIM)
    n_student = 3
    n_teacher = 12

    mapping = feat_fn.layer_mapping(n_student, n_teacher)

    assert len(mapping) == n_student, f"Expected mapping length {n_student}, got {len(mapping)}"
    for s_idx, t_idx in mapping:
        assert 0 <= s_idx < n_student, f"student index {s_idx} out of range"
        assert 0 <= t_idx < n_teacher, f"teacher index {t_idx} out of range"


# ---------------------------------------------------------------------------
# Test 8: AttentionTransferLoss -- scalar >= 0, 0 for identical inputs
# ---------------------------------------------------------------------------


def test_attn_transfer_zero_for_identical():
    attn_fn = AttentionTransferLoss()
    attn = torch.softmax(torch.randn(BATCH, N_HEADS_S, SEQ_LEN, SEQ_LEN), dim=-1)

    loss = attn_fn(attn, attn.clone())

    # Forward pass confirmed
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() >= 0.0, "AttentionTransferLoss must be non-negative"
    assert abs(loss.item()) < 1e-6, f"Expected ~0 for identical inputs, got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 9: AttentionTransferLoss -- mismatched H returns scalar via averaging
# ---------------------------------------------------------------------------


def test_attn_transfer_mismatched_heads():
    attn_fn = AttentionTransferLoss()
    # Student maps need requires_grad for backward
    student_attn = torch.softmax(
        torch.randn(BATCH, N_HEADS_S, SEQ_LEN, SEQ_LEN), dim=-1
    ).requires_grad_(True)
    teacher_attn = torch.softmax(torch.randn(BATCH, N_HEADS_T, SEQ_LEN, SEQ_LEN), dim=-1)

    loss = attn_fn(student_attn, teacher_attn)

    # Forward pass confirmed; just needs to return a scalar without error
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), "AttentionTransferLoss is not finite"
    loss.backward()
    assert student_attn.grad is not None, "Expected grad on student_attn"


# ---------------------------------------------------------------------------
# Test 10: PKDLoss -- scalar >= 0, 0 for identical representations
# ---------------------------------------------------------------------------


def test_pkd_loss_zero_for_identical():
    pkd_fn = PKDLoss(student_layers=[0, 1], teacher_layers=[0, 1])
    hidden = torch.randn(BATCH, SEQ_LEN, STUDENT_DIM)
    hiddens = [hidden.clone(), hidden.clone()]

    loss = pkd_fn(hiddens, hiddens)

    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() >= 0.0, "PKDLoss must be non-negative"
    assert abs(loss.item()) < 1e-6, f"Expected ~0 for identical inputs, got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 11: PKDLoss -- different hiddens produce positive loss
# ---------------------------------------------------------------------------


def test_pkd_loss_positive_for_different():
    pkd_fn = PKDLoss(student_layers=[0], teacher_layers=[0])

    # Orthogonal one-hot vectors -- large MSE after normalisation
    s_h_data = torch.zeros(BATCH, SEQ_LEN, STUDENT_DIM)
    s_h_data[..., 0] = 1.0  # one-hot in dim 0
    t_h_data = torch.zeros(BATCH, SEQ_LEN, STUDENT_DIM)
    t_h_data[..., 1] = 1.0  # one-hot in dim 1

    # requires_grad so backward works
    s_h = [s_h_data.clone().requires_grad_(True)]
    t_h = [t_h_data.clone()]

    loss = pkd_fn(s_h, t_h)

    assert loss.item() > 0.0, (
        f"Expected positive loss for different representations, got {loss.item()}"
    )
    loss.backward()
    assert s_h[0].grad is not None, "Expected grad on student hidden"


# ---------------------------------------------------------------------------
# Test 12: DistilTrainer -- teacher params frozen after init
# ---------------------------------------------------------------------------


def test_distil_trainer_teacher_frozen():
    student = make_student()
    teacher = make_teacher()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    feat_fn = FeatDistilLoss(student_dim=STUDENT_DIM, teacher_dim=TEACHER_DIM)
    soft_fn = SoftTargetLoss(temperature=TEMPERATURE, alpha=ALPHA)
    attn_fn = AttentionTransferLoss()

    trainer = DistilTrainer(student, teacher, optimizer, soft_fn, feat_fn, attn_fn)

    for name, param in trainer.teacher_model.named_parameters():
        assert not param.requires_grad, (
            f"Teacher param '{name}' should be frozen but requires_grad=True"
        )


# ---------------------------------------------------------------------------
# Test 13: DistilTrainer.train_step -- all keys present, total_loss finite
# ---------------------------------------------------------------------------


def test_distil_trainer_step_keys_finite():
    student = make_student()
    teacher = make_teacher()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    feat_fn = FeatDistilLoss(student_dim=STUDENT_DIM, teacher_dim=TEACHER_DIM)
    soft_fn = SoftTargetLoss(temperature=TEMPERATURE, alpha=ALPHA)
    attn_fn = AttentionTransferLoss()

    trainer = DistilTrainer(student, teacher, optimizer, soft_fn, feat_fn, attn_fn)

    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))

    result = trainer.train_step(input_ids, labels)

    expected_keys = {"total_loss", "kd_loss", "feat_loss", "attn_loss", "ce_loss"}
    assert set(result.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(result.keys())}"
    )
    assert math.isfinite(result["total_loss"]), "total_loss is not finite"


# ---------------------------------------------------------------------------
# Test 14: DistilTrainer -- grad flows to student, not teacher
# ---------------------------------------------------------------------------


def test_distil_trainer_grad_student_not_teacher():
    student = make_student()
    teacher = make_teacher()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    feat_fn = FeatDistilLoss(student_dim=STUDENT_DIM, teacher_dim=TEACHER_DIM)
    soft_fn = SoftTargetLoss(temperature=TEMPERATURE, alpha=ALPHA)
    attn_fn = AttentionTransferLoss()

    trainer = DistilTrainer(student, teacher, optimizer, soft_fn, feat_fn, attn_fn)

    # Record student param values before step
    before = {name: p.data.clone() for name, p in trainer.student_model.named_parameters()}

    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    trainer.train_step(input_ids, labels)

    # At least one student param must have changed
    student_changed = any(
        not torch.allclose(p.data, before[name])
        for name, p in trainer.student_model.named_parameters()
    )
    assert student_changed, "Student parameters did not update after train_step"

    # Teacher params must NOT have changed
    for name, param in trainer.teacher_model.named_parameters():
        assert param.grad is None, f"Teacher param '{name}' accumulated a gradient"


# ---------------------------------------------------------------------------
# Test 15: CompressionBenchmark.parameter_ratio < 1 when student smaller
# ---------------------------------------------------------------------------


def test_compression_parameter_ratio():
    bench = CompressionBenchmark()
    student = make_student()  # d_model=8
    teacher = make_teacher()  # d_model=16

    ratio = bench.parameter_ratio(student, teacher)
    assert ratio < 1.0, f"Expected ratio < 1.0 for smaller student, got {ratio}"


# ---------------------------------------------------------------------------
# Test 16 (bonus): perplexity_gap >= 1.0 when student worse than teacher
# ---------------------------------------------------------------------------


def test_compression_perplexity_gap_worse_student():
    bench = CompressionBenchmark()
    # Teacher assigns higher (less negative) log-probs than student
    teacher_logprobs = torch.tensor([-1.0, -1.5, -1.2])
    student_logprobs = torch.tensor([-3.0, -4.0, -3.5])

    gap = bench.perplexity_gap(student_logprobs, teacher_logprobs)
    assert gap >= 1.0, f"Expected perplexity_gap >= 1.0, got {gap}"


# ---------------------------------------------------------------------------
# Test 17 (bonus): layer_similarity in [-1,1], == 1.0 for identical layers
# ---------------------------------------------------------------------------


def test_compression_layer_similarity():
    bench = CompressionBenchmark()
    hidden = torch.randn(BATCH, SEQ_LEN, STUDENT_DIM)
    student_hiddens = [hidden]
    teacher_hiddens = [hidden.clone()]

    sims = bench.layer_similarity(student_hiddens, teacher_hiddens)

    assert len(sims) == 1, f"Expected 1 similarity value, got {len(sims)}"
    assert -1.0 <= sims[0] <= 1.0 + 1e-6, f"Similarity {sims[0]} out of [-1, 1]"
    assert abs(sims[0] - 1.0) < 1e-4, f"Expected ~1.0 for identical hiddens, got {sims[0]}"
