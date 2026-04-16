"""Tests for Chain-of-Thought Distillation (Ho et al. 2022)."""
from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.cot_distillation import (
    CoTDistillConfig,
    CoTDistillTrainer,
    CoTSample,
    cot_cross_entropy_loss,
    knowledge_distillation_loss,
    prepare_cot_batch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 256
MAX_SEQ = 32


def _make_model() -> AureliusTransformer:
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=VOCAB,
        max_seq_len=MAX_SEQ,
    )
    return AureliusTransformer(cfg)


def _make_sample(seed: int = 0) -> CoTSample:
    torch.manual_seed(seed)
    return CoTSample(
        question_ids=torch.randint(1, VOCAB, (5,)),
        reasoning_ids=torch.randint(1, VOCAB, (8,)),
        answer_ids=torch.randint(1, VOCAB, (3,)),
    )


def _make_samples(n: int = 2) -> list[CoTSample]:
    return [_make_sample(seed=i) for i in range(n)]


# ---------------------------------------------------------------------------
# CoTSample tests
# ---------------------------------------------------------------------------

def test_cot_sample_constructs():
    """Test 1: CoTSample constructs with correct fields."""
    sample = _make_sample(0)
    assert sample.question_ids.shape == (5,)
    assert sample.reasoning_ids.shape == (8,)
    assert sample.answer_ids.shape == (3,)
    assert sample.teacher_logits is None


def test_cot_sample_with_teacher_logits():
    """CoTSample accepts optional teacher_logits."""
    torch.manual_seed(1)
    tl = torch.randn(11, VOCAB)  # T_r + T_a = 8 + 3 = 11
    sample = CoTSample(
        question_ids=torch.randint(1, VOCAB, (5,)),
        reasoning_ids=torch.randint(1, VOCAB, (8,)),
        answer_ids=torch.randint(1, VOCAB, (3,)),
        teacher_logits=tl,
    )
    assert sample.teacher_logits is not None
    assert sample.teacher_logits.shape == (11, VOCAB)


# ---------------------------------------------------------------------------
# cot_cross_entropy_loss tests
# ---------------------------------------------------------------------------

def test_cot_ce_loss_returns_tuple():
    """Test 2: cot_cross_entropy_loss returns (tensor, dict) tuple."""
    torch.manual_seed(10)
    B, T = 2, MAX_SEQ
    logits = torch.randn(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T))
    mask = torch.randint(0, 2, (B, T))

    result = cot_cross_entropy_loss(logits, targets, mask)
    assert isinstance(result, tuple)
    assert len(result) == 2
    loss, metrics = result
    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics, dict)


def test_cot_ce_loss_dict_keys():
    """Test 3: dict has cot_loss, answer_loss, total_loss keys."""
    torch.manual_seed(11)
    B, T = 2, MAX_SEQ
    logits = torch.randn(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T))
    mask = torch.randint(0, 2, (B, T))

    _, metrics = cot_cross_entropy_loss(logits, targets, mask)
    assert "cot_loss" in metrics
    assert "answer_loss" in metrics
    assert "total_loss" in metrics


def test_cot_ce_loss_is_finite():
    """Test 4: total_loss is finite."""
    torch.manual_seed(12)
    B, T = 2, MAX_SEQ
    logits = torch.randn(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)

    loss, metrics = cot_cross_entropy_loss(logits, targets, mask)
    assert torch.isfinite(loss)
    assert metrics["total_loss"] == pytest.approx(loss.item(), rel=1e-5)


def test_cot_ce_loss_alpha_one():
    """Test 5: alpha=1.0 → total_loss ≈ cot_loss (no answer component)."""
    torch.manual_seed(13)
    B, T = 2, MAX_SEQ
    logits = torch.randn(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T))
    # Mix of reasoning and answer positions
    mask = torch.zeros(B, T, dtype=torch.long)
    mask[:, :16] = 1  # first half is reasoning

    loss, metrics = cot_cross_entropy_loss(logits, targets, mask, alpha=1.0)
    assert metrics["total_loss"] == pytest.approx(metrics["cot_loss"], rel=1e-4)


def test_cot_ce_loss_alpha_zero():
    """Test 6: alpha=0.0 → total_loss ≈ answer_loss."""
    torch.manual_seed(14)
    B, T = 2, MAX_SEQ
    logits = torch.randn(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T))
    mask = torch.zeros(B, T, dtype=torch.long)
    mask[:, :16] = 1  # first half is reasoning

    loss, metrics = cot_cross_entropy_loss(logits, targets, mask, alpha=0.0)
    assert metrics["total_loss"] == pytest.approx(metrics["answer_loss"], rel=1e-4)


# ---------------------------------------------------------------------------
# knowledge_distillation_loss tests
# ---------------------------------------------------------------------------

def test_kd_loss_returns_scalar():
    """Test 7: knowledge_distillation_loss returns scalar tensor."""
    torch.manual_seed(20)
    B, T = 2, MAX_SEQ
    student_logits = torch.randn(B, T, VOCAB)
    teacher_logits = torch.randn(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T))

    loss = knowledge_distillation_loss(student_logits, teacher_logits, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar


def test_kd_loss_finite_and_nonneg():
    """Test 8: KD loss is finite and non-negative."""
    torch.manual_seed(21)
    B, T = 2, MAX_SEQ
    student_logits = torch.randn(B, T, VOCAB)
    teacher_logits = torch.randn(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T))

    loss = knowledge_distillation_loss(student_logits, teacher_logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# prepare_cot_batch tests
# ---------------------------------------------------------------------------

def test_prepare_cot_batch_returns_required_keys():
    """Test 9: prepare_cot_batch returns dict with all required keys."""
    samples = _make_samples(2)
    batch = prepare_cot_batch(samples, max_len=MAX_SEQ)

    assert "input_ids" in batch
    assert "target_ids" in batch
    assert "reasoning_mask" in batch
    assert "attention_mask" in batch


def test_prepare_cot_batch_input_ids_shape():
    """Test 10: input_ids shape is (B, T)."""
    torch.manual_seed(30)
    samples = _make_samples(3)
    batch = prepare_cot_batch(samples, max_len=MAX_SEQ)

    assert batch["input_ids"].shape == (3, MAX_SEQ)
    assert batch["target_ids"].shape == (3, MAX_SEQ)
    assert batch["reasoning_mask"].shape == (3, MAX_SEQ)
    assert batch["attention_mask"].shape == (3, MAX_SEQ)


def test_prepare_cot_batch_reasoning_mask_binary():
    """Test 11: reasoning_mask is binary (0/1)."""
    samples = _make_samples(2)
    batch = prepare_cot_batch(samples, max_len=MAX_SEQ)

    mask = batch["reasoning_mask"]
    unique_vals = mask.unique().tolist()
    for v in unique_vals:
        assert v in (0, 1), f"reasoning_mask contains unexpected value {v}"


def test_prepare_cot_batch_single_sample():
    """prepare_cot_batch works with a single sample."""
    sample = _make_sample(5)
    batch = prepare_cot_batch([sample], max_len=MAX_SEQ)
    assert batch["input_ids"].shape == (1, MAX_SEQ)


# ---------------------------------------------------------------------------
# CoTDistillTrainer tests
# ---------------------------------------------------------------------------

def test_trainer_train_step_returns_loss_key():
    """Test 12: CoTDistillTrainer train_step returns metrics with 'loss' key."""
    torch.manual_seed(42)
    student = _make_model()
    config = CoTDistillConfig(
        alpha=0.5,
        lr=1e-4,
        max_seq_len=MAX_SEQ,
        cot_loss_type="ce",
    )
    trainer = CoTDistillTrainer(student=student, teacher=None, config=config)

    samples = _make_samples(2)
    metrics = trainer.train_step(samples)

    assert "loss" in metrics
    assert isinstance(metrics["loss"], float)
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_trainer_train_step_has_cot_and_answer_loss():
    """train_step metrics also include 'cot_loss' and 'answer_loss'."""
    torch.manual_seed(43)
    student = _make_model()
    config = CoTDistillConfig(max_seq_len=MAX_SEQ, cot_loss_type="ce")
    trainer = CoTDistillTrainer(student=student, teacher=None, config=config)

    samples = _make_samples(2)
    metrics = trainer.train_step(samples)

    assert "cot_loss" in metrics
    assert "answer_loss" in metrics


def test_trainer_kl_mode_with_teacher():
    """CoTDistillTrainer with cot_loss_type='kl' and a teacher model runs correctly."""
    torch.manual_seed(44)
    student = _make_model()
    teacher = _make_model()
    config = CoTDistillConfig(
        max_seq_len=MAX_SEQ,
        cot_loss_type="kl",
        temperature=2.0,
        alpha=0.5,
    )
    trainer = CoTDistillTrainer(student=student, teacher=teacher, config=config)

    samples = _make_samples(2)
    metrics = trainer.train_step(samples)

    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_trainer_updates_student_params():
    """train_step actually updates student parameters."""
    torch.manual_seed(45)
    student = _make_model()
    config = CoTDistillConfig(max_seq_len=MAX_SEQ, lr=1e-2)
    trainer = CoTDistillTrainer(student=student, teacher=None, config=config)

    # Capture param snapshot before
    before = {n: p.clone() for n, p in student.named_parameters()}

    samples = _make_samples(2)
    trainer.train_step(samples)

    changed = any(
        not torch.equal(before[n], p)
        for n, p in student.named_parameters()
    )
    assert changed, "Student parameters were not updated after train_step"


def test_generate_rationale_shape():
    """generate_rationale returns tensor of shape (max_new_tokens,)."""
    torch.manual_seed(50)
    student = _make_model()
    config = CoTDistillConfig(max_seq_len=MAX_SEQ)
    trainer = CoTDistillTrainer(student=student, teacher=None, config=config)

    question_ids = torch.randint(1, VOCAB, (5,))
    rationale = trainer.generate_rationale(student, question_ids, max_new_tokens=8)

    assert rationale.shape == (8,)
    assert rationale.dtype in (torch.int64, torch.long)
