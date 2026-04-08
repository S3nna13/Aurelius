"""Tests for offline knowledge distillation from pre-saved teacher logits."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.offline_kd import (
    OfflineKDConfig,
    TeacherLogitDataset,
    OfflineKDTrainer,
    offline_kd_loss,
)

# ── Test constants ────────────────────────────────────────────────────────────
VOCAB_SIZE = 64
SEQ_LEN = 8
N_TOKENS = 100  # total corpus tokens


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_npy_files(tmp_path: Path, n_tokens: int = N_TOKENS):
    """Write tiny input_ids.npy and teacher_logits.npy to tmp_path."""
    rng = np.random.default_rng(42)
    input_ids = rng.integers(0, VOCAB_SIZE, size=(n_tokens,), dtype=np.uint16)
    teacher_logits = rng.standard_normal((n_tokens, VOCAB_SIZE)).astype(np.float16)

    ids_path = tmp_path / "input_ids.npy"
    logits_path = tmp_path / "teacher_logits.npy"
    np.save(ids_path, input_ids)
    np.save(logits_path, teacher_logits)
    return str(ids_path), str(logits_path)


def _make_batch(batch_size: int = 2):
    """Return a synthetic batch dict (CPU tensors)."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    labels = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    teacher_logits = torch.randn(batch_size, SEQ_LEN, VOCAB_SIZE)
    return {"input_ids": input_ids, "labels": labels, "teacher_logits": teacher_logits}


def _make_model():
    """Tiny linear model: input_ids → logits (B, S, V)."""

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(7)
            self.embed = nn.Embedding(VOCAB_SIZE, 16)
            self.proj = nn.Linear(16, VOCAB_SIZE)

        def forward(self, input_ids):
            return self.proj(self.embed(input_ids))  # (B, S, V)

    return TinyLM()


# ── Config tests ──────────────────────────────────────────────────────────────

def test_offline_kd_config_defaults():
    cfg = OfflineKDConfig()
    assert cfg.temperature == 4.0
    assert cfg.alpha == 0.5
    assert cfg.top_k_logits == 0


# ── Dataset tests ─────────────────────────────────────────────────────────────

def test_teacher_logit_dataset_len(tmp_path):
    ids_path, logits_path = _make_npy_files(tmp_path)
    ds = TeacherLogitDataset(ids_path, logits_path, seq_len=SEQ_LEN)
    # windows = N_TOKENS - SEQ_LEN  (need 1 extra token for label)
    assert len(ds) == N_TOKENS - SEQ_LEN


def test_teacher_logit_dataset_getitem_shapes(tmp_path):
    ids_path, logits_path = _make_npy_files(tmp_path)
    ds = TeacherLogitDataset(ids_path, logits_path, seq_len=SEQ_LEN)
    item = ds[0]
    assert item["input_ids"].shape == (SEQ_LEN,)
    assert item["labels"].shape == (SEQ_LEN,)
    assert item["teacher_logits"].shape == (SEQ_LEN, VOCAB_SIZE)


def test_teacher_logit_dataset_labels_shifted(tmp_path):
    ids_path, logits_path = _make_npy_files(tmp_path)
    ds = TeacherLogitDataset(ids_path, logits_path, seq_len=SEQ_LEN)

    raw_ids = np.load(ids_path)
    for idx in range(min(5, len(ds))):
        item = ds[idx]
        expected_input = torch.from_numpy(raw_ids[idx : idx + SEQ_LEN].astype(np.int64))
        expected_labels = torch.from_numpy(raw_ids[idx + 1 : idx + SEQ_LEN + 1].astype(np.int64))
        assert torch.equal(item["input_ids"], expected_input), f"input_ids mismatch at idx={idx}"
        assert torch.equal(item["labels"], expected_labels), f"labels mismatch at idx={idx}"


# ── Loss function tests ───────────────────────────────────────────────────────

def test_offline_kd_loss_shape():
    cfg = OfflineKDConfig()
    batch = _make_batch()
    student_logits = torch.randn(2, SEQ_LEN, VOCAB_SIZE)
    loss = offline_kd_loss(student_logits, batch["teacher_logits"], batch["labels"], cfg)
    assert loss.ndim == 0, "loss must be a scalar tensor"


def test_offline_kd_loss_finite():
    cfg = OfflineKDConfig()
    batch = _make_batch()
    student_logits = torch.randn(2, SEQ_LEN, VOCAB_SIZE)
    loss = offline_kd_loss(student_logits, batch["teacher_logits"], batch["labels"], cfg)
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"


def test_offline_kd_loss_alpha_0_equals_ce():
    """alpha=0 → total loss should equal pure CE loss."""
    cfg = OfflineKDConfig(alpha=0.0)
    torch.manual_seed(1)
    B, S, V = 2, SEQ_LEN, VOCAB_SIZE
    student_logits = torch.randn(B, S, V)
    teacher_logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))

    total = offline_kd_loss(student_logits, teacher_logits, labels, cfg)

    import torch.nn.functional as F
    expected_ce = F.cross_entropy(student_logits.view(-1, V), labels.view(-1))
    assert abs(total.item() - expected_ce.item()) < 1e-4, (
        f"alpha=0 loss {total.item():.6f} != CE {expected_ce.item():.6f}"
    )


def test_offline_kd_loss_alpha_1_no_ce():
    """alpha=1 → total loss should equal pure KD loss (CE has zero weight)."""
    cfg = OfflineKDConfig(alpha=1.0)
    torch.manual_seed(2)
    B, S, V = 2, SEQ_LEN, VOCAB_SIZE
    student_logits = torch.randn(B, S, V)
    teacher_logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))

    total = offline_kd_loss(student_logits, teacher_logits, labels, cfg)

    import torch.nn.functional as F
    T = cfg.temperature
    kd = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1).view(-1, V),
        F.softmax(teacher_logits / T, dim=-1).view(-1, V),
        reduction="batchmean",
    ) * T ** 2

    assert abs(total.item() - kd.item()) < 1e-4, (
        f"alpha=1 loss {total.item():.6f} != KD loss {kd.item():.6f}"
    )


def test_offline_kd_loss_top_k_sparse():
    """top_k_logits > 0 should not crash and still produce finite loss."""
    cfg = OfflineKDConfig(top_k_logits=5)
    batch = _make_batch()
    student_logits = torch.randn(2, SEQ_LEN, VOCAB_SIZE)
    loss = offline_kd_loss(student_logits, batch["teacher_logits"], batch["labels"], cfg)
    assert torch.isfinite(loss), f"sparse KD loss is not finite: {loss.item()}"


# ── Trainer tests ─────────────────────────────────────────────────────────────

def test_offline_kd_trainer_step():
    """train_step must return finite loss, kd_loss, and ce_loss."""
    model = _make_model()
    cfg = OfflineKDConfig()
    trainer = OfflineKDTrainer(model, cfg, lr=1e-3)

    batch = _make_batch()
    metrics = trainer.train_step(batch)

    assert "loss" in metrics
    assert "kd_loss" in metrics
    assert "ce_loss" in metrics
    assert math.isfinite(metrics["loss"]), f"loss not finite: {metrics['loss']}"
    assert math.isfinite(metrics["kd_loss"]), f"kd_loss not finite: {metrics['kd_loss']}"
    assert math.isfinite(metrics["ce_loss"]), f"ce_loss not finite: {metrics['ce_loss']}"


def test_offline_kd_trainer_updates_weights():
    """train_step must update model parameters."""
    model = _make_model()
    cfg = OfflineKDConfig()
    trainer = OfflineKDTrainer(model, cfg, lr=1e-3)

    before = {n: p.clone().detach() for n, p in model.named_parameters()}
    batch = _make_batch()
    trainer.train_step(batch)

    changed = any(
        not torch.equal(before[n], p.detach())
        for n, p in model.named_parameters()
        if p.requires_grad
    )
    assert changed, "Model weights did not change after train_step"
