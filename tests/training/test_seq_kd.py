"""Tests for seq_kd.py — Sequence-Level Knowledge Distillation.

Uses tiny configurations: vocab_size=32, d_model=16, max_length=5.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from aurelius.training.seq_kd import (
    GKDLoss,
    MiniLLMLoss,
    SequenceLevelKDLoss,
    TeacherSequenceSampler,
    sequence_log_probs,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB = 32
D_MODEL = 16
MAX_LEN = 5
PROMPT_LEN = 3
BEAM_WIDTH = 4


# ---------------------------------------------------------------------------
# Tiny model helper
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Tiny transformer-like model returning (hidden, logits)."""

    def __init__(self, vocab: int = VOCAB, d: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.proj = nn.Linear(d, vocab)

    def forward(self, input_ids: torch.Tensor):
        h = self.embed(input_ids).float()  # (1, T, d)
        logits = self.proj(h)  # (1, T, vocab)
        return h, logits


def _make_model_fn(seed: int = 0):
    torch.manual_seed(seed)
    model = _TinyModel()
    model.eval()
    return model


def _make_prompt():
    return torch.randint(0, VOCAB, (1, PROMPT_LEN))


def _make_sequences(n: int = 3, length: int = MAX_LEN) -> list:
    return [torch.randint(0, VOCAB, (length,)) for _ in range(n)]


# ---------------------------------------------------------------------------
# TeacherSequenceSampler tests
# ---------------------------------------------------------------------------


def test_sample_sequence_shape():
    """sample_sequence returns (max_length,) tensor."""
    teacher_fn = _make_model_fn()
    sampler = TeacherSequenceSampler(teacher_fn, VOCAB, temperature=1.0)
    prompt = _make_prompt()
    result = sampler.sample_sequence(prompt, MAX_LEN)
    assert result.shape == (MAX_LEN,)


def test_sample_sequence_vocab_range():
    """Token ids from sample_sequence are within [0, vocab_size)."""
    teacher_fn = _make_model_fn()
    sampler = TeacherSequenceSampler(teacher_fn, VOCAB, temperature=1.0)
    prompt = _make_prompt()
    result = sampler.sample_sequence(prompt, MAX_LEN)
    assert result.min().item() >= 0
    assert result.max().item() < VOCAB


def test_sample_sequence_greedy_deterministic():
    """temperature=0 produces identical outputs on repeated calls."""
    teacher_fn = _make_model_fn(seed=42)
    sampler = TeacherSequenceSampler(teacher_fn, VOCAB, temperature=0.0)
    prompt = _make_prompt()
    r1 = sampler.sample_sequence(prompt, MAX_LEN)
    r2 = sampler.sample_sequence(prompt, MAX_LEN)
    assert torch.equal(r1, r2)


def test_beam_search_returns_beam_width_sequences():
    """beam_search returns a list of beam_width sequences."""
    teacher_fn = _make_model_fn()
    sampler = TeacherSequenceSampler(teacher_fn, VOCAB, temperature=1.0)
    prompt = _make_prompt()
    beams = sampler.beam_search(prompt, MAX_LEN, beam_width=BEAM_WIDTH)
    assert len(beams) == BEAM_WIDTH


def test_beam_search_sequence_length():
    """Each beam sequence has the correct length (max_length,)."""
    teacher_fn = _make_model_fn()
    sampler = TeacherSequenceSampler(teacher_fn, VOCAB, temperature=1.0)
    prompt = _make_prompt()
    beams = sampler.beam_search(prompt, MAX_LEN, beam_width=BEAM_WIDTH)
    for seq in beams:
        assert seq.shape == (MAX_LEN,), f"expected ({MAX_LEN},) got {seq.shape}"


def test_sample_sequence_high_temperature_valid_tokens():
    """temperature=2.0 still returns valid token ids in vocab range."""
    teacher_fn = _make_model_fn(seed=7)
    sampler = TeacherSequenceSampler(teacher_fn, VOCAB, temperature=2.0)
    prompt = _make_prompt()
    result = sampler.sample_sequence(prompt, MAX_LEN)
    assert result.shape == (MAX_LEN,)
    assert result.min().item() >= 0
    assert result.max().item() < VOCAB


# ---------------------------------------------------------------------------
# SequenceLevelKDLoss tests
# ---------------------------------------------------------------------------


def test_seq_kd_loss_returns_scalar():
    """SequenceLevelKDLoss.forward returns a scalar tensor."""
    student_fn = _make_model_fn()
    loss_fn = SequenceLevelKDLoss(student_fn)
    prompt = _make_prompt()
    seqs = _make_sequences()
    loss = loss_fn.forward(prompt, seqs)
    assert loss.ndim == 0


def test_seq_kd_loss_finite():
    """SequenceLevelKDLoss value is finite."""
    student_fn = _make_model_fn()
    loss_fn = SequenceLevelKDLoss(student_fn)
    prompt = _make_prompt()
    seqs = _make_sequences()
    loss = loss_fn.forward(prompt, seqs)
    assert torch.isfinite(loss)


def test_seq_kd_loss_non_negative():
    """NLL is always >= 0 (log probs are <= 0, so -log_prob >= 0)."""
    student_fn = _make_model_fn()
    loss_fn = SequenceLevelKDLoss(student_fn)
    prompt = _make_prompt()
    seqs = _make_sequences()
    loss = loss_fn.forward(prompt, seqs)
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# MiniLLMLoss tests
# ---------------------------------------------------------------------------


def test_minillm_loss_returns_scalar():
    """MiniLLMLoss.forward returns a scalar tensor."""
    student_fn = _make_model_fn(seed=1)
    teacher_fn = _make_model_fn(seed=2)
    loss_fn = MiniLLMLoss(student_fn, teacher_fn)
    prompt = _make_prompt()
    student_seqs = _make_sequences()
    loss = loss_fn.forward(prompt, student_seqs)
    assert loss.ndim == 0


# ---------------------------------------------------------------------------
# GKDLoss tests
# ---------------------------------------------------------------------------


def test_gkd_loss_returns_scalar():
    """GKDLoss.forward returns a scalar tensor."""
    student_fn = _make_model_fn(seed=1)
    teacher_fn = _make_model_fn(seed=2)
    loss_fn = GKDLoss(student_fn, teacher_fn, beta=0.5)
    prompt = _make_prompt()
    teacher_seqs = _make_sequences()
    student_seqs = _make_sequences()
    loss = loss_fn.forward(prompt, teacher_seqs, student_seqs)
    assert loss.ndim == 0


def test_gkd_loss_beta_zero_equals_nll():
    """GKDLoss with beta=0 reduces to NLL on teacher sequences (forward KL only)."""
    student_fn = _make_model_fn(seed=5)
    teacher_fn = _make_model_fn(seed=6)

    gkd = GKDLoss(student_fn, teacher_fn, beta=0.0)
    seq_kd = SequenceLevelKDLoss(student_fn)

    prompt = _make_prompt()
    teacher_seqs = _make_sequences(n=2)
    student_seqs = _make_sequences(n=2)  # irrelevant when beta=0

    gkd_loss = gkd.forward(prompt, teacher_seqs, student_seqs)
    nll_loss = seq_kd.forward(prompt, teacher_seqs)

    assert torch.allclose(gkd_loss, nll_loss, atol=1e-5)


def test_gkd_loss_finite():
    """GKDLoss is finite."""
    student_fn = _make_model_fn(seed=1)
    teacher_fn = _make_model_fn(seed=2)
    loss_fn = GKDLoss(student_fn, teacher_fn, beta=0.5)
    prompt = _make_prompt()
    teacher_seqs = _make_sequences()
    student_seqs = _make_sequences()
    loss = loss_fn.forward(prompt, teacher_seqs, student_seqs)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# sequence_log_probs tests
# ---------------------------------------------------------------------------


def test_sequence_log_probs_returns_scalar():
    """sequence_log_probs returns a scalar tensor."""
    model_fn = _make_model_fn()
    prompt = _make_prompt()
    seq = torch.randint(0, VOCAB, (MAX_LEN,))
    result = sequence_log_probs(model_fn, prompt, seq)
    assert result.ndim == 0


def test_sequence_log_probs_non_positive():
    """Log probabilities are always <= 0."""
    model_fn = _make_model_fn()
    prompt = _make_prompt()
    seq = torch.randint(0, VOCAB, (MAX_LEN,))
    result = sequence_log_probs(model_fn, prompt, seq)
    # Mean of log-softmax values is <= 0 (each term <= 0)
    assert result.item() <= 0.0 + 1e-6
