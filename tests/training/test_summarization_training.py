"""
Tests for src/training/summarization_training.py
=================================================
Uses tiny inline encoder/decoder (simple linear layers).
vocab_size=16, T_src=8, T_tgt=4, B=2.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import pytest

from src.training.summarization_training import (
    CoverageVector,
    ExtractivePseudoLabel,
    SummarizationLoss,
    ExtractiveAbstractiveTrainer,
    LeadBiasAugmentation,
    SummarizationConfig,
)

# ---------------------------------------------------------------------------
# Tiny model helpers
# ---------------------------------------------------------------------------

VOCAB = 16
T_SRC = 8
T_TGT = 4
B = 2
D_MODEL = 8


class TinyEncoder(nn.Module):
    """Embedding + linear → [B, T_src, D_MODEL].  Has extract_head."""

    def __init__(self, vocab: int = VOCAB, d: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.extract_head = nn.Linear(d, 1)

    def forward(self, src_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(src_ids)   # [B, T_src, D]


class TinyDecoder(nn.Module):
    """Embedding + linear projection → [B, T_tgt, vocab].
    Ignores enc_out for simplicity (tests only need correct shapes)."""

    def __init__(self, vocab: int = VOCAB, d: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.proj = nn.Linear(d, vocab)

    def forward(self, tgt_ids: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(tgt_ids))   # [B, T_tgt, vocab]


def make_src() -> torch.Tensor:
    return torch.randint(1, VOCAB, (B, T_SRC))


def make_tgt() -> torch.Tensor:
    return torch.randint(1, VOCAB, (B, T_TGT))


def make_attn() -> torch.Tensor:
    """Uniform attention weights over T_SRC source positions."""
    a = torch.ones(B, T_SRC) / T_SRC
    return a


def make_trainer() -> ExtractiveAbstractiveTrainer:
    enc = TinyEncoder()
    dec = TinyDecoder()
    return ExtractiveAbstractiveTrainer(enc, dec, lr=1e-3)


# ===========================================================================
# CoverageVector tests
# ===========================================================================

def test_coverage_vector_update_accumulates():
    """update() should accumulate attention weights into coverage."""
    cv = CoverageVector()
    cv.reset(B, T_SRC)
    a1 = make_attn()
    a2 = make_attn()
    cv.update(a1)
    cv.update(a2)
    # After two uniform updates, each position should have ~2/T_SRC
    expected = a1 + a2
    assert cv.coverage is not None
    assert torch.allclose(cv.coverage, expected, atol=1e-6)


def test_coverage_vector_coverage_loss_nonneg():
    """coverage_loss should be non-negative."""
    cv = CoverageVector()
    cv.reset(B, T_SRC)
    a = make_attn()
    cv.update(a)
    loss = cv.coverage_loss(a)
    assert loss.item() >= 0.0


def test_coverage_vector_coverage_loss_zero_before_update():
    """coverage_loss before any update (no accumulated coverage) = 0."""
    cv = CoverageVector()
    # Do NOT call reset or update so coverage is None
    a = make_attn()
    loss = cv.coverage_loss(a)
    assert loss.item() == 0.0


def test_coverage_vector_reset_clears():
    """reset() should produce all-zero coverage."""
    cv = CoverageVector()
    cv.reset(B, T_SRC)
    cv.update(make_attn())
    cv.reset(B, T_SRC)
    assert cv.coverage is not None
    assert cv.coverage.sum().item() == 0.0


def test_coverage_vector_reset_shape():
    """reset() should produce coverage with shape [B, T_src]."""
    cv = CoverageVector()
    cv.reset(B, T_SRC)
    assert cv.coverage is not None
    assert cv.coverage.shape == (B, T_SRC)


# ===========================================================================
# ExtractivePseudoLabel tests
# ===========================================================================

def test_extractive_pseudo_label_rouge_returns_list_of_floats():
    """compute_rouge_scores should return a list of floats."""
    ep = ExtractivePseudoLabel()
    src = [[1, 2, 3], [4, 5], [1, 6, 7]]
    tgt = [1, 2, 5]
    scores = ep.compute_rouge_scores(src, tgt)
    assert isinstance(scores, list)
    assert all(isinstance(s, float) for s in scores)


def test_extractive_pseudo_label_rouge_scores_in_0_1():
    """ROUGE-1 scores must lie in [0, 1]."""
    ep = ExtractivePseudoLabel()
    src = [[1, 2, 3], [4, 5], [1, 2, 3, 4, 5]]
    tgt = [1, 2, 4]
    scores = ep.compute_rouge_scores(src, tgt)
    for s in scores:
        assert 0.0 <= s <= 1.0, f"Score {s} out of [0, 1]"


def test_extractive_pseudo_label_rouge_length():
    """Length of returned scores equals number of source sentences."""
    ep = ExtractivePseudoLabel()
    src = [[1, 2], [3, 4], [5, 6], [7, 8]]
    tgt = [1, 3, 5]
    scores = ep.compute_rouge_scores(src, tgt)
    assert len(scores) == len(src)


def test_extractive_pseudo_label_oracle_returns_n_sent():
    """extract_oracle should return exactly n_sent indices."""
    ep = ExtractivePseudoLabel()
    src = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    tgt = [1, 4, 7]
    selected, scores = ep.extract_oracle(src, tgt, n_sent=3)
    assert len(selected) == 3
    assert len(scores) == 3


def test_extractive_pseudo_label_oracle_valid_indices():
    """extract_oracle indices must be valid sentence indices."""
    ep = ExtractivePseudoLabel()
    src = [[1, 2], [3, 4], [5, 6]]
    tgt = [1, 3]
    selected, _ = ep.extract_oracle(src, tgt, n_sent=2)
    for idx in selected:
        assert 0 <= idx < len(src)


def test_extractive_pseudo_label_oracle_no_duplicate_indices():
    """extract_oracle must not select the same sentence twice."""
    ep = ExtractivePseudoLabel()
    src = [[1, 2, 3], [4, 5], [6, 7, 8], [9, 10]]
    tgt = [1, 4, 6, 9]
    selected, _ = ep.extract_oracle(src, tgt, n_sent=3)
    assert len(selected) == len(set(selected))


# ===========================================================================
# SummarizationLoss tests
# ===========================================================================

def test_summarization_loss_seq2seq_finite_positive():
    """seq2seq_loss should be a finite, positive scalar."""
    sl = SummarizationLoss()
    logits = torch.randn(B, T_TGT, VOCAB)
    targets = make_tgt()
    loss = sl.seq2seq_loss(logits, targets)
    assert loss.ndim == 0
    assert math.isfinite(loss.item())
    assert loss.item() > 0.0


def test_summarization_loss_coverage_augmented_greater_than_seq2seq():
    """coverage_augmented_loss > seq2seq_loss when coverage_loss > 0."""
    sl = SummarizationLoss(coverage_lambda=1.0)
    logits = torch.randn(B, T_TGT, VOCAB)
    targets = make_tgt()
    coverage_loss = torch.tensor(0.5)
    aug_loss = sl.coverage_augmented_loss(logits, targets, coverage_loss)
    base_loss = sl.seq2seq_loss(logits, targets)
    assert aug_loss.item() > base_loss.item()


def test_summarization_loss_coverage_augmented_zero_coverage_equals_seq2seq():
    """With zero coverage_loss, coverage_augmented_loss == seq2seq_loss."""
    sl = SummarizationLoss(coverage_lambda=1.0)
    logits = torch.randn(B, T_TGT, VOCAB)
    targets = make_tgt()
    aug_loss = sl.coverage_augmented_loss(logits, targets, torch.tensor(0.0))
    base_loss = sl.seq2seq_loss(logits, targets)
    assert abs(aug_loss.item() - base_loss.item()) < 1e-6


def test_summarization_loss_length_penalty_finite():
    """length_penalty_loss should be a finite scalar."""
    sl = SummarizationLoss()
    logits = torch.randn(B, T_TGT, VOCAB)
    targets = make_tgt()
    loss = sl.length_penalty_loss(logits, targets, gen_len=T_TGT)
    assert math.isfinite(loss.item())


def test_summarization_loss_length_penalty_scalar():
    """length_penalty_loss should be a 0-dim tensor."""
    sl = SummarizationLoss()
    logits = torch.randn(B, T_TGT, VOCAB)
    targets = make_tgt()
    loss = sl.length_penalty_loss(logits, targets, gen_len=2)
    assert loss.ndim == 0


# ===========================================================================
# ExtractiveAbstractiveTrainer tests
# ===========================================================================

def test_extractive_abstractive_extractive_step_finite():
    """extractive_step should return a finite scalar loss."""
    trainer = make_trainer()
    src = make_src()
    labels = torch.randint(0, 2, (B, T_SRC)).float()
    loss = trainer.extractive_step(src, labels)
    assert math.isfinite(loss.item())


def test_extractive_abstractive_abstractive_step_finite():
    """abstractive_step should return a finite scalar loss."""
    trainer = make_trainer()
    src = make_src()
    tgt = make_tgt()
    loss = trainer.abstractive_step(src, tgt)
    assert math.isfinite(loss.item())


def test_extractive_abstractive_joint_step_all_finite():
    """joint_step should return three finite scalar losses."""
    trainer = make_trainer()
    src = make_src()
    tgt = make_tgt()
    labels = torch.randint(0, 2, (B, T_SRC)).float()
    total, ext, abs_ = trainer.joint_step(src, tgt, labels, alpha=0.5)
    assert math.isfinite(total.item())
    assert math.isfinite(ext.item())
    assert math.isfinite(abs_.item())


def test_extractive_abstractive_joint_step_alpha_1_total_approx_ext():
    """With alpha=1.0, total loss should equal extractive loss."""
    trainer = make_trainer()
    src = make_src()
    tgt = make_tgt()
    labels = torch.randint(0, 2, (B, T_SRC)).float()
    total, ext, abs_ = trainer.joint_step(src, tgt, labels, alpha=1.0)
    # total = 1.0*ext + 0.0*abs_ => total == ext
    assert abs(total.item() - ext.item()) < 1e-5


# ===========================================================================
# LeadBiasAugmentation tests
# ===========================================================================

def test_lead_bias_augment_output_shape_unchanged():
    """augment() output shape must equal input importance_scores shape."""
    lba = LeadBiasAugmentation(lead_frac=0.3)
    src = make_src()
    scores = torch.rand(B, T_SRC)
    out = lba.augment(src, scores)
    assert out.shape == scores.shape


def test_lead_bias_compute_position_bias_shape():
    """compute_position_bias should return tensor of shape [seq_len]."""
    lba = LeadBiasAugmentation()
    bias = lba.compute_position_bias(T_SRC)
    assert bias.shape == (T_SRC,)


def test_lead_bias_compute_position_bias_decaying():
    """Position bias weights should be strictly decreasing."""
    lba = LeadBiasAugmentation()
    bias = lba.compute_position_bias(10)
    diffs = bias[1:] - bias[:-1]
    assert (diffs < 0).all(), "Bias should be strictly decreasing"


def test_lead_bias_augment_lead_boosted_more():
    """First-position score should be multiplied by a factor > last-position factor."""
    lba = LeadBiasAugmentation(lead_frac=0.5)
    # Uniform importance
    scores = torch.ones(1, T_SRC)
    src = make_src()[:1]
    out = lba.augment(src, scores)
    # First token has weight 1/(1+0)=1, some later token has weight < 1
    assert out[0, 0].item() >= out[0, -1].item()


# ===========================================================================
# SummarizationConfig tests
# ===========================================================================

def test_summarization_config_defaults():
    """SummarizationConfig should have the expected default values."""
    cfg = SummarizationConfig()
    assert cfg.coverage_lambda == 1.0
    assert cfg.length_penalty == 0.6
    assert cfg.alpha == 0.5
    assert cfg.n_extract_sentences == 3
    assert cfg.lead_frac == 0.3
    assert cfg.lr == 1e-4
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
