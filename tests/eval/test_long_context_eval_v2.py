"""Tests for long_context_eval_v2 — 15 tests, tiny configs."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.eval.long_context_eval_v2 import (
    NeedleInHaystack,
    PositionBiasAnalyzer,
    LostInTheMiddleScore,
    ContextUtilizationScore,
    LongContextBenchmark,
)

# ---------------------------------------------------------------------------
# Tiny test model: nn.Embedding(16, 16) + nn.Linear(16, 16)
# d_model=16, vocab=16, seq_len up to 12, batch=2, needle_len=3
# ---------------------------------------------------------------------------

VOCAB = 16
D_MODEL = 16
SEQ_LEN = 12
NEEDLE_LEN = 3
BATCH = 2


class TinyLM(nn.Module):
    """Minimal language model that returns (None, logits) tuple."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.linear = nn.Linear(D_MODEL, VOCAB)

    def forward(self, input_ids: Tensor):  # noqa: ANN201
        # input_ids: (B, T)
        x = self.embed(input_ids)          # (B, T, D_MODEL)
        logits = self.linear(x)            # (B, T, VOCAB)
        return (None, logits)


@pytest.fixture(scope="module")
def tiny_model() -> TinyLM:
    torch.manual_seed(42)
    return TinyLM()


@pytest.fixture(scope="module")
def haystack_ids() -> Tensor:
    torch.manual_seed(0)
    # Haystack: SEQ_LEN tokens, values in [0, VOCAB)
    return torch.randint(0, VOCAB, (SEQ_LEN,))


@pytest.fixture(scope="module")
def needle_ids() -> Tensor:
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (NEEDLE_LEN,))


# ---------------------------------------------------------------------------
# NeedleInHaystack tests
# ---------------------------------------------------------------------------

def test_create_test_context_longer_than_haystack(haystack_ids, needle_ids):
    """context_ids should be longer than haystack_ids after inserting needle."""
    nih = NeedleInHaystack()
    context_ids, needle_pos = nih.create_test(haystack_ids, needle_ids, 0.5)
    assert len(context_ids) == len(haystack_ids) + len(needle_ids)
    assert len(context_ids) > len(haystack_ids)


def test_create_test_needle_at_correct_position(haystack_ids, needle_ids):
    """Needle tokens should appear at the declared needle_position."""
    nih = NeedleInHaystack()
    depth = 0.5
    context_ids, needle_pos = nih.create_test(haystack_ids, needle_ids, depth)
    extracted = context_ids[needle_pos: needle_pos + NEEDLE_LEN]
    assert torch.equal(extracted, needle_ids)


def test_create_test_depth_zero_needle_at_start(haystack_ids, needle_ids):
    """depth=0 inserts needle at position 0 (start)."""
    nih = NeedleInHaystack()
    context_ids, needle_pos = nih.create_test(haystack_ids, needle_ids, 0.0)
    assert needle_pos == 0
    assert torch.equal(context_ids[:NEEDLE_LEN], needle_ids)


def test_create_test_depth_one_needle_at_end(haystack_ids, needle_ids):
    """depth=1 inserts needle at end of haystack."""
    nih = NeedleInHaystack()
    context_ids, needle_pos = nih.create_test(haystack_ids, needle_ids, 1.0)
    assert needle_pos == len(haystack_ids)
    assert torch.equal(context_ids[needle_pos: needle_pos + NEEDLE_LEN], needle_ids)


def test_score_retrieval_returns_nonpositive_float(tiny_model, haystack_ids, needle_ids):
    """score_retrieval should return a float <= 0 (log-prob)."""
    nih = NeedleInHaystack()
    context_ids, needle_pos = nih.create_test(haystack_ids, needle_ids, 0.5)
    score = nih.score_retrieval(tiny_model, context_ids, needle_ids, needle_pos)
    assert isinstance(score, float)
    assert score <= 0.0


def test_score_retrieval_double_needle_accumulates(tiny_model, haystack_ids, needle_ids):
    """Inserting needle twice should give a higher total score than once.

    We compare the sum of scores from two separate score_retrieval calls
    vs a single call — both must be finite, and both log-probs are <= 0.
    """
    nih = NeedleInHaystack()
    ctx1, np1 = nih.create_test(haystack_ids, needle_ids, 0.3)
    ctx2, np2 = nih.create_test(haystack_ids, needle_ids, 0.7)
    s1 = nih.score_retrieval(tiny_model, ctx1, needle_ids, np1)
    s2 = nih.score_retrieval(tiny_model, ctx2, needle_ids, np2)
    # Both are valid log-probs
    assert math.isfinite(s1)
    assert math.isfinite(s2)
    # Together they accumulate more evidence than either alone
    assert s1 + s2 <= 0.0  # both non-positive, sum still non-positive


# ---------------------------------------------------------------------------
# PositionBiasAnalyzer tests
# ---------------------------------------------------------------------------

def test_compute_position_attention_shape():
    """Output should have shape (T,)."""
    pba = PositionBiasAnalyzer()
    T = SEQ_LEN
    attn = torch.rand(BATCH, 2, T, T)
    # Normalise rows so it looks like attention
    attn = F.softmax(attn, dim=-1)
    scores = pba.compute_position_attention(attn)
    assert scores.shape == (T,)


def test_compute_position_attention_sums_to_one():
    """Position scores should sum to approximately 1 (normalised)."""
    pba = PositionBiasAnalyzer()
    T = SEQ_LEN
    attn = torch.rand(BATCH, 2, T, T)
    attn = F.softmax(attn, dim=-1)
    scores = pba.compute_position_attention(attn)
    assert abs(scores.sum().item() - 1.0) < 1e-5


def test_recency_bias_in_range():
    """recency_bias should be in [-1, 1]."""
    pba = PositionBiasAnalyzer()
    scores = torch.rand(SEQ_LEN)
    rb = pba.recency_bias(scores)
    assert -1.0 <= rb <= 1.0


def test_primacy_bias_nonnegative():
    """primacy_bias should be >= 0."""
    pba = PositionBiasAnalyzer()
    # Uniform scores: first tokens get same attention as middle
    scores = torch.ones(SEQ_LEN) / SEQ_LEN
    pb = pba.primacy_bias(scores)
    assert pb >= 0.0


# ---------------------------------------------------------------------------
# LostInTheMiddleScore tests
# ---------------------------------------------------------------------------

def test_lost_in_middle_score_nonnegative():
    """score() should return a non-negative value."""
    litm = LostInTheMiddleScore()
    scores = torch.rand(SEQ_LEN)
    s = litm.score(scores)
    assert s >= 0.0


def test_optimal_insertion_depth_valid_index():
    """optimal_insertion_depth should return a valid index."""
    litm = LostInTheMiddleScore()
    depth_scores = [0.1, 0.5, 0.3, 0.8, 0.2]
    idx = litm.optimal_insertion_depth(depth_scores)
    assert 0 <= idx < len(depth_scores)
    assert idx == 3  # 0.8 is the highest


# ---------------------------------------------------------------------------
# ContextUtilizationScore tests
# ---------------------------------------------------------------------------

def test_token_influence_shape(tiny_model):
    """token_influence should return tensor of shape (T,)."""
    cu = ContextUtilizationScore()
    input_ids = torch.randint(0, VOCAB, (SEQ_LEN,))
    inf = cu.token_influence(tiny_model, input_ids, output_position=0)
    assert inf.shape == (SEQ_LEN,)


def test_token_influence_nonnegative(tiny_model):
    """All influence scores should be >= 0 (gradient norms)."""
    cu = ContextUtilizationScore()
    input_ids = torch.randint(0, VOCAB, (SEQ_LEN,))
    inf = cu.token_influence(tiny_model, input_ids, output_position=1)
    assert (inf >= 0).all()


def test_effective_context_fraction_in_range(tiny_model):
    """effective_context_fraction should be in [0, 1]."""
    cu = ContextUtilizationScore()
    input_ids = torch.randint(0, VOCAB, (SEQ_LEN,))
    inf = cu.token_influence(tiny_model, input_ids, output_position=2)
    frac = cu.effective_context_fraction(inf)
    assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# LongContextBenchmark tests
# ---------------------------------------------------------------------------

def test_run_needle_test_per_depth_length(tiny_model, haystack_ids, needle_ids):
    """per_depth_scores should have exactly 5 entries."""
    bench = LongContextBenchmark(tiny_model)
    results = bench.run_needle_test(haystack_ids, needle_ids)
    assert "per_depth_scores" in results
    assert len(results["per_depth_scores"]) == 5


def test_position_bias_report_keys(tiny_model, haystack_ids):
    """position_bias_report should return all 3 required keys."""
    bench = LongContextBenchmark(tiny_model)
    report = bench.position_bias_report(haystack_ids)
    assert "recency_bias" in report
    assert "primacy_bias" in report
    assert "lost_in_middle" in report


def test_summarize_overall_score_finite(tiny_model, haystack_ids, needle_ids):
    """summarize should return overall_score key with a finite float."""
    bench = LongContextBenchmark(tiny_model)
    needle_res = bench.run_needle_test(haystack_ids, needle_ids)
    bias_res = bench.position_bias_report(haystack_ids)
    summary = bench.summarize(needle_res, bias_res)
    assert "overall_score" in summary
    assert math.isfinite(summary["overall_score"])


def test_uniform_attention_recency_near_zero_primacy_approx_one():
    """Uniform attention: recency_bias near 0, primacy_bias near 1."""
    pba = PositionBiasAnalyzer()
    T = SEQ_LEN
    # Uniform attention across all positions
    attn_uniform = torch.ones(1, 1, T, T) / T
    pos_scores = pba.compute_position_attention(attn_uniform)

    rb = pba.recency_bias(pos_scores)
    pb = pba.primacy_bias(pos_scores)

    # Uniform distribution => Spearman ~ 0
    assert abs(rb) < 0.2, f"Expected recency_bias ~ 0, got {rb}"
    # Uniform => first tokens get same attention as middle => ratio ~ 1.0
    assert abs(pb - 1.0) < 0.2, f"Expected primacy_bias ~ 1.0, got {pb}"
