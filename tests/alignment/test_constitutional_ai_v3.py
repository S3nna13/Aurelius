"""Tests for Constitutional AI v3 (src/alignment/constitutional_ai_v3.py).

15 tests covering:
  - ConstitutionalPrinciple.to_dict
  - CritiqueHead forward shape and value range
  - CritiqueHead.aggregate shape and correctness
  - RevisionScorer.encode shape
  - RevisionScorer.score_revision range and identical-input edge case
  - RevisionScorer.improvement_loss scalar / finite / grad
  - CAITrainer.critique_step keys / loss finite / grad to critique_head
  - CAITrainer.revision_step keys / loss finite
  - ConstitutionalFilter.score range and shape
  - ConstitutionalFilter.should_revise dtype and shape
  - ConstitutionalFilter.revision_priority shape and ordering
  - CritiqueHead aggregate with zero-weight principles
  - critique_step with all-harmless labels converges toward low loss
  - principle_weights summing to 1.0 give aggregate in (0, 1)
  - Full critique + revision cycle succeeds end-to-end
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aurelius.alignment.constitutional_ai_v3 import (
    CAITrainer,
    ConstitutionalFilter,
    ConstitutionalPrinciple,
    CritiqueHead,
    RevisionScorer,
)

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------

D_MODEL = 16
N_PRINCIPLES = 3
SEQ_LEN = 8
BATCH = 2
VOCAB = 16


# ---------------------------------------------------------------------------
# Tiny model fixture: Embedding + Linear that returns (logits, hidden_states)
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal model: Embedding -> Linear, returns (logits, hidden_states)."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, input_ids: torch.Tensor):
        # hidden_states: (B, T, D)
        hidden = self.proj(self.embed(input_ids))
        # logits: (B, T, VOCAB) — dummy, reuse proj output tiled
        logits = hidden.unsqueeze(-1).expand(-1, -1, -1, VOCAB)[..., 0, :]
        return logits, hidden


def _make_model() -> _TinyModel:
    return _TinyModel()


def _make_principles() -> list:
    return [
        ConstitutionalPrinciple(
            name=f"p{i}",
            critique_prompt=f"Is this response safe according to criterion {i}?",
            revision_prompt=f"Revise to meet criterion {i}.",
            weight=1.0,
        )
        for i in range(N_PRINCIPLES)
    ]


def _make_input_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN))


# ---------------------------------------------------------------------------
# Test 1: ConstitutionalPrinciple.to_dict — all fields present
# ---------------------------------------------------------------------------


def test_principle_to_dict_all_fields():
    p = ConstitutionalPrinciple(
        name="honesty",
        critique_prompt="Is it honest?",
        revision_prompt="Make it honest.",
        weight=2.5,
    )
    d = p.to_dict()
    assert "name" in d
    assert "critique_prompt" in d
    assert "revision_prompt" in d
    assert "weight" in d
    assert d["name"] == "honesty"
    assert d["weight"] == 2.5


# ---------------------------------------------------------------------------
# Test 2: CritiqueHead output shape (B, n_principles)
# ---------------------------------------------------------------------------


def test_critique_head_output_shape():
    head = CritiqueHead(D_MODEL, N_PRINCIPLES)
    hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    scores = head(hidden)
    assert scores.shape == (BATCH, N_PRINCIPLES)


# ---------------------------------------------------------------------------
# Test 3: CritiqueHead output values strictly in (0, 1)
# ---------------------------------------------------------------------------


def test_critique_head_output_range():
    head = CritiqueHead(D_MODEL, N_PRINCIPLES)
    hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    scores = head(hidden)
    assert (scores > 0).all(), "Scores must be > 0 (sigmoid output)"
    assert (scores < 1).all(), "Scores must be < 1 (sigmoid output)"


# ---------------------------------------------------------------------------
# Test 4: CritiqueHead.aggregate output shape (B,) and weighted-mean correctness
# ---------------------------------------------------------------------------


def test_critique_head_aggregate_shape_and_correctness():
    head = CritiqueHead(D_MODEL, N_PRINCIPLES)
    # Construct known scores: all 0.5
    scores = torch.full((BATCH, N_PRINCIPLES), 0.5)
    weights = torch.ones(N_PRINCIPLES)
    agg = head.aggregate(scores, weights)
    assert agg.shape == (BATCH,)
    # All scores 0.5, equal weights => aggregate should be 0.5
    assert torch.allclose(agg, torch.full((BATCH,), 0.5), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 5: RevisionScorer.encode output shape (B, D)
# ---------------------------------------------------------------------------


def test_revision_scorer_encode_shape():
    scorer = RevisionScorer(D_MODEL)
    hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    enc = scorer.encode(hidden)
    assert enc.shape == (BATCH, D_MODEL)


# ---------------------------------------------------------------------------
# Test 6: RevisionScorer.score_revision range [-1, 1] and identical inputs => ~1
# ---------------------------------------------------------------------------


def test_revision_scorer_score_revision_range_and_identical():
    scorer = RevisionScorer(D_MODEL)
    hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    # Identical inputs should yield cosine similarity close to 1
    sim_identical = scorer.score_revision(hidden, hidden)
    assert sim_identical.shape == (BATCH,)
    assert (sim_identical >= -1.0 - 1e-5).all()
    assert (sim_identical <= 1.0 + 1e-5).all()
    assert torch.allclose(sim_identical, torch.ones(BATCH), atol=1e-4)


# ---------------------------------------------------------------------------
# Test 7: RevisionScorer.improvement_loss is scalar, finite, and grad flows
# ---------------------------------------------------------------------------


def test_revision_scorer_improvement_loss_grad():
    scorer = RevisionScorer(D_MODEL)
    orig = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
    rev = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
    sim = scorer.score_revision(orig, rev)
    loss = scorer.improvement_loss(sim, target=0.8)
    assert loss.ndim == 0, "improvement_loss must return a scalar"
    assert torch.isfinite(loss), "loss must be finite"
    loss.backward()
    assert orig.grad is not None, "grad must flow to original hidden states"


# ---------------------------------------------------------------------------
# Test 8: CAITrainer.critique_step — all keys present, loss finite, grad flows
# ---------------------------------------------------------------------------


def test_cai_trainer_critique_step_keys_and_grad():
    model = _make_model()
    head = CritiqueHead(D_MODEL, N_PRINCIPLES)
    optimizer = torch.optim.Adam(list(head.parameters()) + list(model.parameters()), lr=1e-3)
    principles = _make_principles()
    trainer = CAITrainer(model, head, optimizer, principles)

    input_ids = _make_input_ids()
    labels = torch.rand(BATCH)  # float in [0, 1]

    # Record param state before step
    param_before = list(head.parameters())[0].data.clone()
    result = trainer.critique_step(input_ids, labels)

    assert "critique_loss" in result
    assert "mean_score" in result
    assert "per_principle_scores" in result
    assert torch.isfinite(torch.tensor(result["critique_loss"]))
    # After backward + step, params should have changed
    param_after = list(head.parameters())[0].data
    assert not torch.allclose(param_before, param_after), "Params must update after critique_step"


# ---------------------------------------------------------------------------
# Test 9: CAITrainer.revision_step — all keys present, loss finite
# ---------------------------------------------------------------------------


def test_cai_trainer_revision_step_keys_and_finite():
    model = _make_model()
    head = CritiqueHead(D_MODEL, N_PRINCIPLES)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    principles = _make_principles()
    trainer = CAITrainer(model, head, optimizer, principles)

    original_ids = _make_input_ids()
    revised_ids = _make_input_ids()
    result = trainer.revision_step(original_ids, revised_ids)

    assert "revision_loss" in result
    assert "similarity_score" in result
    assert torch.isfinite(torch.tensor(result["revision_loss"]))


# ---------------------------------------------------------------------------
# Test 10: ConstitutionalFilter.score — output in (0, 1), shape (B,)
# ---------------------------------------------------------------------------


def test_constitutional_filter_score_shape_and_range():
    head = CritiqueHead(D_MODEL, N_PRINCIPLES)
    filt = ConstitutionalFilter(head, threshold=0.5)
    hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    weights = torch.ones(N_PRINCIPLES)
    scores = filt.score(hidden, weights)
    assert scores.shape == (BATCH,)
    assert (scores > 0).all()
    assert (scores < 1).all()


# ---------------------------------------------------------------------------
# Test 11: ConstitutionalFilter.should_revise — bool tensor, shape (B,)
# ---------------------------------------------------------------------------


def test_constitutional_filter_should_revise_dtype_and_shape():
    head = CritiqueHead(D_MODEL, N_PRINCIPLES)
    filt = ConstitutionalFilter(head, threshold=0.5)
    scores = torch.tensor([0.3, 0.7])
    mask = filt.should_revise(scores)
    assert mask.dtype == torch.bool
    assert mask.shape == (BATCH,)
    assert mask[0].item() is True  # 0.3 < 0.5
    assert mask[1].item() is False  # 0.7 >= 0.5


# ---------------------------------------------------------------------------
# Test 12: ConstitutionalFilter.revision_priority — shape (B,), argsorted ascending
# ---------------------------------------------------------------------------


def test_constitutional_filter_revision_priority_order():
    head = CritiqueHead(D_MODEL, N_PRINCIPLES)
    filt = ConstitutionalFilter(head, threshold=0.5)
    # batch=4 for richer ordering test
    scores = torch.tensor([0.9, 0.1, 0.5, 0.3])
    priority = filt.revision_priority(scores)
    assert priority.shape == (4,)
    # Expect order: 0.1 (idx 1), 0.3 (idx 3), 0.5 (idx 2), 0.9 (idx 0)
    expected = torch.tensor([1, 3, 2, 0])
    assert torch.equal(priority, expected), f"Got {priority}, expected {expected}"


# ---------------------------------------------------------------------------
# Test 13: CritiqueHead aggregate with weight=0 — zero-weight principles ignored
# ---------------------------------------------------------------------------


def test_critique_head_aggregate_zero_weight():
    head = CritiqueHead(D_MODEL, N_PRINCIPLES)
    # Two principles score 1.0, one principle scores 0.0 but has weight 0
    scores = torch.tensor([[1.0, 1.0, 0.0]])  # (1, 3)
    weights = torch.tensor([1.0, 1.0, 0.0])
    agg = head.aggregate(scores, weights)
    # Should average only the first two => 1.0
    assert torch.allclose(agg, torch.tensor([1.0]), atol=1e-5), (
        f"Expected 1.0 when zero-weight principle excluded, got {agg}"
    )


# ---------------------------------------------------------------------------
# Test 14: critique_step with all-harmless labels — loss decreases after 10 steps
# ---------------------------------------------------------------------------


def test_critique_step_all_harmless_loss_decreases():
    model = _make_model()
    head = CritiqueHead(D_MODEL, N_PRINCIPLES)
    optimizer = torch.optim.Adam(list(head.parameters()) + list(model.parameters()), lr=1e-2)
    principles = _make_principles()
    trainer = CAITrainer(model, head, optimizer, principles)

    input_ids = _make_input_ids()
    labels = torch.ones(BATCH)  # all harmless

    initial_result = trainer.critique_step(input_ids, labels)
    initial_loss = initial_result["critique_loss"]

    # Run several more steps
    for _ in range(15):
        result = trainer.critique_step(input_ids, labels)

    final_loss = result["critique_loss"]
    assert final_loss < initial_loss, (
        f"Loss should decrease with all-harmless labels: {initial_loss} -> {final_loss}"
    )


# ---------------------------------------------------------------------------
# Test 15: Full critique + revision cycle — both steps succeed without error
# ---------------------------------------------------------------------------


def test_full_critique_revision_cycle():
    model = _make_model()
    head = CritiqueHead(D_MODEL, N_PRINCIPLES)
    optimizer = torch.optim.Adam(list(head.parameters()) + list(model.parameters()), lr=1e-3)
    principles = _make_principles()
    trainer = CAITrainer(model, head, optimizer, principles)

    input_ids = _make_input_ids()
    revised_ids = _make_input_ids()
    labels = torch.rand(BATCH)

    # Critique step
    critique_result = trainer.critique_step(input_ids, labels)
    assert torch.isfinite(torch.tensor(critique_result["critique_loss"]))

    # Revision step using same IDs as original vs revised
    revision_result = trainer.revision_step(input_ids, revised_ids)
    assert torch.isfinite(torch.tensor(revision_result["revision_loss"]))

    # Filter inference pass
    filt = ConstitutionalFilter(head, threshold=0.5)
    hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    weights = torch.ones(N_PRINCIPLES) / N_PRINCIPLES  # sum to 1
    scores = filt.score(hidden, weights)
    assert scores.shape == (BATCH,)
    assert (scores > 0).all() and (scores < 1).all()
