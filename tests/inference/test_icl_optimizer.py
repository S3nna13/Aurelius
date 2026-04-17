"""Tests for src/inference/icl_optimizer.py

Tiny config: vocab=16, d_model=8, seq_len<=8, k=2, batch<=2
All tests run forward (and some backward) passes.
"""

from __future__ import annotations

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from src.inference.icl_optimizer import (
    DemonstrationExample,
    ExemplarSelector,
    PromptOrdering,
    ICLPromptBuilder,
    ICLEvaluator,
)

# ---------------------------------------------------------------------------
# Tiny model fixture  (nn.Embedding + nn.Linear, causal-ish)
# ---------------------------------------------------------------------------

VOCAB = 16
D_MODEL = 8


class TinyLM(nn.Module):
    """Minimal LM: embed -> linear -> logits.  Accepts (B, T) -> (B, T, V)."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.proj = nn.Linear(D_MODEL, VOCAB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(x))


@pytest.fixture()
def model():
    m = TinyLM()
    m.train(False)
    return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_demo(inp_len: int = 3, lbl_len: int = 2, d: int = D_MODEL) -> DemonstrationExample:
    inp = torch.randint(0, VOCAB, (inp_len,))
    lbl = torch.randint(0, VOCAB, (lbl_len,))
    emb = F.normalize(torch.randn(d), dim=0)
    return DemonstrationExample(input_ids=inp, label_ids=lbl, embedding=emb)


def make_pool(n: int = 6) -> list:
    return [make_demo() for _ in range(n)]


# ---------------------------------------------------------------------------
# 1. DemonstrationExample.to_sequence: length = input + 1 + label
# ---------------------------------------------------------------------------

def test_to_sequence_length():
    demo = make_demo(inp_len=3, lbl_len=2)
    sep_id = 1
    seq = demo.to_sequence(separator_id=sep_id)
    assert seq.shape[0] == 3 + 1 + 2, f"Expected 6, got {seq.shape[0]}"


# ---------------------------------------------------------------------------
# 2. DemonstrationExample.to_sequence: contains separator
# ---------------------------------------------------------------------------

def test_to_sequence_contains_separator():
    demo = make_demo(inp_len=3, lbl_len=2)
    sep_id = 7
    seq = demo.to_sequence(separator_id=sep_id)
    # Separator should appear at index 3 (right after input_ids)
    assert int(seq[3].item()) == sep_id


# ---------------------------------------------------------------------------
# 3. DemonstrationExample.length: correct count
# ---------------------------------------------------------------------------

def test_demonstration_length():
    demo = make_demo(inp_len=4, lbl_len=3)
    assert demo.length() == 4 + 1 + 3


# ---------------------------------------------------------------------------
# 4. ExemplarSelector.random_select: returns k examples from pool
# ---------------------------------------------------------------------------

def test_random_select_returns_k():
    pool = make_pool(6)
    selector = ExemplarSelector(method="random", k=2)
    selected = selector.random_select(pool)
    assert len(selected) == 2
    pool_ids = [id(p) for p in pool]
    for ex in selected:
        assert id(ex) in pool_ids


# ---------------------------------------------------------------------------
# 5. ExemplarSelector.similarity_select: most similar example first, shape correct
# ---------------------------------------------------------------------------

def test_similarity_select_ordering():
    pool = make_pool(6)
    query = F.normalize(torch.randn(D_MODEL), dim=0)
    selector = ExemplarSelector(method="similarity", k=2)
    selected = selector.similarity_select(pool, query)
    assert len(selected) == 2
    # First result should have higher or equal sim than second
    sim0 = float(F.cosine_similarity(query.unsqueeze(0), selected[0].embedding.unsqueeze(0)))
    sim1 = float(F.cosine_similarity(query.unsqueeze(0), selected[1].embedding.unsqueeze(0)))
    assert sim0 >= sim1 - 1e-6


# ---------------------------------------------------------------------------
# 6. ExemplarSelector.diverse_select: returns k distinct examples
# ---------------------------------------------------------------------------

def test_diverse_select_distinct():
    pool = make_pool(6)
    query = F.normalize(torch.randn(D_MODEL), dim=0)
    selector = ExemplarSelector(method="diverse", k=2)
    selected = selector.diverse_select(pool, query)
    assert len(selected) == 2
    assert selected[0] is not selected[1]


# ---------------------------------------------------------------------------
# 7. PromptOrdering.similarity_order: sorted by cos sim, shape preserving
# ---------------------------------------------------------------------------

def test_similarity_order_shape():
    examples = make_pool(4)
    query = F.normalize(torch.randn(D_MODEL), dim=0)
    ordering = PromptOrdering()
    ordered = ordering.similarity_order(examples, query)
    assert len(ordered) == len(examples)
    # Most similar last
    sims = [
        float(F.cosine_similarity(query.unsqueeze(0), ex.embedding.unsqueeze(0)))
        for ex in ordered
    ]
    assert sims == sorted(sims), f"Expected ascending order: {sims}"


# ---------------------------------------------------------------------------
# 8. PromptOrdering.curriculum_order: ascending difficulty
# ---------------------------------------------------------------------------

def test_curriculum_order_ascending():
    examples = make_pool(4)
    scores = [0.9, 0.1, 0.5, 0.3]
    ordering = PromptOrdering()
    ordered = ordering.curriculum_order(examples, scores)
    assert len(ordered) == 4
    # ordered should correspond to sorted scores ascending: 0.1, 0.3, 0.5, 0.9
    sorted_idx = sorted(range(4), key=lambda i: scores[i])
    for j, idx in enumerate(sorted_idx):
        assert ordered[j] is examples[idx]


# ---------------------------------------------------------------------------
# 9. ICLPromptBuilder.build: output is 1D tensor, contains query_ids at end
# ---------------------------------------------------------------------------

def test_prompt_builder_1d_query_at_end():
    demos = [make_demo(2, 2) for _ in range(2)]
    query = torch.randint(0, VOCAB, (3,))
    builder = ICLPromptBuilder(max_length=64, separator_id=1)
    prompt = builder.build(demos, query)
    assert prompt.dim() == 1
    # query_ids appear at the end
    assert torch.equal(prompt[-3:], query)


# ---------------------------------------------------------------------------
# 10. ICLPromptBuilder.build: length <= max_length (truncation works)
# ---------------------------------------------------------------------------

def test_prompt_builder_truncation():
    # Make many long demos to force truncation
    demos = [make_demo(5, 5) for _ in range(10)]
    query = torch.randint(0, VOCAB, (4,))
    builder = ICLPromptBuilder(max_length=16, separator_id=1)
    prompt = builder.build(demos, query)
    assert prompt.shape[0] <= 16


# ---------------------------------------------------------------------------
# 11. ICLPromptBuilder.n_shots_that_fit: <= len(demonstrations), >= 0
# ---------------------------------------------------------------------------

def test_n_shots_that_fit_bounds():
    demos = [make_demo(2, 2) for _ in range(5)]
    query = torch.randint(0, VOCAB, (3,))
    builder = ICLPromptBuilder(max_length=32, separator_id=1)
    n = builder.n_shots_that_fit(demos, query)
    assert 0 <= n <= len(demos)


# ---------------------------------------------------------------------------
# 12. ICLEvaluator.score_sequence: returns negative float (log probability)
# ---------------------------------------------------------------------------

def test_score_sequence_negative(model):
    ev = ICLEvaluator(model)
    prompt = torch.randint(0, VOCAB, (4,))
    label = torch.randint(0, VOCAB, (2,))
    score = ev.score_sequence(prompt, label)
    assert isinstance(score, float)
    assert score < 0.0, f"Log prob should be negative, got {score}"


# ---------------------------------------------------------------------------
# 13. ICLEvaluator.accuracy: returns float in [0,1]
# ---------------------------------------------------------------------------

def test_accuracy_range(model):
    ev = ICLEvaluator(model)
    prompts = [torch.randint(0, VOCAB, (4,)) for _ in range(2)]
    # Make candidates where index 0 is the "correct" label
    labels = [torch.tensor([3, 5]), torch.tensor([7, 2])]
    candidates = [
        [torch.tensor([3, 5]), torch.tensor([1, 2])],
        [torch.tensor([7, 2]), torch.tensor([4, 6])],
    ]
    acc = ev.accuracy(prompts, labels, candidates)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# 14. ICLEvaluator.calibration_bias: returns finite float
# ---------------------------------------------------------------------------

def test_calibration_bias_finite(model):
    ev = ICLEvaluator(model)
    prompts = [torch.randint(0, VOCAB, (4,)) for _ in range(2)]
    labels = [torch.randint(0, VOCAB, (2,)) for _ in range(2)]
    bias = ev.calibration_bias(prompts, labels)
    assert isinstance(bias, float)
    assert math.isfinite(bias), f"Expected finite bias, got {bias}"


# ---------------------------------------------------------------------------
# 15. More shots -> longer prompt (until max_length)
# ---------------------------------------------------------------------------

def test_more_shots_longer_prompt():
    query = torch.randint(0, VOCAB, (2,))
    builder = ICLPromptBuilder(max_length=64, separator_id=1)

    demos_2 = [make_demo(2, 1) for _ in range(2)]
    demos_4 = [make_demo(2, 1) for _ in range(4)]

    prompt_2 = builder.build(demos_2, query)
    prompt_4 = builder.build(demos_4, query)

    assert prompt_4.shape[0] >= prompt_2.shape[0], (
        f"4-shot prompt ({prompt_4.shape[0]}) should be >= 2-shot prompt ({prompt_2.shape[0]})"
    )


# ---------------------------------------------------------------------------
# 16. ExemplarSelector with k=0: returns empty list
# ---------------------------------------------------------------------------

def test_k_zero_returns_empty():
    pool = make_pool(4)
    selector = ExemplarSelector(method="random", k=0)
    result = selector.select(pool)
    assert result == []
