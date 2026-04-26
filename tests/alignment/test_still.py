"""Tests for src/alignment/still.py — STILL: Self-Taught Iterative Learning Loop.

Tiny config: vocab_size=256, seq_len=16, n_samples=4, keep_top_k=2, min_score=0.3
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alignment.still import (
    ExactMatchVerifier,
    RubricVerifier,
    STILLConfig,
    STILLSample,
    STILLTrainer,
)

# ---------------------------------------------------------------------------
# MockLM — self-contained, does NOT import from src/model/transformer.py
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
SEQ_LEN = 16


class MockLM(nn.Module):
    """Tiny language model: input_ids (B, T) -> logits (B, T, vocab_size).

    Also accepts ``labels`` to return a cross-entropy loss as the first
    element of a 3-tuple ``(loss, logits, [])``, matching the project
    convention used by RFT, RLVR, etc.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        x = self.embed(input_ids)  # (B, T, hidden)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return loss, logits, []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TINY_CFG = STILLConfig(
    n_iterations=2,
    n_samples_per_prompt=4,
    keep_top_k=2,
    min_score=0.3,
    temperature=0.8,
    max_new_tokens=8,
)


def _make_trainer(verifier=None, cfg=None):
    if cfg is None:
        cfg = TINY_CFG
    if verifier is None:
        verifier = ExactMatchVerifier()
    model = MockLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return STILLTrainer(model=model, optimizer=optimizer, verifier=verifier, config=cfg)


def _prompts(n: int = 2):
    prompts = [f"prompt_{i}" for i in range(n)]
    input_ids = [torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN)) for _ in range(n)]
    return prompts, input_ids


def _make_sample(score: float, text: str = "42 7 9", iteration: int = 0) -> STILLSample:
    return STILLSample(prompt="q", response=text, score=score, iteration=iteration)


# ---------------------------------------------------------------------------
# Test 1 — ExactMatchVerifier: returns 1.0 when answer is present
# ---------------------------------------------------------------------------


def test_exact_match_hit():
    v = ExactMatchVerifier()
    score = v.score("what is 2+2?", "the answer is 4", reference="4")
    assert score == 1.0, f"Expected 1.0, got {score}"


# ---------------------------------------------------------------------------
# Test 2 — ExactMatchVerifier: returns 0.0 when answer is absent
# ---------------------------------------------------------------------------


def test_exact_match_miss():
    v = ExactMatchVerifier()
    score = v.score("what is 2+2?", "I don't know", reference="4")
    assert score == 0.0, f"Expected 0.0, got {score}"


# ---------------------------------------------------------------------------
# Test 3 — RubricVerifier: proportional to matched items
# ---------------------------------------------------------------------------


def test_rubric_verifier_partial():
    v = RubricVerifier(["step", "therefore", "answer"])
    # Response contains "step" and "answer" but not "therefore" -> 2/3
    score = v.score("q", "step 1: compute... answer: 42", reference=None)
    assert abs(score - 2 / 3) < 1e-6, f"Expected 0.666…, got {score}"


def test_rubric_verifier_full():
    v = RubricVerifier(["alpha", "beta"])
    score = v.score("q", "alpha then beta", reference=None)
    assert score == 1.0, f"Expected 1.0, got {score}"


def test_rubric_verifier_none():
    v = RubricVerifier(["alpha", "beta"])
    score = v.score("q", "nothing relevant here", reference=None)
    assert score == 0.0, f"Expected 0.0, got {score}"


# ---------------------------------------------------------------------------
# Test 4 — filter_samples: keeps only samples above min_score
# ---------------------------------------------------------------------------


def test_filter_keeps_above_min_score():
    trainer = _make_trainer()

    # Manually build candidates with known scores via a verifier that returns
    # the embedded score in the response string.
    class FixedVerifier:
        def __init__(self, scores):
            self._scores = scores
            self._idx = 0

        def score(self, prompt, response, reference=None):
            v = self._scores[self._idx % len(self._scores)]
            self._idx += 1
            return v

    scores_per_group = [0.1, 0.4, 0.2, 0.8]  # only 0.4 and 0.8 >= min_score=0.3
    trainer.verifier = FixedVerifier(scores_per_group)

    candidates = [[_make_sample(0.0, str(i)) for i in range(4)]]
    kept = trainer.filter_samples(candidates)
    # All returned samples must have score >= min_score after verifier runs
    assert all(s.score >= TINY_CFG.min_score for s in kept), (
        f"Unexpected scores: {[s.score for s in kept]}"
    )


# ---------------------------------------------------------------------------
# Test 5 — filter_samples: returns at most keep_top_k per prompt
# ---------------------------------------------------------------------------


def test_filter_at_most_keep_top_k():
    class AlwaysHighVerifier:
        def score(self, prompt, response, reference=None):
            return 1.0

    trainer = _make_trainer(verifier=AlwaysHighVerifier())
    # 4 samples per prompt, keep_top_k=2
    candidates = [[_make_sample(1.0, str(i)) for i in range(4)]]
    kept = trainer.filter_samples(candidates)
    assert len(kept) <= TINY_CFG.keep_top_k, f"Expected <= {TINY_CFG.keep_top_k}, got {len(kept)}"


# ---------------------------------------------------------------------------
# Test 6 — STILLSample has all required fields
# ---------------------------------------------------------------------------


def test_still_sample_fields():
    s = STILLSample(prompt="p", response="r", score=0.9, iteration=2)
    assert hasattr(s, "prompt")
    assert hasattr(s, "response")
    assert hasattr(s, "score")
    assert hasattr(s, "iteration")
    assert s.prompt == "p"
    assert s.response == "r"
    assert s.score == 0.9
    assert s.iteration == 2


# ---------------------------------------------------------------------------
# Test 7 — sft_step: returns dict with 'loss' key
# ---------------------------------------------------------------------------


def test_sft_step_returns_loss_key():
    trainer = _make_trainer()
    samples = [_make_sample(0.9, "1 2 3 4 5"), _make_sample(0.8, "6 7 8")]
    result = trainer.sft_step(samples)
    assert "loss" in result, f"'loss' key missing from {result}"


# ---------------------------------------------------------------------------
# Test 8 — sft_step: gradient flows to model parameters
# ---------------------------------------------------------------------------


def test_sft_step_gradient_flows():
    trainer = _make_trainer()
    samples = [_make_sample(1.0, "10 20 30 40 50 60 70")]

    # Record params before step
    before = {name: param.clone().detach() for name, param in trainer.model.named_parameters()}

    trainer.sft_step(samples)

    # At least one parameter should have changed
    changed = any(
        not torch.allclose(before[name], param) for name, param in trainer.model.named_parameters()
    )
    assert changed, "No model parameters changed after sft_step — gradient did not flow"


# ---------------------------------------------------------------------------
# Test 9 — run_iteration: returns stats with 'n_kept' key
# ---------------------------------------------------------------------------


def test_run_iteration_has_n_kept():
    class AlwaysHighVerifier:
        def score(self, prompt, response, reference=None):
            return 1.0

    trainer = _make_trainer(verifier=AlwaysHighVerifier())
    prompts, input_ids = _prompts(2)
    stats = trainer.run_iteration(prompts, input_ids)
    assert "n_kept" in stats, f"'n_kept' missing from stats: {stats}"
    assert isinstance(stats["n_kept"], int)


# ---------------------------------------------------------------------------
# Test 10 — run_iteration with all-zero scores returns empty filtered set
# ---------------------------------------------------------------------------


def test_run_iteration_zero_scores_empty_kept():
    class ZeroVerifier:
        def score(self, prompt, response, reference=None):
            return 0.0

    trainer = _make_trainer(verifier=ZeroVerifier())
    prompts, input_ids = _prompts(2)
    stats = trainer.run_iteration(prompts, input_ids)
    assert stats["n_kept"] == 0, f"Expected n_kept=0 when all scores=0, got {stats['n_kept']}"


# ---------------------------------------------------------------------------
# Test 11 — higher temperature -> more diverse token distribution
# ---------------------------------------------------------------------------


def test_higher_temperature_more_diverse():
    """Higher temperature should produce a flatter token distribution.

    We measure diversity as the number of unique first tokens across many
    samples.  With temperature~0 (greedy) we expect very low diversity;
    with temperature=2.0 we expect higher diversity.
    """
    torch.manual_seed(0)
    model = MockLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # no updates

    n_samples = 32
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))

    def collect_first_tokens(temp: float) -> set:
        cfg = STILLConfig(
            n_samples_per_prompt=n_samples,
            keep_top_k=n_samples,
            min_score=0.0,
            temperature=temp,
            max_new_tokens=1,
        )
        trainer = STILLTrainer(
            model=model,
            optimizer=optimizer,
            verifier=ExactMatchVerifier(),
            config=cfg,
        )
        groups = trainer.generate_candidates(["p"], [input_ids])
        return {s.response for s in groups[0]}

    low_diversity = collect_first_tokens(0.01)
    high_diversity = collect_first_tokens(5.0)

    assert len(high_diversity) >= len(low_diversity), (
        f"Expected higher temp to produce >= diversity: "
        f"low={len(low_diversity)}, high={len(high_diversity)}"
    )


# ---------------------------------------------------------------------------
# Test 12 — n_iterations loop runs without error
# ---------------------------------------------------------------------------


def test_n_iterations_loop():
    class AlwaysHighVerifier:
        def score(self, prompt, response, reference=None):
            return 1.0

    cfg = STILLConfig(
        n_iterations=3,
        n_samples_per_prompt=2,
        keep_top_k=1,
        min_score=0.5,
        temperature=1.0,
        max_new_tokens=4,
    )
    model = MockLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = STILLTrainer(
        model=model,
        optimizer=optimizer,
        verifier=AlwaysHighVerifier(),
        config=cfg,
    )
    prompts, input_ids = _prompts(2)
    history = trainer.run(prompts, input_ids)

    assert len(history) == 3, f"Expected 3 iteration records, got {len(history)}"
    for i, stats in enumerate(history):
        assert "loss" in stats, f"Iter {i} stats missing 'loss'"
        assert "n_kept" in stats, f"Iter {i} stats missing 'n_kept'"
