"""Tests for Token Budget Forcing (~14 tests)."""
from __future__ import annotations

import pytest
import torch
from torch import LongTensor, Tensor

from aurelius.inference.token_budget_forcing import (
    BudgetConfig,
    BudgetForcingDecoder,
    BudgetForcingLogitsProcessor,
    BudgetTracker,
)

# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32
ANSWER_TOK = 5          # arbitrary token that starts the answer phase
BUDGET = 4


# ---------------------------------------------------------------------------
# Helper: tiny deterministic model_fn
# ---------------------------------------------------------------------------

def _make_model_fn(vocab_size: int = VOCAB_SIZE, seed: int = 42):
    """Returns a callable that produces fixed random logits.

    The logits are constant so that generation is deterministic.
    """
    torch.manual_seed(seed)
    weight = torch.randn(vocab_size)

    def model_fn(input_ids: LongTensor) -> Tensor:
        # Returns (1, T, V) — all positions get the same fixed logits
        T = input_ids.shape[1]
        return weight.unsqueeze(0).unsqueeze(0).expand(1, T, vocab_size)

    return model_fn


# ===========================================================================
# 1. BudgetConfig dataclass has correct fields and defaults
# ===========================================================================

class TestBudgetConfig:
    def test_fields_and_defaults(self):
        cfg = BudgetConfig(thinking_budget=10, answer_start_token_id=ANSWER_TOK)
        assert cfg.thinking_budget == 10
        assert cfg.answer_start_token_id == ANSWER_TOK
        assert cfg.transition_bias == 100.0
        assert cfg.truncate_thinking is False

    def test_custom_values(self):
        cfg = BudgetConfig(
            thinking_budget=5,
            answer_start_token_id=3,
            transition_bias=50.0,
            truncate_thinking=True,
        )
        assert cfg.thinking_budget == 5
        assert cfg.answer_start_token_id == 3
        assert cfg.transition_bias == 50.0
        assert cfg.truncate_thinking is True


# ===========================================================================
# 2. BudgetTracker.step returns False when under budget
# ===========================================================================

class TestBudgetTrackerUnderBudget:
    def test_step_false_under_budget(self):
        cfg = BudgetConfig(thinking_budget=BUDGET, answer_start_token_id=ANSWER_TOK)
        tracker = BudgetTracker(cfg)
        # First token (non-answer) — count becomes 1, budget=4, not exhausted
        assert tracker.step(1) is False


# ===========================================================================
# 3. BudgetTracker.step returns True when budget exhausted
# ===========================================================================

class TestBudgetTrackerExhausted:
    def test_step_true_when_exhausted(self):
        cfg = BudgetConfig(thinking_budget=BUDGET, answer_start_token_id=ANSWER_TOK)
        tracker = BudgetTracker(cfg)
        for tok in range(BUDGET):       # tokens 0..3 (all non-answer-start)
            exhausted = tracker.step(tok)
        # After BUDGET steps the budget is exactly met
        assert exhausted is True


# ===========================================================================
# 4. BudgetTracker.reset resets count to 0
# ===========================================================================

class TestBudgetTrackerReset:
    def test_reset(self):
        cfg = BudgetConfig(thinking_budget=BUDGET, answer_start_token_id=ANSWER_TOK)
        tracker = BudgetTracker(cfg)
        tracker.step(1)
        tracker.step(2)
        assert tracker.thinking_count == 2
        tracker.reset()
        assert tracker.thinking_count == 0
        assert tracker.is_in_thinking_phase is True


# ===========================================================================
# 5. thinking_count increments each step
# ===========================================================================

class TestThinkingCountIncrements:
    def test_count_increments(self):
        cfg = BudgetConfig(thinking_budget=100, answer_start_token_id=ANSWER_TOK)
        tracker = BudgetTracker(cfg)
        # Use token IDs that are not equal to ANSWER_TOK (5)
        non_answer_tokens = [1, 2, 3, 4, 6]
        for i, tok in enumerate(non_answer_tokens, start=1):
            tracker.step(tok)
            assert tracker.thinking_count == i


# ===========================================================================
# 6. Tracker transitions out of thinking phase on answer_start_token_id
# ===========================================================================

class TestTrackerTransition:
    def test_transitions_on_answer_token(self):
        cfg = BudgetConfig(thinking_budget=100, answer_start_token_id=ANSWER_TOK)
        tracker = BudgetTracker(cfg)
        assert tracker.is_in_thinking_phase is True
        tracker.step(1)
        assert tracker.is_in_thinking_phase is True
        tracker.step(ANSWER_TOK)    # trigger transition
        assert tracker.is_in_thinking_phase is False

    def test_count_does_not_increment_after_transition(self):
        cfg = BudgetConfig(thinking_budget=100, answer_start_token_id=ANSWER_TOK)
        tracker = BudgetTracker(cfg)
        tracker.step(1)          # count = 1
        tracker.step(ANSWER_TOK) # transition
        count_after_transition = tracker.thinking_count
        tracker.step(2)          # answer-phase token — should not increment
        assert tracker.thinking_count == count_after_transition


# ===========================================================================
# 7. BudgetForcingLogitsProcessor adds bias when forcing
# ===========================================================================

class TestProcessorAddsBias:
    def test_adds_bias_when_budget_exhausted(self):
        cfg = BudgetConfig(
            thinking_budget=BUDGET,
            answer_start_token_id=ANSWER_TOK,
            transition_bias=100.0,
        )
        tracker = BudgetTracker(cfg)
        # Exhaust the budget
        for i in range(BUDGET):
            tracker.step(i)

        processor = BudgetForcingLogitsProcessor(cfg, tracker)
        scores = torch.zeros(VOCAB_SIZE)
        dummy_ids = torch.zeros(1, 1, dtype=torch.long)
        modified = processor(dummy_ids, scores)

        assert modified[ANSWER_TOK].item() == pytest.approx(100.0)


# ===========================================================================
# 8. Processor doesn't modify logits when under budget
# ===========================================================================

class TestProcessorNoModifyUnderBudget:
    def test_no_bias_under_budget(self):
        cfg = BudgetConfig(
            thinking_budget=BUDGET,
            answer_start_token_id=ANSWER_TOK,
            transition_bias=100.0,
        )
        tracker = BudgetTracker(cfg)  # fresh tracker — 0 steps
        processor = BudgetForcingLogitsProcessor(cfg, tracker)
        scores = torch.zeros(VOCAB_SIZE)
        dummy_ids = torch.zeros(1, 1, dtype=torch.long)
        modified = processor(dummy_ids, scores)
        # All logits should remain zero
        assert torch.all(modified == 0.0)


# ===========================================================================
# 9. Processor handles (B, V) input
# ===========================================================================

class TestProcessorBatchedInput:
    def test_batched_scores(self):
        B = 3
        cfg = BudgetConfig(
            thinking_budget=1,
            answer_start_token_id=ANSWER_TOK,
            transition_bias=50.0,
        )
        tracker = BudgetTracker(cfg)
        tracker.step(0)  # exhaust budget (count=1 >= budget=1)

        processor = BudgetForcingLogitsProcessor(cfg, tracker)
        scores = torch.zeros(B, VOCAB_SIZE)
        dummy_ids = torch.zeros(B, 1, dtype=torch.long)
        modified = processor(dummy_ids, scores)

        assert modified.shape == (B, VOCAB_SIZE)
        assert torch.allclose(modified[:, ANSWER_TOK], torch.full((B,), 50.0))


# ===========================================================================
# 10. BudgetForcingDecoder.generate output ≤ max_total_tokens
# ===========================================================================

class TestDecoderOutputLength:
    def test_length_at_most_max(self):
        cfg = BudgetConfig(thinking_budget=BUDGET, answer_start_token_id=ANSWER_TOK)
        model_fn = _make_model_fn()
        decoder = BudgetForcingDecoder(model_fn, cfg)
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        out = decoder.generate(prompt, max_total_tokens=10)
        assert out.shape[0] <= 10


# ===========================================================================
# 11. Generate returns LongTensor
# ===========================================================================

class TestDecoderReturnType:
    def test_returns_long_tensor(self):
        cfg = BudgetConfig(thinking_budget=BUDGET, answer_start_token_id=ANSWER_TOK)
        model_fn = _make_model_fn()
        decoder = BudgetForcingDecoder(model_fn, cfg)
        prompt = torch.tensor([1, 2], dtype=torch.long)
        out = decoder.generate(prompt, max_total_tokens=5)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.long


# ===========================================================================
# 12. budget=0: first token is answer_start_token_id or has highest forced logit
# ===========================================================================

class TestDecoderBudgetZero:
    def test_budget_zero_forces_answer_start(self):
        # With budget=0, the very first logit evaluation should have bias applied
        cfg = BudgetConfig(
            thinking_budget=0,
            answer_start_token_id=ANSWER_TOK,
            transition_bias=1000.0,  # large enough to always win
        )
        model_fn = _make_model_fn()
        decoder = BudgetForcingDecoder(model_fn, cfg)
        prompt = torch.tensor([1], dtype=torch.long)
        out = decoder.generate(prompt, max_total_tokens=1)
        # The first generated token should be answer_start_token_id
        assert out[0].item() == ANSWER_TOK


# ===========================================================================
# 13. Deterministic: same prompt → same output
# ===========================================================================

class TestDecoderDeterministic:
    def test_same_prompt_same_output(self):
        cfg = BudgetConfig(thinking_budget=BUDGET, answer_start_token_id=ANSWER_TOK)
        model_fn = _make_model_fn(seed=7)
        decoder = BudgetForcingDecoder(model_fn, cfg)
        prompt = torch.tensor([10, 20, 30], dtype=torch.long)
        out1 = decoder.generate(prompt, max_total_tokens=8)
        out2 = decoder.generate(prompt, max_total_tokens=8)
        assert torch.equal(out1, out2)


# ===========================================================================
# 14. max_total_tokens=1 works
# ===========================================================================

class TestDecoderMaxOne:
    def test_max_total_tokens_one(self):
        cfg = BudgetConfig(thinking_budget=BUDGET, answer_start_token_id=ANSWER_TOK)
        model_fn = _make_model_fn()
        decoder = BudgetForcingDecoder(model_fn, cfg)
        prompt = torch.tensor([1, 2], dtype=torch.long)
        out = decoder.generate(prompt, max_total_tokens=1)
        assert out.shape[0] == 1
        assert out.dtype == torch.long
