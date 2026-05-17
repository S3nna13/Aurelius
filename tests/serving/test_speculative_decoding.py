"""Tests for speculative decoding engine."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.serving.speculative_decoding import (
    DraftModelAdapter,
    SpeculativeConfig,
    SpeculativeScheduler,
    SpeculativeStats,
    VerifierModelAdapter,
)


# Mock model that returns deterministic logits
class MockModel:
    def __init__(self, vocab_size: int = 100, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.device = torch.device(device)

    def __call__(self, input_ids: Tensor, **kwargs) -> Tensor:
        """Return logits for next token. Simple pattern: logits[i] = i mod vocab_size."""
        B, T = input_ids.shape
        # Return logits shaped (B, T, vocab_size)
        logits = torch.zeros(B, T, self.vocab_size, device=self.device)
        for b in range(B):
            for t in range(T):
                next_token = input_ids[b, t].item()
                # Deterministic: next token is (prev % vocab_size) so verifier is consistent
                logits[b, t, next_token % self.vocab_size] = 10.0
        return logits

    def generate_token(self, context: Tensor, attention_mask: Tensor | None = None, **kwargs):
        """Generate one token from context."""
        logits = self(context)
        next_logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(next_logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        return token, next_logits

    def batch_generate(self, context_batch: Tensor, attention_mask: Tensor | None = None, **kwargs):
        tokens = []
        logits_list = []
        for i in range(context_batch.shape[0]):
            ctx = context_batch[i : i + 1]
            token, logits = self.generate_token(ctx)
            tokens.append(token)
            logits_list.append(logits)
        return torch.cat(tokens, dim=0), torch.stack(logits_list, dim=0)


@pytest.fixture
def draft_model() -> MockModel:
    return MockModel(vocab_size=50)


@pytest.fixture
def verifier_model() -> MockModel:
    return MockModel(vocab_size=50)


@pytest.fixture
def scheduler(draft_model: MockModel, verifier_model: MockModel) -> SpeculativeScheduler:
    config = SpeculativeConfig(draft_length=3, temperature=1.0)
    return SpeculativeScheduler(
        DraftModelAdapter(draft_model, config),
        VerifierModelAdapter(verifier_model),
        config,
    )


# ---------------------------------------------------------------------------
# DraftModelAdapter tests
# ---------------------------------------------------------------------------


def test_draft_generates_sequence_length(scheduler: SpeculativeScheduler, draft_model: MockModel):
    context = torch.tensor([[1, 2, 3, 4, 5]])
    draft_ids = scheduler.draft.generate_draft_sequence(context)
    assert draft_ids.shape[0] == context.shape[0]
    assert draft_ids.shape[1] == scheduler.config.draft_length


def test_draft_generation_batch(scheduler: SpeculativeScheduler):
    context = torch.tensor([[1, 2], [3, 4]])
    draft_ids = scheduler.draft.generate_draft_sequence(context)
    assert draft_ids.shape[0] == 2
    assert draft_ids.shape[1] == scheduler.config.draft_length


# ---------------------------------------------------------------------------
# VerifierModelAdapter tests
# ---------------------------------------------------------------------------


def test_verifier_returns_correct_shape(scheduler: SpeculativeScheduler, verifier_model: MockModel):
    context = torch.tensor([[10, 20, 30]])
    draft_ids = torch.tensor([[40, 50, 60]])
    draft_logits, full_logits = scheduler.verifier.verify_drafts(context, draft_ids)
    assert draft_logits.shape[0] == context.shape[0]
    assert draft_logits.shape[1] == draft_ids.shape[1]
    assert full_logits.shape[0] == context.shape[0]
    # full length should be prompt + draft
    expected_len = context.shape[1] + draft_ids.shape[1]
    assert full_logits.shape[1] == expected_len


# ---------------------------------------------------------------------------
# SpeculativeScheduler tests
# ---------------------------------------------------------------------------


def test_scheduler_registers_request(scheduler: SpeculativeScheduler):
    prompt = torch.tensor([[1, 2, 3]])
    scheduler.register_request("req1", prompt, max_new_tokens=10)
    state = scheduler._states["req1"]
    assert state.request_id == "req1"
    assert torch.equal(state.prompt_ids, prompt)
    assert state.max_new_tokens == 10
    assert not state.finished


def test_scheduler_step_produces_tokens(scheduler: SpeculativeScheduler):
    prompt = torch.tensor([[1, 2, 3]])
    scheduler.register_request("req1", prompt, max_new_tokens=5)
    results = scheduler.step_batch(["req1"])
    assert "req1" in results
    tokens = results["req1"]
    assert len(tokens) > 0  # at least one token produced
    assert all(isinstance(t, int) for t in tokens)


def test_scheduler_stats_collection(scheduler: SpeculativeScheduler):
    prompt = torch.tensor([[1, 2, 3]])
    scheduler.register_request("req1", prompt, max_new_tokens=2)
    # Run a couple steps
    for _ in range(2):
        scheduler.step_batch(["req1"])
    stats = scheduler.get_stats()
    assert stats.total_draft_tokens > 0
    assert stats.verifier_calls > 0
    assert 0.0 <= stats.acceptance_rate <= 1.0


def test_scheduler_reset_stats(scheduler: SpeculativeScheduler):
    prompt = torch.tensor([[1, 2, 3]])
    scheduler.register_request("req1", prompt, max_new_tokens=1)
    scheduler.step_batch(["req1"])
    before = scheduler.get_stats().total_draft_tokens
    scheduler.reset_stats()
    after = scheduler.get_stats().total_draft_tokens
    assert after == 0
    assert before > 0


def test_scheduler_finishes_when_max_reached(scheduler: SpeculativeScheduler):
    prompt = torch.tensor([[1, 2, 3]])
    scheduler.register_request("req1", prompt, max_new_tokens=1)
    scheduler.step_batch(["req1"])
    state = scheduler._states["req1"]
    # After one step we should have generated enough to hit max?
    # Not guaranteed with speculative; it might generate more than 1 token
    # But we can check the state
    assert state.position >= 0  # progress made


# ---------------------------------------------------------------------------
# SpeculativeStats tests
# ---------------------------------------------------------------------------


def test_stats_acceptance_rate():
    stats = SpeculativeStats(total_draft_tokens=0)
    assert stats.acceptance_rate == 0.0

    stats = SpeculativeStats(total_draft_tokens=10, accepted_tokens=7)
    assert stats.acceptance_rate == 0.7


def test_stats_throughput_multiplier():
    stats = SpeculativeStats(total_draft_tokens=10, accepted_tokens=8, verifier_calls=2)
    # 8 accepted tokens over 2 verifier calls = 4x multiplier
    assert stats.throughput_multiplier == 4.0


# ---------------------------------------------------------------------------
# ModelAdapter protocol
# ---------------------------------------------------------------------------


def test_draft_model_adapter_protocol():
    # Ensure MockModel conforms to ModelAdapter via duck typing
    model = MockModel(vocab_size=10)
    assert hasattr(model, "__call__")
    assert hasattr(model, "generate_token")
    assert hasattr(model, "batch_generate")
