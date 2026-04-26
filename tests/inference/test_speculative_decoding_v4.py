"""Tests for speculative_decoding_v4.

Tiny config: vocab=16, d_model=8, n_draft=3, max_new_tokens=6, B=1.
All tests run forward/backward passes (where applicable) or full
generate() calls.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.inference.speculative_decoding_v4 import (
    DraftModel,
    SpeculativeDecoder,
    SpeculativeVerifier,
    SpeedupBenchmark,
    TargetModel,
)

# ---------------------------------------------------------------------------
# Shared tiny model factory
# ---------------------------------------------------------------------------

VOCAB = 16
D_MODEL = 8
N_DRAFT = 3
MAX_NEW = 6
SEQ_LEN = 4  # prompt length


def make_tiny_model() -> nn.Module:
    """Embedding + linear head: (T,) -> (T, VOCAB) logits."""

    class TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(VOCAB, D_MODEL)
            self.head = nn.Linear(D_MODEL, VOCAB)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            # Accept 1-D (T,) or 2-D (B, T)
            if input_ids.dim() == 1:
                x = self.embed(input_ids)  # (T, D)
                return self.head(x)  # (T, V)
            x = self.embed(input_ids)  # (B, T, D)
            return self.head(x)  # (B, T, V)

    return TinyLM()


def make_prompt() -> torch.Tensor:
    """1-D LongTensor prompt of length SEQ_LEN."""
    return torch.randint(0, VOCAB, (SEQ_LEN,))


def make_batched_prompt() -> torch.Tensor:
    """2-D (1, SEQ_LEN) prompt."""
    return torch.randint(0, VOCAB, (1, SEQ_LEN))


# ---------------------------------------------------------------------------
# SpeculativeVerifier tests (5)
# ---------------------------------------------------------------------------


def test_verifier_accept_when_target_dominates():
    """verify_token accepts when target_prob >> draft_prob."""
    verifier = SpeculativeVerifier()
    target_probs = torch.zeros(VOCAB)
    target_probs[3] = 0.95
    target_probs[5] = 0.05

    # draft_prob very small -> ratio >> 1 -> always accepted
    accepted_count = sum(verifier.verify_token(3, 1e-6, target_probs)[0] for _ in range(20))
    assert accepted_count == 20, f"Expected 20 accepts, got {accepted_count}"


def test_verifier_reject_when_draft_dominates():
    """verify_token rejects and corrects when draft_prob >> target_prob."""
    torch.manual_seed(0)
    verifier = SpeculativeVerifier()
    target_probs = torch.zeros(VOCAB)
    target_probs[3] = 0.01
    target_probs[7] = 0.99

    # draft assigned 0.99 to token 3, but target assigns only 0.01 -> ratio ~0.01
    # Over many trials, almost all should be rejected
    rejected_count = sum(not verifier.verify_token(3, 0.99, target_probs)[0] for _ in range(100))
    assert rejected_count >= 80, f"Expected many rejections, got {rejected_count}"


def test_verifier_rejection_correction_is_valid_vocab():
    """verify_token: corrected token on rejection must be a valid vocab id."""
    torch.manual_seed(42)
    verifier = SpeculativeVerifier()
    target_probs_skewed = torch.zeros(VOCAB)
    target_probs_skewed[1] = 0.001
    target_probs_skewed[2:] = (1.0 - 0.001) / (VOCAB - 2)

    for _ in range(10):
        accepted, tok = verifier.verify_token(1, 1.0, target_probs_skewed)
        # Whether accepted or not, tok must be in vocab
        assert 0 <= tok < VOCAB


def test_verify_sequence_n_accepted_in_range():
    """verify_sequence: n_accepted in [0, K] and final_token is valid vocab id."""
    torch.manual_seed(7)
    verifier = SpeculativeVerifier()
    K = N_DRAFT
    draft_tokens = torch.randint(0, VOCAB, (K,))
    draft_probs = torch.rand(K).clamp(0.01, 0.99)

    target_logits = torch.randn(K, VOCAB)

    n_accepted, final_token = verifier.verify_sequence(draft_tokens, draft_probs, target_logits)

    assert 0 <= n_accepted <= K, f"n_accepted={n_accepted} out of [0, {K}]"
    assert 0 <= final_token < VOCAB, f"final_token={final_token} out of vocab"


def test_verify_sequence_same_model_all_accepted():
    """With target=draft (same logits), all K tokens should be accepted."""
    torch.manual_seed(99)
    model = make_tiny_model()
    model.eval()

    K = N_DRAFT
    prompt = make_prompt()

    # Draft K tokens
    draft_model = DraftModel(model, temperature=1.0)
    draft_tokens, _ = draft_model.draft(prompt, K)

    # Score with the same model
    target_model = TargetModel(model)
    target_logits = target_model.score(prompt, draft_tokens)

    # Get exact probabilities the same model assigns at draft positions
    target_probs_all = torch.softmax(target_logits.float(), dim=-1)  # (K, V)
    draft_probs_exact = torch.stack([target_probs_all[i, draft_tokens[i]] for i in range(K)])

    verifier = SpeculativeVerifier()
    n_accepted, final_token = verifier.verify_sequence(
        draft_tokens, draft_probs_exact, target_logits
    )

    assert n_accepted == K, f"Same model should accept all {K} tokens, got {n_accepted}"
    assert 0 <= final_token < VOCAB


# ---------------------------------------------------------------------------
# DraftModel tests (3)
# ---------------------------------------------------------------------------


def test_draft_model_output_shapes():
    """DraftModel.draft returns tensors of shape (n_tokens,) for both outputs."""
    model = make_tiny_model()
    draft_model = DraftModel(model, temperature=1.0)
    prompt = make_prompt()

    draft_tokens, draft_probs = draft_model.draft(prompt, N_DRAFT)

    assert draft_tokens.shape == (N_DRAFT,), (
        f"draft_tokens shape {draft_tokens.shape} != ({N_DRAFT},)"
    )
    assert draft_probs.shape == (N_DRAFT,), f"draft_probs shape {draft_probs.shape} != ({N_DRAFT},)"


def test_draft_probs_valid_range():
    """All draft_probs must be in (0, 1] and all draft_tokens in valid vocab."""
    model = make_tiny_model()
    draft_model = DraftModel(model, temperature=1.0)
    prompt = make_prompt()

    draft_tokens, draft_probs = draft_model.draft(prompt, N_DRAFT)

    assert (draft_probs > 0).all(), "Some draft_probs are <= 0"
    assert (draft_probs <= 1.0 + 1e-6).all(), "Some draft_probs > 1"
    assert (draft_tokens >= 0).all() and (draft_tokens < VOCAB).all(), (
        f"Some draft_tokens out of [0, {VOCAB})"
    )


def test_draft_no_grad_does_not_accumulate():
    """DraftModel.draft runs under torch.no_grad() -- no grad accumulation."""
    model = make_tiny_model()
    draft_model = DraftModel(model, temperature=1.0)
    prompt = make_prompt()

    draft_tokens, draft_probs = draft_model.draft(prompt, N_DRAFT)

    for p in model.parameters():
        assert p.grad is None, "Grad accumulated inside DraftModel.draft"


# ---------------------------------------------------------------------------
# TargetModel tests (2)
# ---------------------------------------------------------------------------


def test_target_model_score_output_shape():
    """TargetModel.score returns shape (n_tokens, V)."""
    model = make_tiny_model()
    target_model = TargetModel(model)
    prompt = make_prompt()
    draft_tokens = torch.randint(0, VOCAB, (N_DRAFT,))

    logits = target_model.score(prompt, draft_tokens)

    assert logits.shape == (N_DRAFT, VOCAB), (
        f"score output shape {logits.shape} != ({N_DRAFT}, {VOCAB})"
    )


def test_target_model_logits_finite():
    """TargetModel.score logits must contain no NaN or Inf."""
    model = make_tiny_model()
    target_model = TargetModel(model)
    prompt = make_prompt()
    draft_tokens = torch.randint(0, VOCAB, (N_DRAFT,))

    logits = target_model.score(prompt, draft_tokens)

    assert torch.isfinite(logits).all(), "TargetModel.score returned non-finite logits"


# ---------------------------------------------------------------------------
# SpeculativeDecoder tests (5)
# ---------------------------------------------------------------------------


def _make_decoder(model_a=None, model_b=None):
    """Build a SpeculativeDecoder from two (possibly same) tiny models."""
    if model_a is None:
        model_a = make_tiny_model()
    if model_b is None:
        model_b = make_tiny_model()
    draft_model = DraftModel(model_a, temperature=1.0)
    target_model = TargetModel(model_b)
    verifier = SpeculativeVerifier()
    return SpeculativeDecoder(draft_model, target_model, verifier, n_draft_tokens=N_DRAFT)


def test_decoder_output_shape_batched():
    """generate returns output_ids of shape (B, T + max_new_tokens)."""
    decoder = _make_decoder()
    prompt = make_batched_prompt()  # (1, SEQ_LEN)

    output_ids, _ = decoder.generate(prompt, max_new_tokens=MAX_NEW)

    expected = (1, SEQ_LEN + MAX_NEW)
    assert output_ids.shape == expected, f"output_ids shape {output_ids.shape} != {expected}"


def test_decoder_stats_has_all_keys():
    """generate stats dict must contain all 4 required keys."""
    decoder = _make_decoder()
    prompt = make_batched_prompt()

    _, stats = decoder.generate(prompt, max_new_tokens=MAX_NEW)

    required = {"n_steps", "total_accepted", "total_drafted", "acceptance_rate"}
    assert required <= stats.keys(), f"Missing keys: {required - stats.keys()}"


def test_decoder_acceptance_rate_in_range():
    """acceptance_rate must be in [0, 1]."""
    decoder = _make_decoder()
    prompt = make_batched_prompt()

    _, stats = decoder.generate(prompt, max_new_tokens=MAX_NEW)

    ar = stats["acceptance_rate"]
    assert 0.0 <= ar <= 1.0, f"acceptance_rate={ar} out of [0, 1]"


def test_decoder_same_models_high_acceptance():
    """Using the same model for draft and target should yield acceptance near 1.0."""
    torch.manual_seed(123)
    shared_model = make_tiny_model()
    shared_model.eval()

    decoder = _make_decoder(model_a=shared_model, model_b=shared_model)
    prompt = make_batched_prompt()

    _, stats = decoder.generate(prompt, max_new_tokens=MAX_NEW)

    ar = stats["acceptance_rate"]
    # Same model -> ratio is always 1 -> should be very close to 1.0
    assert ar >= 0.9, f"Same-model acceptance_rate={ar:.3f} expected >= 0.9"


def test_decoder_full_loop_no_error():
    """Full generation with n_draft=3 speculative steps completes without error."""
    torch.manual_seed(55)
    decoder = _make_decoder()
    prompt = make_batched_prompt()

    # Should not raise
    output_ids, stats = decoder.generate(prompt, max_new_tokens=MAX_NEW)

    assert output_ids.dtype == torch.long
    assert stats["n_steps"] >= 1
    assert stats["total_drafted"] >= N_DRAFT


# ---------------------------------------------------------------------------
# SpeedupBenchmark tests (3)
# ---------------------------------------------------------------------------


def test_speedup_theoretical_gt_one():
    """theoretical_speedup > 1.0 for any acceptance_rate > 0."""
    bench = SpeedupBenchmark()
    for alpha in [0.1, 0.5, 0.8, 0.95, 0.99]:
        sp = bench.theoretical_speedup(alpha, N_DRAFT)
        assert sp > 1.0, f"theoretical_speedup({alpha}, {N_DRAFT}) = {sp:.4f} not > 1.0"


def test_speedup_empirical_formula():
    """empirical_speedup == (total_accepted + n_steps) / n_steps."""
    bench = SpeedupBenchmark()
    stats = {"n_steps": 7, "total_accepted": 14, "total_drafted": 21}
    expected = (14 + 7) / 7  # = 3.0
    result = bench.empirical_speedup(stats)
    assert abs(result - expected) < 1e-6, f"empirical_speedup={result} != expected {expected}"


def test_speedup_theoretical_alpha_one():
    """theoretical_speedup with acceptance_rate=1.0 equals n_draft + 1."""
    bench = SpeedupBenchmark()
    sp = bench.theoretical_speedup(1.0, N_DRAFT)
    assert abs(sp - (N_DRAFT + 1)) < 1e-5, (
        f"theoretical_speedup(1.0, {N_DRAFT}) = {sp} != {N_DRAFT + 1}"
    )
