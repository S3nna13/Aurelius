"""Tests for src/inference/speculative_decoding_v3.py.

All tests use tiny tensors / small vocabs so they run fast without a GPU.
No HuggingFace, scipy, or sklearn dependencies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.inference.speculative_decoding_v3 import (
    SpeculativeConfig,
    SpeculativeDecoder,
    compute_acceptance_prob,
    compute_speedup_ratio,
    sample_from_logits,
    speculative_sample_step,
)

# ---------------------------------------------------------------------------
# Tiny model helpers
# ---------------------------------------------------------------------------

VOCAB = 16  # small vocab for fast tests


def _uniform_model_fn(input_ids: torch.Tensor) -> torch.Tensor:
    """Returns uniform logits for every position — deterministic and simple."""
    seq_len = input_ids.shape[0]
    return torch.zeros(seq_len, VOCAB)


def _peaked_model_fn(peak_token: int = 0):
    """Returns logits strongly peaked at ``peak_token``."""

    def fn(input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[0]
        logits = torch.full((seq_len, VOCAB), -1e6)
        logits[:, peak_token] = 0.0
        return logits

    return fn


def _make_decoder(
    draft_fn=None,
    target_fn=None,
    n_draft_tokens: int = 3,
    max_new_tokens: int = 10,
) -> SpeculativeDecoder:
    if draft_fn is None:
        draft_fn = _uniform_model_fn
    if target_fn is None:
        target_fn = _uniform_model_fn
    cfg = SpeculativeConfig(
        n_draft_tokens=n_draft_tokens,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=2,
    )
    return SpeculativeDecoder(draft_fn, target_fn, cfg)


# ---------------------------------------------------------------------------
# 1. SpeculativeConfig defaults
# ---------------------------------------------------------------------------


class TestSpeculativeConfigDefaults:
    def test_default_n_draft_tokens(self):
        cfg = SpeculativeConfig()
        assert cfg.n_draft_tokens == 4

    def test_default_temperature(self):
        cfg = SpeculativeConfig()
        assert cfg.temperature == 1.0

    def test_default_top_p(self):
        cfg = SpeculativeConfig()
        assert cfg.top_p == 1.0

    def test_default_max_new_tokens(self):
        cfg = SpeculativeConfig()
        assert cfg.max_new_tokens == 100

    def test_default_eos_token_id(self):
        cfg = SpeculativeConfig()
        assert cfg.eos_token_id == 2


# ---------------------------------------------------------------------------
# 2. sample_from_logits — valid token id
# ---------------------------------------------------------------------------


class TestSampleFromLogits:
    def test_returns_int(self):
        logits = torch.randn(VOCAB)
        token = sample_from_logits(logits)
        assert isinstance(token, int)

    def test_in_range(self):
        logits = torch.randn(VOCAB)
        for _ in range(30):
            token = sample_from_logits(logits)
            assert 0 <= token < VOCAB

    def test_near_greedy_selects_argmax(self):
        """With very low temperature the highest-logit token should be sampled almost always."""
        torch.manual_seed(42)
        logits = torch.zeros(VOCAB)
        logits[7] = 100.0  # strongly peaked at 7
        results = {sample_from_logits(logits, temperature=0.01) for _ in range(20)}
        assert 7 in results
        # With such a strong peak, nearly all samples should be 7
        counts = [sample_from_logits(logits, temperature=0.01) for _ in range(50)]
        assert counts.count(7) >= 48

    def test_top_p_restricts_vocab(self):
        """top_p < 1 should restrict sampling to a subset of tokens."""
        torch.manual_seed(0)
        # Two tokens at high prob, rest negligible
        logits = torch.full((VOCAB,), -100.0)
        logits[3] = 5.0
        logits[5] = 4.0
        tokens = {sample_from_logits(logits, temperature=1.0, top_p=0.95) for _ in range(40)}
        # Only tokens 3 and 5 should ever be sampled
        assert tokens.issubset({3, 5})

    def test_uniform_logits_all_tokens_reachable(self):
        """Uniform logits — all tokens should be reachable over many draws."""
        torch.manual_seed(1)
        logits = torch.zeros(VOCAB)
        seen = set()
        for _ in range(300):
            seen.add(sample_from_logits(logits))
        assert len(seen) == VOCAB


# ---------------------------------------------------------------------------
# 3. compute_acceptance_prob
# ---------------------------------------------------------------------------


class TestComputeAcceptanceProb:
    def test_returns_one_when_target_ge_draft(self):
        assert compute_acceptance_prob(0.3, 0.5) == 1.0

    def test_returns_one_when_equal(self):
        assert compute_acceptance_prob(0.4, 0.4) == 1.0

    def test_less_than_one_when_draft_exceeds_target(self):
        result = compute_acceptance_prob(0.8, 0.4)
        assert result < 1.0
        assert abs(result - 0.5) < 1e-6

    def test_clamped_to_zero_lower_bound(self):
        # ratio can never be negative — probabilities are non-negative
        result = compute_acceptance_prob(1.0, 0.0)
        assert result == 0.0

    def test_zero_draft_prob_returns_one(self):
        """If the draft model assigns zero probability, accept unconditionally."""
        result = compute_acceptance_prob(0.0, 0.5)
        assert result == 1.0

    def test_result_in_unit_interval(self):
        for dp, tp in [(0.1, 0.9), (0.9, 0.1), (0.5, 0.5), (0.0, 0.3)]:
            r = compute_acceptance_prob(dp, tp)
            assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# 4. speculative_sample_step shapes
# ---------------------------------------------------------------------------


class TestSpeculativeSampleStep:
    def _uniform_probs(self, n, vocab=VOCAB):
        return torch.full((n, vocab), 1.0 / vocab)

    def test_accepted_mask_shape(self):
        n_draft = 5
        dp = self._uniform_probs(n_draft)
        tp = self._uniform_probs(n_draft)
        mask, _ = speculative_sample_step(dp, tp)
        assert mask.shape == (n_draft,)

    def test_accepted_mask_is_bool(self):
        n_draft = 4
        dp = self._uniform_probs(n_draft)
        tp = self._uniform_probs(n_draft)
        mask, _ = speculative_sample_step(dp, tp)
        assert mask.dtype == torch.bool

    def test_corrected_probs_shape(self):
        n_draft = 3
        vocab = 8
        dp = torch.full((n_draft, vocab), 1.0 / vocab)
        tp = torch.full((n_draft, vocab), 1.0 / vocab)
        _, corrected = speculative_sample_step(dp, tp)
        assert corrected.shape == (vocab,)

    def test_corrected_probs_sums_to_one(self):
        n_draft = 4
        vocab = 10
        torch.manual_seed(5)
        dp = F.softmax(torch.randn(n_draft, vocab), dim=-1)
        tp = F.softmax(torch.randn(n_draft, vocab), dim=-1)
        _, corrected = speculative_sample_step(dp, tp)
        assert abs(corrected.sum().item() - 1.0) < 1e-5

    def test_all_accepted_when_target_dominates(self):
        """All tokens accepted when target_prob >= draft_prob at every argmax position.

        speculative_sample_step identifies each row's drafted token as the argmax of
        draft_probs.  If target_prob at that argmax >= draft_prob at that argmax, the
        acceptance ratio is >= 1 and the token is always accepted.  We construct draft
        and target probs so that the argmax of each draft row is token 0 (since all
        rows are uniform except token 0 which is slightly higher), and the target prob
        at token 0 is strictly higher than the draft prob there.
        """
        torch.manual_seed(7)
        n_draft = 6
        vocab = VOCAB
        # Draft: slightly peaked at token 0 so argmax=0 for every row
        draft_raw = torch.full((n_draft, vocab), 1.0)
        draft_raw[:, 0] = 2.0  # argmax is token 0 in every row
        draft_probs = F.normalize(draft_raw, p=1, dim=-1)

        # Target: even more peaked at token 0 → target_prob[i,0] > draft_prob[i,0]
        # so acceptance ratio min(1, q/p) = 1 for every row
        target_raw = torch.full((n_draft, vocab), 1.0)
        target_raw[:, 0] = 100.0  # dominant mass at token 0
        target_probs = F.normalize(target_raw, p=1, dim=-1)

        # With ratio >= 1 at the argmax token, every row must be accepted
        for _ in range(10):
            mask, _ = speculative_sample_step(draft_probs, target_probs)
            assert mask.all(), f"Expected all accepted, got mask={mask.tolist()}"


# ---------------------------------------------------------------------------
# 5. SpeculativeDecoder
# ---------------------------------------------------------------------------


class TestSpeculativeDecoderDraftTokens:
    def test_draft_token_ids_shape(self):
        decoder = _make_decoder(n_draft_tokens=4)
        prompt = torch.tensor([1, 5, 3], dtype=torch.long)
        ids, probs = decoder.draft_tokens(prompt)
        assert ids.shape == (4,)

    def test_draft_probs_shape(self):
        n_draft = 3
        decoder = _make_decoder(n_draft_tokens=n_draft)
        prompt = torch.tensor([1, 2], dtype=torch.long)
        ids, probs = decoder.draft_tokens(prompt)
        assert probs.shape == (n_draft, VOCAB)

    def test_draft_probs_sum_to_one(self):
        decoder = _make_decoder(n_draft_tokens=4)
        prompt = torch.tensor([0, 1], dtype=torch.long)
        _, probs = decoder.draft_tokens(prompt)
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)

    def test_draft_token_ids_in_range(self):
        decoder = _make_decoder(n_draft_tokens=5)
        prompt = torch.tensor([1], dtype=torch.long)
        ids, _ = decoder.draft_tokens(prompt)
        assert ids.min().item() >= 0
        assert ids.max().item() < VOCAB


class TestSpeculativeDecoderVerifyAndAccept:
    def test_returns_tensor(self):
        decoder = _make_decoder(n_draft_tokens=3)
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        draft_ids, draft_probs = decoder.draft_tokens(prompt)
        accepted, n_acc = decoder.verify_and_accept(prompt, draft_ids, draft_probs)
        assert isinstance(accepted, torch.Tensor)

    def test_n_accepted_matches_length(self):
        decoder = _make_decoder(n_draft_tokens=3)
        prompt = torch.tensor([1, 2], dtype=torch.long)
        draft_ids, draft_probs = decoder.draft_tokens(prompt)
        accepted, n_acc = decoder.verify_and_accept(prompt, draft_ids, draft_probs)
        assert n_acc == len(accepted)

    def test_accepted_at_least_one_token(self):
        """Must produce at least one token (the corrected / bonus token)."""
        decoder = _make_decoder(n_draft_tokens=4)
        prompt = torch.tensor([0], dtype=torch.long)
        draft_ids, draft_probs = decoder.draft_tokens(prompt)
        accepted, n_acc = decoder.verify_and_accept(prompt, draft_ids, draft_probs)
        assert n_acc >= 1

    def test_accepted_at_most_n_draft_plus_one(self):
        """Cannot accept more than n_draft + 1 tokens (n_draft + bonus)."""
        n_draft = 4
        decoder = _make_decoder(n_draft_tokens=n_draft)
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        draft_ids, draft_probs = decoder.draft_tokens(prompt)
        _, n_acc = decoder.verify_and_accept(prompt, draft_ids, draft_probs)
        assert n_acc <= n_draft + 1

    def test_all_accepted_with_identical_models(self):
        """When draft == target the draft tokens are accepted with probability 1."""
        torch.manual_seed(0)
        # Strongly peaked model — draft and target agree
        model_fn = _peaked_model_fn(peak_token=3)
        cfg = SpeculativeConfig(n_draft_tokens=4, max_new_tokens=20)
        decoder = SpeculativeDecoder(model_fn, model_fn, cfg)
        prompt = torch.tensor([1, 5], dtype=torch.long)
        draft_ids, draft_probs = decoder.draft_tokens(prompt)
        accepted, n_acc = decoder.verify_and_accept(prompt, draft_ids, draft_probs)
        # All 4 draft tokens accepted plus 1 bonus = 5
        assert n_acc == 5
        # Every accepted token should be 3 (the peak)
        assert accepted.tolist() == [3] * 5


# ---------------------------------------------------------------------------
# 6. decode end-to-end
# ---------------------------------------------------------------------------


class TestDecodeEndToEnd:
    def test_output_longer_than_prompt(self):
        decoder = _make_decoder(max_new_tokens=8)
        prompt = torch.tensor([1, 2], dtype=torch.long)
        output = decoder.decode(prompt)
        assert output.shape[0] > prompt.shape[0]

    def test_output_starts_with_prompt(self):
        decoder = _make_decoder(max_new_tokens=6)
        prompt = torch.tensor([5, 9, 3], dtype=torch.long)
        output = decoder.decode(prompt)
        assert output[:3].tolist() == [5, 9, 3]

    def test_stops_at_eos(self):
        """EOS model always produces token 2 — decode should stop immediately."""
        eos_id = 2
        model_fn = _peaked_model_fn(peak_token=eos_id)
        cfg = SpeculativeConfig(n_draft_tokens=4, max_new_tokens=50, eos_token_id=eos_id)
        decoder = SpeculativeDecoder(model_fn, model_fn, cfg)
        prompt = torch.tensor([1], dtype=torch.long)
        output = decoder.decode(prompt)
        # Should stop very shortly after EOS is generated
        assert output.shape[0] <= len(prompt) + cfg.n_draft_tokens + 2

    def test_respects_max_new_tokens(self):
        max_new = 5
        decoder = _make_decoder(max_new_tokens=max_new)
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        output = decoder.decode(prompt)
        n_generated = output.shape[0] - prompt.shape[0]
        assert n_generated <= max_new


# ---------------------------------------------------------------------------
# 7. compute_speedup_ratio
# ---------------------------------------------------------------------------


class TestComputeSpeedupRatio:
    def test_correct_value(self):
        ratio = compute_speedup_ratio(n_tokens_generated=20, n_target_calls=5)
        assert abs(ratio - 4.0) < 1e-9

    def test_handles_zero_target_calls(self):
        ratio = compute_speedup_ratio(n_tokens_generated=10, n_target_calls=0)
        assert ratio == 0.0

    def test_ratio_one_for_sequential(self):
        """Vanilla decoding: one call per token → ratio = 1."""
        ratio = compute_speedup_ratio(n_tokens_generated=100, n_target_calls=100)
        assert abs(ratio - 1.0) < 1e-9

    def test_fractional_ratio(self):
        ratio = compute_speedup_ratio(n_tokens_generated=7, n_target_calls=3)
        assert abs(ratio - 7 / 3) < 1e-9
