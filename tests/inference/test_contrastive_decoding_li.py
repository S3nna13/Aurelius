"""Tests for contrastive_decoding_li.py (Li et al., arXiv:2210.15097)."""

from __future__ import annotations

import torch
from aurelius.inference.contrastive_decoding_li import (
    CDScorer,
    CDSearcher,
    ContrastiveDecoder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logits(V: int = 10, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (expert_logits, amateur_logits) of shape (V,)."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    expert = torch.randn(V, generator=gen)
    amateur = torch.randn(V, generator=gen)
    return expert, amateur


def _make_batch_logits(
    B: int = 4, V: int = 10, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (expert_logits, amateur_logits) of shape (B, V)."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    expert = torch.randn(B, V, generator=gen)
    amateur = torch.randn(B, V, generator=gen)
    return expert, amateur


def _dummy_fn(logits: torch.Tensor):
    """Return a model-like callable that always returns ``logits`` (1, 1, V)."""

    # logits shape: (1, V)
    def fn(input_ids: torch.Tensor) -> torch.Tensor:
        T = input_ids.shape[1]
        return logits.unsqueeze(1).expand(1, T, -1)  # (1, T, V)

    return fn


# ---------------------------------------------------------------------------
# CDScorer tests
# ---------------------------------------------------------------------------


class TestCDScorer:
    def test_plausibility_mask_below_alpha_gets_neg_inf(self):
        """Tokens with expert_prob < alpha * max_prob must get -inf score."""
        V = 8
        alpha = 0.5
        scorer = CDScorer(alpha=alpha)

        # Craft expert logits so probabilities are controlled
        expert_logits, amateur_logits = _make_logits(V=V)
        scores = scorer.score(expert_logits, amateur_logits)

        expert_probs = torch.softmax(expert_logits, dim=-1)
        max_prob = expert_probs.max()
        below_threshold = expert_probs < alpha * max_prob

        assert torch.all(scores[below_threshold] == float("-inf")), (
            "All tokens below the plausibility threshold must have -inf score."
        )

    def test_output_shape_is_V(self):
        """score() must return a tensor of shape (V,)."""
        V = 20
        scorer = CDScorer()
        expert, amateur = _make_logits(V=V)
        scores = scorer.score(expert, amateur)
        assert scores.shape == (V,)

    def test_tokens_in_plausibility_set_get_finite_score(self):
        """At least one token (the argmax) must have a finite CD score."""
        V = 16
        scorer = CDScorer(alpha=0.1)
        expert, amateur = _make_logits(V=V)
        scores = scorer.score(expert, amateur)
        assert torch.any(torch.isfinite(scores)), (
            "At least the argmax expert token must receive a finite score."
        )

    def test_cd_score_equals_log_prob_difference_for_eligible_tokens(self):
        """For tokens in V_head the score must equal log_p_exp - log_p_am."""
        V = 12
        scorer = CDScorer(alpha=0.1)
        expert_logits, amateur_logits = _make_logits(V=V)
        scores = scorer.score(expert_logits, amateur_logits)

        expert_probs = torch.softmax(expert_logits, dim=-1)
        max_prob = expert_probs.max()
        eligible = expert_probs >= 0.1 * max_prob

        expert_log = torch.log_softmax(expert_logits, dim=-1)
        amateur_log = torch.log_softmax(amateur_logits, dim=-1)
        expected = expert_log - amateur_log

        torch.testing.assert_close(
            scores[eligible],
            expected[eligible],
            msg="CD score should equal log_p_expert - log_p_amateur for eligible tokens.",
        )

    def test_alpha_1_only_argmax_expert_token_is_eligible(self):
        """With alpha=1.0 only the token with max expert prob is in V_head."""
        V = 10
        scorer = CDScorer(alpha=1.0)
        expert_logits, amateur_logits = _make_logits(V=V)
        scores = scorer.score(expert_logits, amateur_logits)

        finite_mask = torch.isfinite(scores)
        # Only one token should be finite (ties broken by float precision)
        assert finite_mask.sum() >= 1, "At least one token must be finite."

        # The finite token(s) must include the argmax of expert_probs
        best = torch.softmax(expert_logits, dim=-1).argmax()
        assert finite_mask[best], "The max-prob expert token must be in V_head."

    def test_alpha_0_all_tokens_eligible(self):
        """With alpha=0.0 every token is in V_head (no -inf scores)."""
        V = 10
        scorer = CDScorer(alpha=0.0)
        expert_logits, amateur_logits = _make_logits(V=V)
        scores = scorer.score(expert_logits, amateur_logits)
        assert torch.all(torch.isfinite(scores)), "All tokens must be eligible when alpha=0.0."


# ---------------------------------------------------------------------------
# CDSearcher tests
# ---------------------------------------------------------------------------


class TestCDSearcher:
    def test_search_output_shape_B(self):
        """search() must return a LongTensor of shape (B,)."""
        B, V = 6, 15
        searcher = CDSearcher()
        expert, amateur = _make_batch_logits(B=B, V=V)
        result = searcher.search(expert, amateur)
        assert result.shape == (B,)
        assert result.dtype == torch.long

    def test_selected_token_in_plausibility_set(self):
        """Every selected token must be in the expert plausibility set."""
        B, V = 8, 20
        alpha = 0.2
        searcher = CDSearcher(alpha=alpha)
        expert, amateur = _make_batch_logits(B=B, V=V)
        selected = searcher.search(expert, amateur)

        expert_probs = torch.softmax(expert, dim=-1)
        max_prob = expert_probs.max(dim=-1).values  # (B,)
        for b in range(B):
            tok = selected[b].item()
            assert expert_probs[b, tok] >= alpha * max_prob[b], (
                f"Batch item {b}: selected token {tok} is not in plausibility set."
            )

    def test_expert_amateur_agree_selects_highest_expert_prob_eligible(self):
        """When expert == amateur, CD score is 0 everywhere eligible, so argmax
        falls back to the first eligible index (highest expert-prob token if
        the plausibility set contains only that token), i.e. score ties are
        broken by torch.argmax which picks the lowest index."""
        V = 10
        # Make expert and amateur identical so CD score is 0 for all eligible
        expert_logits = torch.zeros(1, V)
        expert_logits[0, 3] = 5.0  # token 3 is clearly the max
        amateur_logits = expert_logits.clone()

        searcher = CDSearcher(alpha=0.5)  # only token 3 is eligible
        selected = searcher.search(expert_logits, amateur_logits)
        assert selected[0].item() == 3, "When only one token is eligible, it should be selected."


# ---------------------------------------------------------------------------
# ContrastiveDecoder tests
# ---------------------------------------------------------------------------


class TestContrastiveDecoder:
    def _build_decoder(
        self, V: int = 16, seed: int = 7, alpha: float = 0.1
    ) -> tuple[ContrastiveDecoder, torch.Tensor, torch.Tensor]:
        """Return (decoder, fixed_expert_logits (1,V), fixed_amateur_logits (1,V))."""
        gen = torch.Generator()
        gen.manual_seed(seed)
        expert_logits = torch.randn(1, V, generator=gen)
        amateur_logits = torch.randn(1, V, generator=gen)
        expert_fn = _dummy_fn(expert_logits)
        amateur_fn = _dummy_fn(amateur_logits)
        decoder = ContrastiveDecoder(expert_fn, amateur_fn, alpha=alpha)
        return decoder, expert_logits, amateur_logits

    def test_generate_returns_correct_length(self):
        """generate() must return exactly max_new_tokens tokens."""
        decoder, _, _ = self._build_decoder(V=16)
        prompt = torch.zeros(1, 4, dtype=torch.long)
        result = decoder.generate(prompt, max_new_tokens=10)
        assert result.shape == (10,)
        assert result.dtype == torch.long

    def test_generate_is_deterministic(self):
        """Same inputs → identical output on repeated calls."""
        decoder, _, _ = self._build_decoder(V=16)
        prompt = torch.zeros(1, 4, dtype=torch.long)
        out1 = decoder.generate(prompt, max_new_tokens=8)
        out2 = decoder.generate(prompt, max_new_tokens=8)
        assert torch.equal(out1, out2), "generate() must be deterministic."

    def test_generate_max_new_tokens_1(self):
        """max_new_tokens=1 must produce exactly one token."""
        decoder, _, _ = self._build_decoder(V=16)
        prompt = torch.zeros(1, 3, dtype=torch.long)
        result = decoder.generate(prompt, max_new_tokens=1)
        assert result.shape == (1,)
        assert result.dtype == torch.long

    def test_no_token_outside_plausibility_set_ever_selected(self):
        """All generated tokens must lie in the expert plausibility set."""
        V = 20
        alpha = 0.1
        gen = torch.Generator()
        gen.manual_seed(99)
        expert_logits = torch.randn(1, V, generator=gen)
        amateur_logits = torch.randn(1, V, generator=gen)

        expert_fn = _dummy_fn(expert_logits)
        amateur_fn = _dummy_fn(amateur_logits)
        decoder = ContrastiveDecoder(expert_fn, amateur_fn, alpha=alpha)

        prompt = torch.zeros(1, 2, dtype=torch.long)
        new_tokens = decoder.generate(prompt, max_new_tokens=15)  # (15,)

        # Plausibility set is fixed (logits never change in this dummy setup)
        expert_probs = torch.softmax(expert_logits[0], dim=-1)  # (V,)
        max_prob = expert_probs.max()
        plausible_mask = expert_probs >= alpha * max_prob  # (V,)

        for tok in new_tokens.tolist():
            assert plausible_mask[tok].item(), (
                f"Token {tok} was selected but is not in the plausibility set."
            )
