from __future__ import annotations

import torch

from src.inference.speculative_engine import SpecConfig, SpecResult, SpeculativeEngine

VOCAB = 64
N_DRAFT = 4


def make_engine(n_draft=N_DRAFT, temperature=1.0, top_p=1.0, vocab_size=VOCAB):
    cfg = SpecConfig(
        n_draft_tokens=n_draft,
        temperature=temperature,
        top_p=top_p,
        vocab_size=vocab_size,
    )
    return SpeculativeEngine(cfg)


def uniform_logits(n, v=VOCAB):
    return torch.zeros(n, v)


def peaked_logits(n, tok, v=VOCAB, peak=10.0):
    logits = torch.zeros(n, v)
    logits[:, tok] = peak
    return logits


class TestSpecConfig:
    def test_defaults(self):
        cfg = SpecConfig()
        assert cfg.n_draft_tokens == 4
        assert cfg.temperature == 1.0
        assert cfg.top_p == 1.0
        assert cfg.vocab_size == 32000

    def test_custom_values(self):
        cfg = SpecConfig(n_draft_tokens=8, temperature=0.7, vocab_size=128)
        assert cfg.n_draft_tokens == 8
        assert cfg.temperature == 0.7
        assert cfg.vocab_size == 128


class TestSpecResult:
    def test_fields(self):
        r = SpecResult(accepted_tokens=[1, 2], n_accepted=2, bonus_token=3, acceptance_rate=0.5)
        assert r.accepted_tokens == [1, 2]
        assert r.n_accepted == 2
        assert r.bonus_token == 3
        assert r.acceptance_rate == 0.5

    def test_none_bonus(self):
        r = SpecResult(accepted_tokens=[], n_accepted=0, bonus_token=None, acceptance_rate=0.0)
        assert r.bonus_token is None


class TestDraft:
    def test_returns_n_draft_tokens(self):
        engine = make_engine()
        logits = uniform_logits(N_DRAFT)
        tokens = engine.draft(logits)
        assert len(tokens) == N_DRAFT

    def test_tokens_in_vocab_range(self):
        engine = make_engine()
        logits = uniform_logits(N_DRAFT)
        tokens = engine.draft(logits)
        assert all(0 <= t < VOCAB for t in tokens)

    def test_greedy_picks_argmax_at_zero_temp(self):
        engine = make_engine(temperature=0.0)
        logits = peaked_logits(N_DRAFT, tok=7)
        tokens = engine.draft(logits)
        assert all(t == 7 for t in tokens)

    def test_draft_is_list_of_int(self):
        engine = make_engine()
        tokens = engine.draft(uniform_logits(N_DRAFT))
        assert all(isinstance(t, int) for t in tokens)


class TestVerify:
    def _run_verify(self, engine, draft_tok):
        n = len(draft_tok)
        draft_logits = peaked_logits(n, tok=draft_tok[0])
        target_logits = peaked_logits(n + 1, tok=draft_tok[0])
        return engine.verify(draft_tok, draft_logits, target_logits)

    def test_result_type(self):
        engine = make_engine()
        result = self._run_verify(engine, [0, 1, 2, 3])
        assert isinstance(result, SpecResult)

    def test_n_accepted_leq_n_draft(self):
        engine = make_engine()
        result = self._run_verify(engine, [0, 1, 2, 3])
        assert result.n_accepted <= N_DRAFT

    def test_accepted_tokens_length_matches_n_accepted(self):
        engine = make_engine()
        result = self._run_verify(engine, [0, 1, 2, 3])
        assert len(result.accepted_tokens) == result.n_accepted

    def test_acceptance_rate_in_range(self):
        engine = make_engine()
        for _ in range(10):
            result = self._run_verify(engine, [0, 1, 2, 3])
            assert 0.0 <= result.acceptance_rate <= 1.0

    def test_all_accepted_when_draft_equals_target(self):
        torch.manual_seed(0)
        engine = make_engine()
        n = N_DRAFT
        logits = peaked_logits(n, tok=5, peak=100.0)
        target_logits = peaked_logits(n + 1, tok=5, peak=100.0)
        draft_tokens = [5] * n
        result = engine.verify(draft_tokens, logits, target_logits)
        assert result.n_accepted == n
        assert result.bonus_token is not None

    def test_rejection_produces_bonus_token(self):
        engine = make_engine()
        draft_logits = peaked_logits(N_DRAFT, tok=0, peak=100.0)
        target_logits = peaked_logits(N_DRAFT + 1, tok=63, peak=100.0)
        draft_tokens = [0] * N_DRAFT
        result = engine.verify(draft_tokens, draft_logits, target_logits)
        assert result.bonus_token is not None

    def test_bonus_token_in_vocab(self):
        engine = make_engine()
        result = self._run_verify(engine, [0, 1, 2, 3])
        if result.bonus_token is not None:
            assert 0 <= result.bonus_token < VOCAB

    def test_target_logits_narrower_than_draft_plus1(self):
        engine = make_engine()
        draft_logits = peaked_logits(N_DRAFT, tok=5)
        target_logits = peaked_logits(N_DRAFT, tok=5)
        draft_tokens = [5] * N_DRAFT
        result = engine.verify(draft_tokens, draft_logits, target_logits)
        assert isinstance(result, SpecResult)


class TestStep:
    def test_step_returns_spec_result(self):
        engine = make_engine()
        draft_logits = uniform_logits(N_DRAFT)
        target_logits = uniform_logits(N_DRAFT + 1)
        result = engine.step(draft_logits, target_logits)
        assert isinstance(result, SpecResult)

    def test_step_n_accepted_leq_n_draft(self):
        engine = make_engine()
        draft_logits = uniform_logits(N_DRAFT)
        target_logits = uniform_logits(N_DRAFT + 1)
        result = engine.step(draft_logits, target_logits)
        assert result.n_accepted <= N_DRAFT


class TestExpectedSpeedup:
    def test_speedup_gt1_high_acceptance(self):
        engine = make_engine()
        speedup = engine.expected_speedup(acceptance_rate=0.9)
        assert speedup > 1.0

    def test_speedup_low_acceptance_near1(self):
        engine = make_engine()
        speedup = engine.expected_speedup(acceptance_rate=0.0)
        assert speedup > 0.0

    def test_speedup_perfect_acceptance(self):
        engine = make_engine()
        speedup = engine.expected_speedup(acceptance_rate=1.0)
        assert speedup > 0.0

    def test_speedup_returns_float(self):
        engine = make_engine()
        assert isinstance(engine.expected_speedup(0.5), float)
