"""
Tests for src/alignment/constitutional_constraints.py

Parameters used throughout:
    vocab_size = 16
    d_model    = 16
    B          = 2  (batch size)
    T          = 6  (sequence length)
    max_new    = 4
"""

import math

import torch
import torch.nn as nn

from src.alignment.constitutional_constraints import (
    ConstitutionalConfig,
    ConstitutionalRule,
    ConstitutionalSampler,
    ConstrainedDecoder,
    ConstraintRepairModel,
    LogitConstraintEnforcer,
    TokenConstraintSet,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB = 16
D_MODEL = 16
B = 2
T = 6
MAX_NEW = 4


# ---------------------------------------------------------------------------
# Minimal stub language model for decoding tests
# ---------------------------------------------------------------------------


class _TinyLM(nn.Module):
    """Single-layer embedding + linear head for shape-correctness tests."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]  ->  [B, T, vocab_size]
        x = self.embed(input_ids)
        return self.head(x)


def _make_lm() -> _TinyLM:
    torch.manual_seed(0)
    return _TinyLM(VOCAB, D_MODEL)


def _make_input() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (B, T))


# ---------------------------------------------------------------------------
# 1. ConstitutionalRule — forbidden_tokens: check fails for forbidden token
# ---------------------------------------------------------------------------


def test_rule_forbidden_tokens_check_fails():
    rule = ConstitutionalRule("no_bad", "forbidden_tokens", {"token_ids": [3, 7]})
    tokens_with_forbidden = [1, 2, 3, 4]  # contains 3
    tokens_clean = [1, 2, 4, 5]
    assert rule.check(tokens_with_forbidden) is False
    assert rule.check(tokens_clean) is True


# ---------------------------------------------------------------------------
# 2. ConstitutionalRule — length_bound: check fails for too-long sequence
# ---------------------------------------------------------------------------


def test_rule_length_bound_check_fails_too_long():
    rule = ConstitutionalRule("short", "length_bound", {"max_length": 3})
    long_seq = [0, 1, 2, 3, 4]  # length 5 > max 3
    short_seq = [0, 1, 2]  # length 3 == max 3
    assert rule.check(long_seq) is False
    assert rule.check(short_seq) is True


# ---------------------------------------------------------------------------
# 3. ConstitutionalRule — max_repetition: violation_score > 0 when violated
# ---------------------------------------------------------------------------


def test_rule_max_repetition_violation_score():
    rule = ConstitutionalRule("no_repeat", "max_repetition", {"max_reps": 2})
    # token 5 appears 4 times -> excess = 2
    tokens = [5, 5, 5, 5, 1, 2]
    score = rule.violation_score(tokens)
    assert score > 0.0


# ---------------------------------------------------------------------------
# 4. ConstitutionalRule — max_repetition: zero violation when within limit
# ---------------------------------------------------------------------------


def test_rule_max_repetition_no_violation():
    rule = ConstitutionalRule("no_repeat", "max_repetition", {"max_reps": 3})
    tokens = [5, 5, 5, 1, 2, 3]  # 5 appears exactly 3 times
    assert rule.violation_score(tokens) == 0.0
    assert rule.check(tokens) is True


# ---------------------------------------------------------------------------
# 5. TokenConstraintSet — check_all returns dict with rule names as keys
# ---------------------------------------------------------------------------


def test_constraint_set_check_all_returns_dict():
    r1 = ConstitutionalRule("r1", "forbidden_tokens", {"token_ids": [9]})
    r2 = ConstitutionalRule("r2", "length_bound", {"max_length": 10})
    cset = TokenConstraintSet([r1, r2])
    tokens = [1, 2, 3]
    result = cset.check_all(tokens)
    assert isinstance(result, dict)
    assert "r1" in result
    assert "r2" in result
    assert result["r1"] is True
    assert result["r2"] is True


# ---------------------------------------------------------------------------
# 6. TokenConstraintSet — satisfied returns True when all rules pass
# ---------------------------------------------------------------------------


def test_constraint_set_satisfied_all_pass():
    r1 = ConstitutionalRule("r1", "forbidden_tokens", {"token_ids": [99]})
    r2 = ConstitutionalRule("r2", "length_bound", {"max_length": 5})
    cset = TokenConstraintSet([r1, r2])
    tokens = [0, 1, 2]
    assert cset.satisfied(tokens) is True


# ---------------------------------------------------------------------------
# 7. TokenConstraintSet — satisfied returns False when any rule fails
# ---------------------------------------------------------------------------


def test_constraint_set_satisfied_fails():
    r1 = ConstitutionalRule("r1", "forbidden_tokens", {"token_ids": [2]})
    cset = TokenConstraintSet([r1])
    tokens = [1, 2, 3]  # contains forbidden token 2
    assert cset.satisfied(tokens) is False


# ---------------------------------------------------------------------------
# 8. TokenConstraintSet — total_violation >= 0
# ---------------------------------------------------------------------------


def test_constraint_set_total_violation_nonneg():
    r1 = ConstitutionalRule("r1", "forbidden_tokens", {"token_ids": [3, 7]})
    r2 = ConstitutionalRule("r2", "max_repetition", {"max_reps": 1})
    cset = TokenConstraintSet([r1, r2])
    tokens = [3, 3, 7]  # r1 violation=2, r2 violation=1 -> total=3
    total = cset.total_violation(tokens)
    assert total >= 0.0
    assert total > 0.0  # must actually detect violations here


# ---------------------------------------------------------------------------
# 9. LogitConstraintEnforcer — forward output shape [B, vocab_size]
# ---------------------------------------------------------------------------


def test_enforcer_output_shape():
    enforcer = LogitConstraintEnforcer(vocab_size=VOCAB)
    logits = torch.randn(B, VOCAB)
    generated = torch.randint(0, VOCAB, (B, T))
    out = enforcer(logits, generated)
    assert out.shape == (B, VOCAB)


# ---------------------------------------------------------------------------
# 10. LogitConstraintEnforcer — forbidden tokens set to -inf
# ---------------------------------------------------------------------------


def test_enforcer_forbidden_tokens_neginf():
    enforcer = LogitConstraintEnforcer(vocab_size=VOCAB)
    forbidden_ids = [2, 5, 9]
    enforcer.set_forbidden(forbidden_ids)
    logits = torch.zeros(B, VOCAB)
    generated = torch.zeros(B, T, dtype=torch.long)
    out = enforcer(logits, generated)
    for fid in forbidden_ids:
        assert out[0, fid].item() == float("-inf")
        assert out[1, fid].item() == float("-inf")


# ---------------------------------------------------------------------------
# 11. LogitConstraintEnforcer — non-forbidden tokens remain finite
# ---------------------------------------------------------------------------


def test_enforcer_non_forbidden_finite():
    enforcer = LogitConstraintEnforcer(vocab_size=VOCAB)
    enforcer.set_forbidden([0])
    logits = torch.zeros(B, VOCAB)
    generated = torch.zeros(B, T, dtype=torch.long)
    out = enforcer(logits, generated)
    # Token 1 should still be finite
    assert math.isfinite(out[0, 1].item())


# ---------------------------------------------------------------------------
# 12. LogitConstraintEnforcer — repetition penalty reduces repeated token score
# ---------------------------------------------------------------------------


def test_enforcer_repetition_penalty():
    enforcer = LogitConstraintEnforcer(vocab_size=VOCAB)
    enforcer.set_repetition_penalty(penalty=2.0, max_reps=1)
    # Seed logits: token 3 has positive value
    logits = torch.ones(B, VOCAB) * 1.0
    # generated_so_far: token 3 repeated twice -> penalty applies
    generated = torch.full((B, T), fill_value=3, dtype=torch.long)
    out = enforcer(logits, generated)
    # Token 3 should be penalised (score reduced since positive / 2.0)
    assert out[0, 3].item() < logits[0, 3].item()


# ---------------------------------------------------------------------------
# 13. ConstrainedDecoder — decode output shape [B, T + max_new]
# ---------------------------------------------------------------------------


def test_constrained_decoder_output_shape():
    lm = _make_lm()
    enforcer = LogitConstraintEnforcer(vocab_size=VOCAB)
    decoder = ConstrainedDecoder(lm, enforcer)
    input_ids = _make_input()
    out = decoder.decode(input_ids, max_new=MAX_NEW)
    assert out.shape == (B, T + MAX_NEW)


# ---------------------------------------------------------------------------
# 14. ConstrainedDecoder — forbidden tokens never appear in generated portion
# ---------------------------------------------------------------------------


def test_constrained_decoder_no_forbidden_tokens():
    torch.manual_seed(42)
    lm = _make_lm()
    enforcer = LogitConstraintEnforcer(vocab_size=VOCAB)
    # Forbid tokens 0..7 (first half of vocab)
    forbidden = list(range(8))
    enforcer.set_forbidden(forbidden)
    input_ids = torch.randint(8, VOCAB, (B, T))  # input uses only upper half
    out = ConstrainedDecoder(lm, enforcer).decode(input_ids, max_new=MAX_NEW)
    # Check only the newly generated portion
    generated_portion = out[:, T:]  # [B, MAX_NEW]
    for fid in forbidden:
        assert (generated_portion == fid).sum().item() == 0, (
            f"Forbidden token {fid} appeared in generated output"
        )


# ---------------------------------------------------------------------------
# 15. ConstraintRepairModel — forward output shape [B, T, vocab_size]
# ---------------------------------------------------------------------------


def test_repair_model_forward_shape():
    torch.manual_seed(0)
    model = ConstraintRepairModel(d_model=D_MODEL, vocab_size=VOCAB, n_layers=2)
    input_ids = torch.randint(0, VOCAB, (B, T))
    out = model(input_ids)
    assert out.shape == (B, T, VOCAB)


# ---------------------------------------------------------------------------
# 16. ConstraintRepairModel — repair_loss is finite scalar
# ---------------------------------------------------------------------------


def test_repair_model_repair_loss_finite():
    torch.manual_seed(0)
    model = ConstraintRepairModel(d_model=D_MODEL, vocab_size=VOCAB, n_layers=2)
    input_ids = torch.randint(0, VOCAB, (B, T))
    target_ids = torch.randint(0, VOCAB, (B, T))
    loss = model.repair_loss(input_ids, target_ids)
    assert loss.ndim == 0  # scalar
    assert math.isfinite(loss.item())


# ---------------------------------------------------------------------------
# 17. ConstitutionalSampler — best_of_n returns correct shapes
# ---------------------------------------------------------------------------


def test_sampler_best_of_n_ids_shape():
    lm = _make_lm()
    rules = [ConstitutionalRule("r1", "length_bound", {"max_length": 100})]
    sampler = ConstitutionalSampler(lm, rules, n_candidates=3)
    input_ids = _make_input()
    best_ids, scores = sampler.best_of_n(input_ids, max_new=MAX_NEW)
    assert best_ids.shape == (B, T + MAX_NEW)


# ---------------------------------------------------------------------------
# 18. ConstitutionalSampler — best_of_n scores shape [B]
# ---------------------------------------------------------------------------


def test_sampler_best_of_n_scores_shape():
    lm = _make_lm()
    rules = [ConstitutionalRule("r1", "length_bound", {"max_length": 100})]
    sampler = ConstitutionalSampler(lm, rules, n_candidates=3)
    input_ids = _make_input()
    best_ids, scores = sampler.best_of_n(input_ids, max_new=MAX_NEW)
    assert scores.shape == (B,)


# ---------------------------------------------------------------------------
# 19. ConstitutionalConfig — default values
# ---------------------------------------------------------------------------


def test_constitutional_config_defaults():
    cfg = ConstitutionalConfig()
    assert cfg.vocab_size == 64
    assert cfg.d_model == 32
    assert cfg.n_layers == 2
    assert cfg.forbidden_tokens == []
    assert cfg.max_rep_penalty == 1.5
    assert cfg.max_reps == 3
    assert cfg.n_candidates == 4
    assert cfg.max_attempts == 10


# ---------------------------------------------------------------------------
# 20. ConstitutionalConfig — custom values round-trip
# ---------------------------------------------------------------------------


def test_constitutional_config_custom():
    cfg = ConstitutionalConfig(
        vocab_size=128,
        d_model=64,
        forbidden_tokens=[1, 2, 3],
        max_rep_penalty=2.0,
    )
    assert cfg.vocab_size == 128
    assert cfg.d_model == 64
    assert cfg.forbidden_tokens == [1, 2, 3]
    assert cfg.max_rep_penalty == 2.0


# ---------------------------------------------------------------------------
# 21. ConstitutionalRule — sentiment_bound violation
# ---------------------------------------------------------------------------


def test_rule_sentiment_bound_violation():
    rule = ConstitutionalRule(
        "sentiment",
        "sentiment_bound",
        {"min_positive_ratio": 0.8, "negative_token_ids": [10, 11, 12]},
    )
    # 3 out of 4 tokens are negative -> positive_ratio = 0.25 < 0.8
    tokens = [10, 11, 12, 1]
    assert rule.check(tokens) is False
    assert rule.violation_score(tokens) > 0.0


# ---------------------------------------------------------------------------
# 22. ConstitutionalRule — required_prefix violation
# ---------------------------------------------------------------------------


def test_rule_required_prefix_violation():
    rule = ConstitutionalRule("prefix", "required_prefix", {"prefix": [1, 2, 3]})
    wrong_start = [9, 2, 3, 4]  # first token wrong
    correct_start = [1, 2, 3, 4]
    assert rule.check(wrong_start) is False
    assert rule.check(correct_start) is True


# ---------------------------------------------------------------------------
# 23. ConstrainedDecoder — beam_decode_constrained output shape
# ---------------------------------------------------------------------------


def test_beam_decode_output_shape():
    lm = _make_lm()
    enforcer = LogitConstraintEnforcer(vocab_size=VOCAB)
    decoder = ConstrainedDecoder(lm, enforcer)
    input_ids = _make_input()
    out = decoder.beam_decode_constrained(input_ids, beam_size=2, max_new=MAX_NEW)
    assert out.shape == (B, T + MAX_NEW)
