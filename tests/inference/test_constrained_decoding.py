"""Tests for src/inference/constrained_decoding.py — lexical constraint automaton."""

import pytest
import torch

from src.inference.constrained_decoding import (
    ConstraintConfig,
    ConstraintState,
    ConstraintBank,
    apply_constraint_mask,
    ConstrainedDecoder,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
CONSTRAINT_TOKEN = 5  # the test constraint token id


# ---------------------------------------------------------------------------
# Fake model fixture
# ---------------------------------------------------------------------------

class FakeModel:
    """Returns uniform logits so greedy picks token 0 unless constraint overrides."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, bias_token: int = 0):
        self.vocab_size = vocab_size
        self.bias_token = bias_token

    def __call__(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size)
        logits[:, :, self.bias_token] = 1.0  # greedy would pick bias_token
        loss = torch.tensor(0.0)
        pkv = None
        return loss, logits, pkv


# ---------------------------------------------------------------------------
# ConstraintState tests
# ---------------------------------------------------------------------------

def test_constraint_state_advance_returns_true_on_completion():
    """advance() returns True when the last token completes the constraint."""
    state = ConstraintState([CONSTRAINT_TOKEN])
    result = state.advance(CONSTRAINT_TOKEN)
    assert result is True


def test_constraint_state_completed_after_full_sequence():
    """completed is True after all tokens in the sequence are matched."""
    state = ConstraintState([CONSTRAINT_TOKEN, 10])
    state.advance(CONSTRAINT_TOKEN)
    assert not state.completed
    state.advance(10)
    assert state.completed


def test_constraint_state_next_tokens_returns_correct_next():
    """next_tokens returns the single next expected token."""
    state = ConstraintState([CONSTRAINT_TOKEN, 10])
    assert state.next_tokens == [CONSTRAINT_TOKEN]
    state.advance(CONSTRAINT_TOKEN)
    assert state.next_tokens == [10]


def test_constraint_state_next_tokens_empty_when_completed():
    """next_tokens is empty once the constraint is fully satisfied."""
    state = ConstraintState([CONSTRAINT_TOKEN])
    state.advance(CONSTRAINT_TOKEN)
    assert state.next_tokens == []


def test_constraint_state_reset_returns_to_start():
    """reset() resets the pointer so the constraint can be matched again."""
    state = ConstraintState([CONSTRAINT_TOKEN, 10])
    state.advance(CONSTRAINT_TOKEN)
    assert state.next_tokens == [10]
    state.reset()
    assert state.next_tokens == [CONSTRAINT_TOKEN]
    assert not state.completed


def test_constraint_state_advance_no_match_returns_false():
    """advance() returns False when the token does not match."""
    state = ConstraintState([CONSTRAINT_TOKEN])
    result = state.advance(99)  # wrong token
    assert result is False
    assert not state.completed


# ---------------------------------------------------------------------------
# ConstraintBank tests
# ---------------------------------------------------------------------------

def test_constraint_bank_all_completed_when_all_done():
    """all_completed is True when every constraint has been satisfied."""
    bank = ConstraintBank([[CONSTRAINT_TOKEN], [10]])
    bank.advance(CONSTRAINT_TOKEN)
    assert not bank.all_completed
    bank.advance(10)
    assert bank.all_completed


def test_constraint_bank_get_active_next_tokens_union():
    """get_active_next_tokens returns the union of next_tokens from all pending constraints."""
    bank = ConstraintBank([[CONSTRAINT_TOKEN], [10]])
    active = bank.get_active_next_tokens()
    assert CONSTRAINT_TOKEN in active
    assert 10 in active


def test_constraint_bank_advance_updates_states():
    """advance() propagates the token to all internal ConstraintState objects."""
    bank = ConstraintBank([[CONSTRAINT_TOKEN], [10]])
    bank.advance(CONSTRAINT_TOKEN)
    # token 5 constraint should be done; token 10 still pending
    active = bank.get_active_next_tokens()
    assert CONSTRAINT_TOKEN not in active
    assert 10 in active


def test_constraint_bank_empty_constraints_all_completed():
    """A bank with no constraints is immediately all_completed."""
    bank = ConstraintBank([])
    assert bank.all_completed


# ---------------------------------------------------------------------------
# apply_constraint_mask tests
# ---------------------------------------------------------------------------

def test_apply_constraint_mask_returns_same_shape():
    """Output tensor has the same shape as the input logits."""
    logits = torch.zeros(VOCAB_SIZE)
    bank = ConstraintBank([[CONSTRAINT_TOKEN]])
    out = apply_constraint_mask(logits, bank)
    assert out.shape == logits.shape


def test_apply_constraint_mask_boosts_required_token():
    """Required token logit is NOT penalised; all others are penalised."""
    logits = torch.zeros(VOCAB_SIZE)
    bank = ConstraintBank([[CONSTRAINT_TOKEN]])
    out = apply_constraint_mask(logits, bank, penalty=-1e9)
    # Required token keeps its value (0 + 0 = 0)
    assert out[CONSTRAINT_TOKEN].item() == pytest.approx(0.0)
    # A non-required token should be heavily penalised
    assert out[0].item() == pytest.approx(-1e9)


def test_apply_constraint_mask_unchanged_when_no_active_constraints():
    """Logits are returned unchanged when all constraints are already met."""
    bank = ConstraintBank([[CONSTRAINT_TOKEN]])
    bank.advance(CONSTRAINT_TOKEN)  # satisfy the constraint
    logits = torch.ones(VOCAB_SIZE)
    out = apply_constraint_mask(logits, bank)
    assert torch.equal(out, logits)


def test_apply_constraint_mask_2d_logits():
    """apply_constraint_mask works on (B, V) shaped logits."""
    logits = torch.zeros(2, VOCAB_SIZE)
    bank = ConstraintBank([[CONSTRAINT_TOKEN]])
    out = apply_constraint_mask(logits, bank)
    assert out.shape == (2, VOCAB_SIZE)
    # required token: no penalty in either row
    assert out[0, CONSTRAINT_TOKEN].item() == pytest.approx(0.0)
    assert out[1, CONSTRAINT_TOKEN].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ConstrainedDecoder tests
# ---------------------------------------------------------------------------

def _make_input_ids(seq_len: int = 4) -> torch.Tensor:
    return torch.zeros(1, seq_len, dtype=torch.long)


def test_constrained_decoder_generate_returns_tuple():
    """generate() returns a (Tensor, bool) tuple."""
    model = FakeModel(bias_token=0)
    cfg = ConstraintConfig(constraints=[[CONSTRAINT_TOKEN]])
    decoder = ConstrainedDecoder(model, cfg)
    result = decoder.generate(_make_input_ids(), max_new_tokens=4)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], bool)


def test_constrained_decoder_generate_output_shape():
    """generated_ids has shape (1, max_new_tokens)."""
    model = FakeModel(bias_token=0)
    cfg = ConstraintConfig(constraints=[])  # no constraints, fast path
    decoder = ConstrainedDecoder(model, cfg)
    generated_ids, _ = decoder.generate(_make_input_ids(), max_new_tokens=4)
    assert generated_ids.shape == (1, 4)


def test_constrained_decoder_single_token_constraint_appears_in_output():
    """With a single-token constraint, that token must appear in the generated output."""
    # Model biases toward token 0, but constraint forces token CONSTRAINT_TOKEN
    model = FakeModel(bias_token=0)
    cfg = ConstraintConfig(constraints=[[CONSTRAINT_TOKEN]])
    decoder = ConstrainedDecoder(model, cfg)
    generated_ids, all_met = decoder.generate(_make_input_ids(), max_new_tokens=4)
    token_list = generated_ids[0].tolist()
    assert CONSTRAINT_TOKEN in token_list
    assert all_met is True


def test_constrained_decoder_empty_constraints_all_met():
    """With no constraints, all_constraints_met is always True."""
    model = FakeModel(bias_token=0)
    cfg = ConstraintConfig(constraints=[])
    decoder = ConstrainedDecoder(model, cfg)
    _, all_met = decoder.generate(_make_input_ids(), max_new_tokens=4)
    assert all_met is True
