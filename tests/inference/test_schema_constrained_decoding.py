"""Tests for src/inference/schema_constrained_decoding.py."""

import pytest
import torch

from src.inference.schema_constrained_decoding import (
    FSMState,
    TokenCategory,
    JsonSchemeFSM,
    FSMConstrainedDecoder,
    ConstraintSatisfactionChecker,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB = 20  # Small vocab; categories repeat cleanly (20 tokens, 4 per category)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fsm() -> JsonSchemeFSM:
    return JsonSchemeFSM(vocab_size=VOCAB)


def make_valid_sequence(vocab_size: int = VOCAB) -> torch.Tensor:
    """Build a minimal token sequence that walks the FSM to acceptance.

    Path: open_brace -> string -> colon -> string -> close_brace
    We pick the first token id in each required category.
    """
    cat = TokenCategory(vocab_size)
    tokens = [
        cat.tokens_for_category('open_brace')[0],   # state 0 -> 1
        cat.tokens_for_category('string')[0],        # state 1 -> 2
        cat.tokens_for_category('colon')[0],         # state 2 -> 3
        cat.tokens_for_category('string')[0],        # state 3 -> 4  (value)
        cat.tokens_for_category('close_brace')[0],   # state 4 -> 5
    ]
    return torch.tensor(tokens, dtype=torch.long)


def mock_model_biased(preferred_token: int, vocab_size: int = VOCAB):
    """Return a model_fn that always strongly prefers preferred_token."""
    def _model(ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, ids.shape[1], vocab_size), -100.0)
        logits[:, :, preferred_token] = 100.0
        return logits
    return _model


# ---------------------------------------------------------------------------
# Test 1 — FSMState stores state_id and accepting flag
# ---------------------------------------------------------------------------

def test_fsm_state_stores_fields():
    """FSMState.__init__ should store state_id, accepting, and empty transitions."""
    s = FSMState(state_id=3, accepting=True)
    assert s.state_id == 3
    assert s.accepting is True
    assert s.transitions == {}


# ---------------------------------------------------------------------------
# Test 2 — add_transition updates the transitions dict
# ---------------------------------------------------------------------------

def test_add_transition_updates_dict():
    """add_transition should insert the category->next_state mapping."""
    s = FSMState(0)
    s.add_transition('open_brace', 1)
    s.add_transition('string', 2)
    assert s.transitions['open_brace'] == 1
    assert s.transitions['string'] == 2
    assert len(s.transitions) == 2


# ---------------------------------------------------------------------------
# Test 3 — TokenCategory.category returns one of 5 expected strings
# ---------------------------------------------------------------------------

def test_token_category_returns_valid_category():
    """TokenCategory.category should return one of the 5 defined category strings."""
    valid_cats = {'open_brace', 'close_brace', 'string', 'colon', 'number'}
    tc = TokenCategory(VOCAB)
    for tid in range(VOCAB):
        assert tc.category(tid) in valid_cats, f"token {tid} gave unknown category"


# ---------------------------------------------------------------------------
# Test 4 — tokens_for_category returns correct token ids
# ---------------------------------------------------------------------------

def test_tokens_for_category_correct_ids():
    """tokens_for_category should return ids whose category matches the query."""
    tc = TokenCategory(VOCAB)
    for cat in ('open_brace', 'close_brace', 'string', 'colon', 'number'):
        ids = tc.tokens_for_category(cat)
        assert len(ids) > 0, f"No tokens for category {cat}"
        for tid in ids:
            assert tc.category(tid) == cat, (
                f"token {tid} in tokens_for_category({cat!r}) has wrong category"
            )


# ---------------------------------------------------------------------------
# Test 5 — union of all tokens_for_category covers full vocab
# ---------------------------------------------------------------------------

def test_tokens_for_category_covers_vocab():
    """The union of all category token lists should cover every token id exactly once."""
    tc = TokenCategory(VOCAB)
    all_ids: list[int] = []
    for cat in ('open_brace', 'close_brace', 'string', 'colon', 'number'):
        all_ids.extend(tc.tokens_for_category(cat))
    assert sorted(all_ids) == list(range(VOCAB))


# ---------------------------------------------------------------------------
# Test 6 — JsonSchemeFSM initializes with state 0 as start
# ---------------------------------------------------------------------------

def test_json_schema_fsm_initial_state():
    """JsonSchemeFSM should start at state 0."""
    fsm = make_fsm()
    assert fsm.current_state == 0
    assert 0 in fsm.states


# ---------------------------------------------------------------------------
# Test 7 — valid_categories from state 0 contains only 'open_brace'
# ---------------------------------------------------------------------------

def test_valid_categories_state_0():
    """From state 0 (start), only 'open_brace' should be a valid next category."""
    fsm = make_fsm()
    cats = fsm.valid_categories()
    assert cats == ['open_brace'], f"Expected ['open_brace'], got {cats}"


# ---------------------------------------------------------------------------
# Test 8 — valid_token_mask has correct shape
# ---------------------------------------------------------------------------

def test_valid_token_mask_shape():
    """valid_token_mask should return a bool tensor of shape (vocab_size,)."""
    fsm = make_fsm()
    mask = fsm.valid_token_mask()
    assert mask.shape == (VOCAB,)
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# Test 9 — valid_token_mask is True only for open_brace tokens at state 0
# ---------------------------------------------------------------------------

def test_valid_token_mask_state_0_open_brace_only():
    """At state 0 the mask should be True only for open_brace tokens."""
    fsm = make_fsm()
    mask = fsm.valid_token_mask()
    open_brace_ids = set(fsm.categorizer.tokens_for_category('open_brace'))
    for tid in range(VOCAB):
        if tid in open_brace_ids:
            assert mask[tid].item() is True, f"token {tid} (open_brace) should be True"
        else:
            assert mask[tid].item() is False, f"token {tid} should be False at state 0"


# ---------------------------------------------------------------------------
# Test 10 — transition moves to the expected next state
# ---------------------------------------------------------------------------

def test_transition_advances_state():
    """Transitioning with a valid token should move current_state forward."""
    fsm = make_fsm()
    open_brace_tok = fsm.categorizer.tokens_for_category('open_brace')[0]
    success = fsm.transition(open_brace_tok)
    assert success is True
    assert fsm.current_state == 1  # in_object


# ---------------------------------------------------------------------------
# Test 11 — FSM reaches accepting state after full valid sequence
# ---------------------------------------------------------------------------

def test_fsm_accepts_full_valid_sequence():
    """Replaying a valid token sequence should leave the FSM in an accepting state."""
    fsm = make_fsm()
    seq = make_valid_sequence(VOCAB)
    for tid in seq.tolist():
        ok = fsm.transition(tid)
        assert ok, f"Unexpected FSM failure at token {tid}"
    assert fsm.is_complete() is True


# ---------------------------------------------------------------------------
# Test 12 — reset returns FSM to state 0
# ---------------------------------------------------------------------------

def test_fsm_reset():
    """reset() should restore current_state to 0 regardless of prior traversal."""
    fsm = make_fsm()
    seq = make_valid_sequence(VOCAB)
    for tid in seq.tolist():
        fsm.transition(tid)
    assert fsm.current_state != 0  # sanity check: we advanced
    fsm.reset()
    assert fsm.current_state == 0


# ---------------------------------------------------------------------------
# Test 13 — FSMConstrainedDecoder.apply_fsm_mask sets invalid tokens to -inf
# ---------------------------------------------------------------------------

def test_apply_fsm_mask_sets_invalid_to_neg_inf():
    """apply_fsm_mask should leave valid tokens unchanged and set others to -inf."""
    fsm = make_fsm()
    # Use a trivial model_fn placeholder; we won't call generate here
    decoder = FSMConstrainedDecoder(
        model_fn=lambda ids: torch.zeros(1, ids.shape[1], VOCAB),
        fsm=fsm,
        vocab_size=VOCAB,
    )
    logits = torch.zeros(VOCAB)
    masked = decoder.apply_fsm_mask(logits)

    open_brace_ids = set(fsm.categorizer.tokens_for_category('open_brace'))
    for tid in range(VOCAB):
        if tid in open_brace_ids:
            assert masked[tid].item() != float('-inf'), (
                f"Valid token {tid} should not be -inf"
            )
        else:
            assert masked[tid].item() == float('-inf'), (
                f"Invalid token {tid} should be -inf"
            )


# ---------------------------------------------------------------------------
# Test 14 — ConstraintSatisfactionChecker.check_sequence returns expected keys
# ---------------------------------------------------------------------------

def test_check_sequence_returns_expected_keys():
    """check_sequence should return a dict with 'valid', 'n_valid_transitions',
    and 'failed_at'."""
    fsm = make_fsm()
    checker = ConstraintSatisfactionChecker(fsm)
    seq = make_valid_sequence(VOCAB)
    result = checker.check_sequence(seq)
    assert 'valid' in result
    assert 'n_valid_transitions' in result
    assert 'failed_at' in result


# ---------------------------------------------------------------------------
# Test 15 — check_sequence returns valid=True for a well-formed sequence
# ---------------------------------------------------------------------------

def test_check_sequence_valid_for_well_formed():
    """check_sequence should report valid=True for a sequence that walks the FSM
    to an accepting state."""
    fsm = make_fsm()
    checker = ConstraintSatisfactionChecker(fsm)
    seq = make_valid_sequence(VOCAB)
    result = checker.check_sequence(seq)
    assert result['valid'] is True
    assert result['failed_at'] is None
    assert result['n_valid_transitions'] == len(seq)
