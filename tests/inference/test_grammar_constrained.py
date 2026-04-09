"""Tests for grammar-constrained generation module."""

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.grammar_constrained import (
    GrammarConfig,
    RegexFSM,
    json_schema_mask,
    apply_grammar_mask,
    validate_against_pattern,
    ConstrainedDecoder,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256


@pytest.fixture(scope="module")
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def small_model(small_cfg):
    torch.manual_seed(0)
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


def _encode(text: str) -> list[int]:
    """Simple character-level encode: ord(c) % VOCAB_SIZE."""
    return [ord(c) % VOCAB_SIZE for c in text] if text else [0]


def _decode(token_id: int) -> str:
    """Simple decode: chr of token_id if printable, else empty string."""
    if 32 <= token_id < 127:
        return chr(token_id)
    return ""


# ---------------------------------------------------------------------------
# GrammarConfig tests
# ---------------------------------------------------------------------------

def test_grammar_config_defaults():
    cfg = GrammarConfig()
    assert cfg.max_new_tokens == 64
    assert cfg.grammar_type == "regex"
    assert cfg.pattern == ""
    assert cfg.allow_any_on_fail is True


def test_grammar_config_custom():
    cfg = GrammarConfig(max_new_tokens=10, grammar_type="json_schema", pattern=r"\d+", allow_any_on_fail=False)
    assert cfg.max_new_tokens == 10
    assert cfg.grammar_type == "json_schema"
    assert cfg.pattern == r"\d+"
    assert cfg.allow_any_on_fail is False


# ---------------------------------------------------------------------------
# RegexFSM tests
# ---------------------------------------------------------------------------

def test_regex_fsm_allowed_tokens_shape():
    fsm = RegexFSM(r"\d+", VOCAB_SIZE)
    mask = fsm.get_allowed_tokens(VOCAB_SIZE, _decode)
    assert mask.shape == (VOCAB_SIZE,)


def test_regex_fsm_advance_updates_text():
    fsm = RegexFSM(r".*", VOCAB_SIZE)
    assert fsm.current_text == ""
    # token_id for '5' is 53 (ord('5'))
    fsm.advance(53, _decode)
    assert fsm.current_text == "5"


def test_regex_fsm_digit_pattern_allows_digits():
    """For pattern r'\\d+', digit tokens should be allowed."""
    fsm = RegexFSM(r"\d+", VOCAB_SIZE)
    mask = fsm.get_allowed_tokens(VOCAB_SIZE, _decode)
    # Digits '0'..'9' are token_ids 48..57
    for digit_id in range(48, 58):
        assert mask[digit_id] == 1.0, f"Token {digit_id} ('{chr(digit_id)}') should be allowed"


def test_regex_fsm_digit_pattern_blocks_letters():
    """For pattern r'\\d+', letter tokens should be blocked."""
    fsm = RegexFSM(r"\d+", VOCAB_SIZE)
    mask = fsm.get_allowed_tokens(VOCAB_SIZE, _decode)
    # Letter 'A' is token_id 65
    assert mask[65] == 0.0, "Token 65 ('A') should be blocked by digit-only pattern"


def test_regex_fsm_advance_accumulates_text():
    fsm = RegexFSM(r"\d+", VOCAB_SIZE)
    fsm.advance(49, _decode)  # '1'
    fsm.advance(50, _decode)  # '2'
    assert fsm.current_text == "12"


# ---------------------------------------------------------------------------
# apply_grammar_mask tests
# ---------------------------------------------------------------------------

def test_apply_grammar_mask_blocks_zeros():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    result = apply_grammar_mask(logits, mask)
    assert result[1] == float("-inf")
    assert result[3] == float("-inf")


def test_apply_grammar_mask_keeps_allowed():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    result = apply_grammar_mask(logits, mask)
    assert result[0] == 1.0
    assert result[2] == 3.0


def test_apply_grammar_mask_does_not_modify_input():
    logits = torch.tensor([1.0, 2.0, 3.0])
    original = logits.clone()
    mask = torch.tensor([1.0, 0.0, 1.0])
    apply_grammar_mask(logits, mask)
    assert torch.equal(logits, original)


# ---------------------------------------------------------------------------
# json_schema_mask tests
# ---------------------------------------------------------------------------

def test_json_schema_mask_shape():
    mask = json_schema_mask({}, "", VOCAB_SIZE, _decode)
    assert mask.shape == (VOCAB_SIZE,)


def test_json_schema_mask_allows_brace():
    mask = json_schema_mask({}, "", VOCAB_SIZE, _decode)
    # '{' is token_id 123
    assert mask[123] == 1.0


def test_json_schema_mask_returns_tensor():
    mask = json_schema_mask({}, "{", VOCAB_SIZE, _decode)
    assert isinstance(mask, torch.Tensor)


# ---------------------------------------------------------------------------
# validate_against_pattern tests
# ---------------------------------------------------------------------------

def test_validate_against_pattern_match():
    assert validate_against_pattern("12345", r"\d+") is True


def test_validate_against_pattern_no_match():
    assert validate_against_pattern("hello", r"\d+") is False


def test_validate_against_pattern_partial_does_not_match():
    # Full match required -- partial prefix should return False
    assert validate_against_pattern("123abc", r"\d+") is False


def test_validate_against_pattern_empty():
    assert validate_against_pattern("", r"\d*") is True


# ---------------------------------------------------------------------------
# ConstrainedDecoder tests
# ---------------------------------------------------------------------------

def test_constrained_decoder_free_returns_string(small_model):
    cfg = GrammarConfig(max_new_tokens=4, grammar_type="free")
    decoder = ConstrainedDecoder(small_model, _encode, _decode, cfg)
    text, stats = decoder.generate("hi")
    assert isinstance(text, str)


def test_constrained_decoder_stats_keys(small_model):
    cfg = GrammarConfig(max_new_tokens=4, grammar_type="free")
    decoder = ConstrainedDecoder(small_model, _encode, _decode, cfg)
    _, stats = decoder.generate("hi")
    assert "n_tokens" in stats
    assert "n_constrained_steps" in stats


def test_constrained_decoder_stats_n_tokens(small_model):
    cfg = GrammarConfig(max_new_tokens=4, grammar_type="free")
    decoder = ConstrainedDecoder(small_model, _encode, _decode, cfg)
    _, stats = decoder.generate("hi")
    assert stats["n_tokens"] == 4


def test_constrained_decoder_regex_returns_string(small_model):
    cfg = GrammarConfig(max_new_tokens=4, grammar_type="regex", pattern=r".*", allow_any_on_fail=True)
    decoder = ConstrainedDecoder(small_model, _encode, _decode, cfg)
    text, stats = decoder.generate("test")
    assert isinstance(text, str)
    assert stats["n_tokens"] == 4


def test_constrained_decoder_json_schema_mode(small_model):
    cfg = GrammarConfig(max_new_tokens=4, grammar_type="json_schema", allow_any_on_fail=True)
    decoder = ConstrainedDecoder(small_model, _encode, _decode, cfg)
    text, stats = decoder.generate("start")
    assert isinstance(text, str)
    assert "n_tokens" in stats
    assert "n_constrained_steps" in stats
