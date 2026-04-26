"""Tests for src/inference/format_enforcer.py — token-level logit masking."""

import pytest
import torch

from src.inference.format_enforcer import (
    ConstrainedGenerator,
    FormatEnforcer,
    FormatSpec,
    JsonStateMachine,
    TokenMask,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256


def dummy_tokenizer_encode(text: str) -> list[int]:
    """Encode text as UTF-8 byte values (IDs = bytes)."""
    return list(text.encode("utf-8"))


def dummy_tokenizer_decode(token_ids: list[int]) -> str:
    """Decode byte values back to string, ignoring non-decodeable bytes."""
    return bytes(b for b in token_ids if 0 <= b < 256).decode("utf-8", errors="replace")


class FakeModel:
    """A fake model that returns a fixed logits tensor."""

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size

    def __call__(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        # Return uniform logits — shape (B, T, V)
        logits = torch.zeros(B, T, self.vocab_size)
        # Bias token 65 ('A') slightly so greedy picks it
        logits[:, :, 65] = 1.0
        loss = torch.tensor(0.0)
        pkv = None
        return loss, logits, pkv


# ---------------------------------------------------------------------------
# 1. FormatSpec fields
# ---------------------------------------------------------------------------


def test_format_spec_fields():
    spec = FormatSpec(
        format_type="json",
        required_prefix='{"key":',
        required_suffix="}",
        max_length=128,
        allowed_chars="",
    )
    assert spec.format_type == "json"
    assert spec.required_prefix == '{"key":'
    assert spec.required_suffix == "}"
    assert spec.max_length == 128
    assert spec.allowed_chars == ""


def test_format_spec_defaults():
    spec = FormatSpec(format_type="free")
    assert spec.required_prefix == ""
    assert spec.required_suffix == ""
    assert spec.max_length == 256
    assert spec.allowed_chars == ""


# ---------------------------------------------------------------------------
# 2. TokenMask.allow_all
# ---------------------------------------------------------------------------


def test_token_mask_allow_all_permits_every_token():
    mask = TokenMask(VOCAB_SIZE)
    # Block some first
    mask.block([0, 1, 2])
    mask.allow_all()
    bias = mask.to_logit_bias()
    assert (bias == 0).all(), "allow_all should result in zero bias for every token"


# ---------------------------------------------------------------------------
# 3. TokenMask.allow_only
# ---------------------------------------------------------------------------


def test_token_mask_allow_only_blocks_others():
    mask = TokenMask(VOCAB_SIZE)
    allowed = [65, 66, 67]  # 'A', 'B', 'C'
    mask.allow_only(allowed)
    bias = mask.to_logit_bias()
    # Allowed tokens should have bias 0
    for tid in allowed:
        assert bias[tid].item() == 0.0
    # All others should be blocked
    for tid in range(VOCAB_SIZE):
        if tid not in allowed:
            assert bias[tid].item() < 0


# ---------------------------------------------------------------------------
# 4. TokenMask.block
# ---------------------------------------------------------------------------


def test_token_mask_block_blocks_specified():
    mask = TokenMask(VOCAB_SIZE)
    mask.allow_all()
    blocked = [10, 20, 30]
    mask.block(blocked)
    bias = mask.to_logit_bias()
    for tid in blocked:
        assert bias[tid].item() < 0
    # Unblocked tokens should still be 0
    assert bias[0].item() == 0.0  # 0 was not blocked


# ---------------------------------------------------------------------------
# 5. TokenMask.to_logit_bias shape
# ---------------------------------------------------------------------------


def test_token_mask_to_logit_bias_shape():
    mask = TokenMask(VOCAB_SIZE)
    bias = mask.to_logit_bias()
    assert bias.shape == (VOCAB_SIZE,)


# ---------------------------------------------------------------------------
# 6. TokenMask.to_logit_bias blocked tokens get -1e9
# ---------------------------------------------------------------------------


def test_token_mask_to_logit_bias_blocked_value():
    mask = TokenMask(VOCAB_SIZE)
    mask.allow_only([42])
    bias = mask.to_logit_bias(blocked_value=-1e9)
    # Token 0 should be blocked
    assert bias[0].item() == pytest.approx(-1e9)
    # Token 42 should be 0
    assert bias[42].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. TokenMask.apply_to_logits shape preserved
# ---------------------------------------------------------------------------


def test_token_mask_apply_to_logits_shape_1d():
    mask = TokenMask(VOCAB_SIZE)
    logits = torch.zeros(VOCAB_SIZE)
    result = mask.apply_to_logits(logits)
    assert result.shape == (VOCAB_SIZE,)


def test_token_mask_apply_to_logits_shape_2d():
    mask = TokenMask(VOCAB_SIZE)
    B = 3
    logits = torch.zeros(B, VOCAB_SIZE)
    result = mask.apply_to_logits(logits)
    assert result.shape == (B, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# 8. TokenMask.apply_to_logits blocked tokens get very low value
# ---------------------------------------------------------------------------


def test_token_mask_apply_to_logits_blocked_tokens_very_low():
    mask = TokenMask(VOCAB_SIZE)
    mask.allow_only([65])  # only 'A'
    logits = torch.zeros(VOCAB_SIZE)
    result = mask.apply_to_logits(logits)
    # All tokens except 65 should be very negative
    assert result[0].item() < -1e8
    assert result[65].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. JsonStateMachine.update changes state
# ---------------------------------------------------------------------------


def test_json_state_machine_update_changes_state():
    sm = JsonStateMachine()
    assert sm._state == "start"
    sm.update(ord("{"))
    assert sm._state != "start"


def test_json_state_machine_done_after_close():
    sm = JsonStateMachine()
    sm.update(ord("{"))
    sm.update(ord("}"))
    assert sm._state == "done"


# ---------------------------------------------------------------------------
# 10. JsonStateMachine.allowed_next_bytes returns list of ints
# ---------------------------------------------------------------------------


def test_json_state_machine_allowed_next_bytes_returns_list():
    sm = JsonStateMachine()
    result = sm.allowed_next_bytes()
    assert isinstance(result, list)
    assert all(isinstance(b, int) for b in result)


def test_json_state_machine_start_only_allows_open_brace():
    sm = JsonStateMachine()
    allowed = sm.allowed_next_bytes()
    assert allowed == [ord("{")]


def test_json_state_machine_done_allows_nothing():
    sm = JsonStateMachine()
    sm.update(ord("{"))
    sm.update(ord("}"))
    assert sm._state == "done"
    assert sm.allowed_next_bytes() == []


# ---------------------------------------------------------------------------
# 11. FormatEnforcer.get_allowed_tokens returns TokenMask
# ---------------------------------------------------------------------------


def test_format_enforcer_get_allowed_tokens_returns_token_mask():
    spec = FormatSpec(format_type="free")
    enforcer = FormatEnforcer(spec, dummy_tokenizer_encode, VOCAB_SIZE)
    result = enforcer.get_allowed_tokens([])
    assert isinstance(result, TokenMask)


# ---------------------------------------------------------------------------
# 12. FormatEnforcer.is_complete with suffix present
# ---------------------------------------------------------------------------


def test_format_enforcer_is_complete_with_suffix():
    spec = FormatSpec(format_type="json", required_suffix="}")
    enforcer = FormatEnforcer(spec, dummy_tokenizer_encode, VOCAB_SIZE)
    # Generate something ending with '}'
    generated = list(b'{"key": "val"}')
    assert enforcer.is_complete(generated) is True


def test_format_enforcer_is_complete_without_suffix():
    spec = FormatSpec(format_type="json", required_suffix="}")
    enforcer = FormatEnforcer(spec, dummy_tokenizer_encode, VOCAB_SIZE)
    generated = list(b'{"key": "val"')  # missing closing brace
    assert enforcer.is_complete(generated) is False


def test_format_enforcer_is_complete_no_suffix_always_true():
    spec = FormatSpec(format_type="free")
    enforcer = FormatEnforcer(spec, dummy_tokenizer_encode, VOCAB_SIZE)
    assert enforcer.is_complete([]) is True
    assert enforcer.is_complete([65, 66]) is True


# ---------------------------------------------------------------------------
# 13. ConstrainedGenerator.generate returns string
# ---------------------------------------------------------------------------


def test_constrained_generator_generate_returns_string():
    model = FakeModel(VOCAB_SIZE)
    spec = FormatSpec(format_type="free", max_length=16)
    gen = ConstrainedGenerator(
        model, spec, dummy_tokenizer_encode, dummy_tokenizer_decode, VOCAB_SIZE
    )
    result = gen.generate("hello", max_new_tokens=4)
    assert isinstance(result, str)


def test_constrained_generator_generate_respects_max_new_tokens():
    model = FakeModel(VOCAB_SIZE)
    spec = FormatSpec(format_type="free", max_length=256)
    gen = ConstrainedGenerator(
        model, spec, dummy_tokenizer_encode, dummy_tokenizer_decode, VOCAB_SIZE
    )
    result = gen.generate("hi", max_new_tokens=5)
    # Should produce at most 5 new tokens
    assert len(result.encode("utf-8")) <= 5


# ---------------------------------------------------------------------------
# 14. FormatEnforcer.enforce_prefix returns logits same shape
# ---------------------------------------------------------------------------


def test_format_enforcer_enforce_prefix_shape_1d():
    spec = FormatSpec(format_type="free")
    enforcer = FormatEnforcer(spec, dummy_tokenizer_encode, VOCAB_SIZE)
    logits = torch.zeros(VOCAB_SIZE)
    result = enforcer.enforce_prefix(logits, [])
    assert result.shape == (VOCAB_SIZE,)


def test_format_enforcer_enforce_prefix_shape_2d():
    spec = FormatSpec(format_type="free")
    enforcer = FormatEnforcer(spec, dummy_tokenizer_encode, VOCAB_SIZE)
    B = 2
    logits = torch.zeros(B, VOCAB_SIZE)
    result = enforcer.enforce_prefix(logits, [])
    assert result.shape == (B, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# 15. FormatEnforcer prefix enforcement forces first tokens
# ---------------------------------------------------------------------------


def test_format_enforcer_prefix_forces_first_token():
    """With required_prefix='AB', position 0 should only allow ord('A')=65."""
    spec = FormatSpec(format_type="free", required_prefix="AB")
    enforcer = FormatEnforcer(spec, dummy_tokenizer_encode, VOCAB_SIZE)
    logits = torch.zeros(VOCAB_SIZE)
    result = enforcer.enforce_prefix(logits, [])
    # Only token 65 ('A') should be non-negative
    assert result[65].item() == pytest.approx(0.0)
    # All other tokens must be blocked (very negative)
    assert result[66].item() < -1e8
    assert result[0].item() < -1e8


def test_format_enforcer_prefix_forces_second_token():
    """After consuming 'A' (65), should force 'B' (66)."""
    spec = FormatSpec(format_type="free", required_prefix="AB")
    enforcer = FormatEnforcer(spec, dummy_tokenizer_encode, VOCAB_SIZE)
    logits = torch.zeros(VOCAB_SIZE)
    result = enforcer.enforce_prefix(logits, [65])  # already generated 'A'
    assert result[66].item() == pytest.approx(0.0)  # 'B' allowed
    assert result[65].item() < -1e8  # 'A' blocked (not the next prefix byte)


def test_format_enforcer_free_after_prefix():
    """After the full prefix is consumed, all tokens should be allowed (bias=0)."""
    spec = FormatSpec(format_type="free", required_prefix="A")
    enforcer = FormatEnforcer(spec, dummy_tokenizer_encode, VOCAB_SIZE)
    logits = torch.zeros(VOCAB_SIZE)
    result = enforcer.enforce_prefix(logits, [65])  # prefix 'A' consumed
    # No restrictions; all should be 0
    assert (result == 0).all()
