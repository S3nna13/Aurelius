"""Tests for constrained generation module."""

import math
import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.constrained_generation import (
    ConstraintConfig,
    apply_repetition_penalty,
    build_vocab_mask,
    apply_vocab_mask,
    PrefixConstrainedDecoder,
    RegexConstrainedDecoder,
    BannedTokensDecoder,
)

VOCAB_SIZE = 256
TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=VOCAB_SIZE,
    max_seq_len=512,
)


@pytest.fixture
def small_model():
    torch.manual_seed(42)
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


def token_decode(ids: list) -> str:
    return bytes([t % 256 for t in ids]).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 1. ConstraintConfig defaults
# ---------------------------------------------------------------------------

def test_constraint_config_defaults():
    cfg = ConstraintConfig()
    assert cfg.max_new_tokens == 64
    assert cfg.temperature == 1.0
    assert cfg.top_p == 1.0
    assert cfg.repetition_penalty == 1.0


# ---------------------------------------------------------------------------
# 2. apply_repetition_penalty — output shape (V,)
# ---------------------------------------------------------------------------

def test_apply_repetition_penalty_shape():
    logits = torch.randn(VOCAB_SIZE)
    result = apply_repetition_penalty(logits, [1, 2, 3], penalty=1.5)
    assert result.shape == (VOCAB_SIZE,)


# ---------------------------------------------------------------------------
# 3. apply_repetition_penalty — penalized token logit is smaller (positive logit case)
# ---------------------------------------------------------------------------

def test_apply_repetition_penalty_reduces_logit():
    logits = torch.zeros(VOCAB_SIZE)
    token_id = 10
    logits[token_id] = 2.0  # positive logit
    result = apply_repetition_penalty(logits, [token_id], penalty=2.0)
    # logit / penalty => 2.0 / 2.0 = 1.0, which is less than 2.0
    assert result[token_id].item() < logits[token_id].item()


# ---------------------------------------------------------------------------
# 4. build_vocab_mask — True for allowed tokens
# ---------------------------------------------------------------------------

def test_build_vocab_mask_allowed():
    allowed = [0, 5, 100, 255]
    mask = build_vocab_mask(allowed, VOCAB_SIZE)
    for tok in allowed:
        assert mask[tok].item() is True


# ---------------------------------------------------------------------------
# 5. build_vocab_mask — False for disallowed tokens
# ---------------------------------------------------------------------------

def test_build_vocab_mask_disallowed():
    allowed = [0, 5, 100]
    mask = build_vocab_mask(allowed, VOCAB_SIZE)
    for tok in range(VOCAB_SIZE):
        if tok not in allowed:
            assert mask[tok].item() is False


# ---------------------------------------------------------------------------
# 6. apply_vocab_mask — disallowed tokens become -inf
# ---------------------------------------------------------------------------

def test_apply_vocab_mask_inf():
    logits = torch.zeros(VOCAB_SIZE)
    allowed = [1, 2, 3]
    mask = build_vocab_mask(allowed, VOCAB_SIZE)
    result = apply_vocab_mask(logits, mask)
    for tok in range(VOCAB_SIZE):
        if tok not in allowed:
            assert result[tok].item() == float("-inf")
    for tok in allowed:
        assert math.isfinite(result[tok].item())


# ---------------------------------------------------------------------------
# 7. PrefixConstrainedDecoder — first n tokens match required_prefix
# ---------------------------------------------------------------------------

def test_prefix_constrained_decoder_prefix_enforced(small_model):
    required_prefix = [10, 20, 30]
    config = ConstraintConfig(max_new_tokens=6, temperature=1.0, top_p=1.0)
    decoder = PrefixConstrainedDecoder(small_model, required_prefix, config)

    prompt_len = 3
    input_ids = torch.randint(0, VOCAB_SIZE, (1, prompt_len))
    output = decoder.generate(input_ids)

    # The generated portion starts at index prompt_len
    generated = output[0, prompt_len:].tolist()
    for i, expected_tok in enumerate(required_prefix):
        assert generated[i] == expected_tok, (
            f"Step {i}: expected prefix token {expected_tok}, got {generated[i]}"
        )


# ---------------------------------------------------------------------------
# 8. PrefixConstrainedDecoder — output shape
# ---------------------------------------------------------------------------

def test_prefix_constrained_decoder_output_shape(small_model):
    required_prefix = [5, 15]
    max_new = 8
    config = ConstraintConfig(max_new_tokens=max_new)
    decoder = PrefixConstrainedDecoder(small_model, required_prefix, config)

    prompt_len = 4
    input_ids = torch.randint(0, VOCAB_SIZE, (1, prompt_len))
    output = decoder.generate(input_ids)

    assert output.shape == (1, prompt_len + max_new)


# ---------------------------------------------------------------------------
# 9. BannedTokensDecoder — generated tokens don't include banned tokens
# ---------------------------------------------------------------------------

def test_banned_tokens_decoder_no_banned_in_output(small_model):
    # Ban a large range of tokens so there's a meaningful constraint
    banned = list(range(50, 200))
    config = ConstraintConfig(max_new_tokens=10)
    decoder = BannedTokensDecoder(small_model, banned, config)

    prompt_len = 3
    input_ids = torch.randint(0, 50, (1, prompt_len))  # prompt uses non-banned tokens
    output = decoder.generate(input_ids)

    generated_tokens = output[0, prompt_len:].tolist()
    for tok in generated_tokens:
        assert tok not in banned, f"Banned token {tok} appeared in output"


# ---------------------------------------------------------------------------
# 10. BannedTokensDecoder — output shape
# ---------------------------------------------------------------------------

def test_banned_tokens_decoder_output_shape(small_model):
    banned = [1, 2, 3]
    max_new = 7
    config = ConstraintConfig(max_new_tokens=max_new)
    decoder = BannedTokensDecoder(small_model, banned, config)

    prompt_len = 5
    input_ids = torch.randint(0, VOCAB_SIZE, (1, prompt_len))
    output = decoder.generate(input_ids)

    assert output.shape == (1, prompt_len + max_new)


# ---------------------------------------------------------------------------
# 11. RegexConstrainedDecoder — output shape
# ---------------------------------------------------------------------------

def test_regex_constrained_decoder_output_shape(small_model):
    # Simple pattern: allow any sequence of digits or letters
    pattern = r"[a-zA-Z0-9 ]*"
    max_new = 5
    config = ConstraintConfig(max_new_tokens=max_new)
    decoder = RegexConstrainedDecoder(small_model, pattern, token_decode, config)

    prompt_len = 3
    input_ids = torch.randint(0, VOCAB_SIZE, (1, prompt_len))
    output = decoder.generate(input_ids, prompt_text="")

    assert output.shape == (1, prompt_len + max_new)
