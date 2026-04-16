"""Tests for src/inference/constrained_decoding.py."""

import pytest
import torch

from src.inference.constrained_decoding import (
    ConstraintConfig,
    apply_token_constraints,
    apply_min_length_constraint,
    force_prefix,
    LogitProcessor,
    ConstrainedGreedyDecoder,
    ConstrainedSampler,
    compute_constraint_satisfaction,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB = 32
PROMPT_LEN = 4
B = 1
EOS = 2


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------

def mock_model(ids: torch.Tensor) -> torch.Tensor:
    """Return random logits of shape (1, T, VOCAB)."""
    return torch.randn(1, ids.shape[1], VOCAB)


def biased_mock_model(bias_token: int):
    """Return a mock model that always strongly prefers bias_token."""
    def _model(ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, ids.shape[1], VOCAB), -100.0)
        logits[:, :, bias_token] = 100.0
        return logits
    return _model


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_prompt(length: int = PROMPT_LEN) -> torch.Tensor:
    return torch.zeros(length, dtype=torch.long)


# ---------------------------------------------------------------------------
# Test 1: ConstraintConfig default values
# ---------------------------------------------------------------------------

def test_config_defaults():
    """ConstraintConfig should have correct default field values."""
    cfg = ConstraintConfig()
    assert cfg.allowed_tokens is None
    assert cfg.banned_tokens is None
    assert cfg.min_new_tokens == 0
    assert cfg.max_new_tokens == 100
    assert cfg.force_eos_token is None
    assert cfg.prefix_tokens is None


# ---------------------------------------------------------------------------
# Test 2: apply_token_constraints — banned makes logit -inf
# ---------------------------------------------------------------------------

def test_apply_token_constraints_banned_sets_neg_inf():
    """Banned tokens must have -inf logit after apply_token_constraints."""
    logits = torch.zeros(VOCAB)
    banned = [5, 10]
    out = apply_token_constraints(logits, banned=banned)
    for tok in banned:
        assert out[tok].item() == float("-inf"), f"Token {tok} should be -inf"
    # Non-banned tokens should remain 0
    assert out[0].item() == 0.0


# ---------------------------------------------------------------------------
# Test 3: apply_token_constraints — allowed masks all non-allowed tokens
# ---------------------------------------------------------------------------

def test_apply_token_constraints_allowed_masks_others():
    """Only allowed tokens should keep their original logit; rest become -inf."""
    logits = torch.ones(VOCAB)
    allowed = [3, 7, 15]
    out = apply_token_constraints(logits, allowed=allowed)
    for tok in allowed:
        assert out[tok].item() == 1.0, f"Allowed token {tok} should remain 1.0"
    for tok in range(VOCAB):
        if tok not in allowed:
            assert out[tok].item() == float("-inf"), f"Non-allowed token {tok} should be -inf"


# ---------------------------------------------------------------------------
# Test 4: apply_min_length_constraint — blocks EOS before min_len
# ---------------------------------------------------------------------------

def test_apply_min_length_constraint_blocks_eos_before_min_len():
    """EOS logit must be -inf when current_len < min_len."""
    logits = torch.zeros(VOCAB)
    out = apply_min_length_constraint(logits, current_len=2, min_len=5, eos_token_id=EOS)
    assert out[EOS].item() == float("-inf")
    # Other tokens unchanged
    assert out[0].item() == 0.0


def test_apply_min_length_constraint_allows_eos_at_min_len():
    """EOS logit must NOT be suppressed when current_len >= min_len."""
    logits = torch.zeros(VOCAB)
    out = apply_min_length_constraint(logits, current_len=5, min_len=5, eos_token_id=EOS)
    assert out[EOS].item() == 0.0


# ---------------------------------------------------------------------------
# Test 5: force_prefix returns correct token within prefix
# ---------------------------------------------------------------------------

def test_force_prefix_returns_correct_token():
    """force_prefix should return prefix[step] when step < len(prefix)."""
    prefix = [10, 20, 30]
    assert force_prefix([], prefix, 0) == 10
    assert force_prefix([10], prefix, 1) == 20
    assert force_prefix([10, 20], prefix, 2) == 30


# ---------------------------------------------------------------------------
# Test 6: force_prefix returns None after prefix is done
# ---------------------------------------------------------------------------

def test_force_prefix_returns_none_after_prefix():
    """force_prefix should return None when step >= len(prefix)."""
    prefix = [10, 20]
    assert force_prefix([10, 20], prefix, 2) is None
    assert force_prefix([10, 20, 5], prefix, 3) is None


# ---------------------------------------------------------------------------
# Test 7: LogitProcessor applies banned tokens correctly
# ---------------------------------------------------------------------------

def test_logit_processor_applies_banned():
    """LogitProcessor must zero out banned tokens."""
    cfg = ConstraintConfig(banned_tokens=[5, 6])
    processor = LogitProcessor(cfg)
    logits = torch.zeros(VOCAB)
    out = processor(logits, generated_ids=[])
    assert out[5].item() == float("-inf")
    assert out[6].item() == float("-inf")
    assert out[0].item() == 0.0


# ---------------------------------------------------------------------------
# Test 8: ConstrainedGreedyDecoder output length > PROMPT_LEN
# ---------------------------------------------------------------------------

def test_greedy_decoder_output_longer_than_prompt():
    """decode() output should contain more tokens than the original prompt."""
    cfg = ConstraintConfig(max_new_tokens=5)
    decoder = ConstrainedGreedyDecoder(mock_model, cfg, eos_token_id=EOS)
    prompt = make_prompt(PROMPT_LEN)
    out = decoder.decode(prompt)
    assert out.shape[0] > PROMPT_LEN


# ---------------------------------------------------------------------------
# Test 9: decoder output is 1D
# ---------------------------------------------------------------------------

def test_greedy_decoder_output_is_1d():
    """decode() must return a 1-D tensor."""
    cfg = ConstraintConfig(max_new_tokens=3)
    decoder = ConstrainedGreedyDecoder(mock_model, cfg, eos_token_id=EOS)
    prompt = make_prompt(PROMPT_LEN)
    out = decoder.decode(prompt)
    assert out.dim() == 1


# ---------------------------------------------------------------------------
# Test 10: decoder respects max_new_tokens
# ---------------------------------------------------------------------------

def test_greedy_decoder_respects_max_new_tokens():
    """decode() must not generate more than max_new_tokens new tokens."""
    max_new = 6
    cfg = ConstraintConfig(max_new_tokens=max_new)
    # Use a model that never generates EOS so we can test the hard cap
    model = biased_mock_model(bias_token=0)  # token 0 != EOS(2)
    decoder = ConstrainedGreedyDecoder(model, cfg, eos_token_id=EOS)
    prompt = make_prompt(PROMPT_LEN)
    out = decoder.decode(prompt)
    n_generated = out.shape[0] - PROMPT_LEN
    assert n_generated <= max_new


# ---------------------------------------------------------------------------
# Test 11: decoder with banned tokens never generates them
# ---------------------------------------------------------------------------

def test_greedy_decoder_never_generates_banned_tokens():
    """Banned tokens must not appear in the generated portion of the output."""
    # Ban tokens 5-29 so only 0,1,2,3,4 are reachable (EOS=2 may stop early)
    banned = list(range(5, VOCAB))
    cfg = ConstraintConfig(banned_tokens=banned, max_new_tokens=10)
    decoder = ConstrainedGreedyDecoder(mock_model, cfg, eos_token_id=EOS)
    prompt = make_prompt(PROMPT_LEN)
    out = decoder.decode(prompt)
    generated_tokens = out[PROMPT_LEN:].tolist()
    for tok in generated_tokens:
        assert tok not in banned, f"Banned token {tok} found in output"


# ---------------------------------------------------------------------------
# Test 12: ConstrainedSampler returns n_samples sequences
# ---------------------------------------------------------------------------

def test_constrained_sampler_returns_n_samples():
    """sample() must return exactly n_samples tensors."""
    cfg = ConstraintConfig(max_new_tokens=5)
    sampler = ConstrainedSampler(mock_model, cfg, eos_token_id=EOS)
    prompt = make_prompt(PROMPT_LEN)
    n = 4
    results = sampler.sample(prompt, n_samples=n)
    assert len(results) == n
    for r in results:
        assert isinstance(r, torch.Tensor)
        assert r.dim() == 1


# ---------------------------------------------------------------------------
# Test 13: compute_constraint_satisfaction keys present
# ---------------------------------------------------------------------------

def test_compute_constraint_satisfaction_keys_present():
    """compute_constraint_satisfaction must return dict with the required keys."""
    cfg = ConstraintConfig(banned_tokens=[99], allowed_tokens=[0, 1, 2], min_new_tokens=3)
    sequence = torch.tensor([0, 1, 2, 1, 0])
    result = compute_constraint_satisfaction(sequence, cfg)
    assert "no_banned_tokens" in result
    assert "all_allowed" in result
    assert "min_length_met" in result


# ---------------------------------------------------------------------------
# Test 14: compute_constraint_satisfaction correctness
# ---------------------------------------------------------------------------

def test_compute_constraint_satisfaction_correctness():
    """compute_constraint_satisfaction returns correct boolean values."""
    cfg = ConstraintConfig(
        banned_tokens=[9],
        allowed_tokens=[0, 1, 2, 3],
        min_new_tokens=5,
    )
    # Sequence has no banned tokens, all tokens allowed, length >= 5
    seq_good = torch.tensor([0, 1, 2, 3, 0])
    result = compute_constraint_satisfaction(seq_good, cfg)
    assert result["no_banned_tokens"] is True
    assert result["all_allowed"] is True
    assert result["min_length_met"] is True

    # Sequence contains banned token 9
    seq_banned = torch.tensor([0, 1, 9, 3, 0])
    result2 = compute_constraint_satisfaction(seq_banned, cfg)
    assert result2["no_banned_tokens"] is False

    # Sequence too short for min_new_tokens=5
    seq_short = torch.tensor([0, 1, 2])
    result3 = compute_constraint_satisfaction(seq_short, cfg)
    assert result3["min_length_met"] is False


# ---------------------------------------------------------------------------
# Test 15: LogitProcessor forces prefix tokens correctly
# ---------------------------------------------------------------------------

def test_logit_processor_forces_prefix_tokens():
    """LogitProcessor must force the prefix token at each prefix step."""
    prefix = [7, 13]
    cfg = ConstraintConfig(prefix_tokens=prefix)
    processor = LogitProcessor(cfg)
    logits = torch.zeros(VOCAB)

    # Step 0: prefix[0]=7 should be the only non -inf token
    out0 = processor(logits.clone(), generated_ids=[])
    assert out0[7].item() != float("-inf")
    for tok in range(VOCAB):
        if tok != 7:
            assert out0[tok].item() == float("-inf")

    # Step 1: prefix[1]=13 should be the only non -inf token
    out1 = processor(logits.clone(), generated_ids=[7])
    assert out1[13].item() != float("-inf")
    for tok in range(VOCAB):
        if tok != 13:
            assert out1[tok].item() == float("-inf")

    # Step 2: beyond prefix, no forcing (logits unchanged by prefix)
    out2 = processor(logits.clone(), generated_ids=[7, 13])
    # After prefix, tokens should not all be -inf
    assert not all(v == float("-inf") for v in out2.tolist())
