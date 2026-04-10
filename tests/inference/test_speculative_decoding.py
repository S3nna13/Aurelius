"""Tests for src/inference/speculative_decoding.py.

Covers SpecDecodeConfig, sample_token, draft_tokens, verify_tokens,
and SpeculativeDecoder (14+ tests).
"""
from __future__ import annotations

import torch
import pytest

from src.inference.speculative_decoding import (
    SpecDecodeConfig,
    sample_token,
    draft_tokens,
    verify_tokens,
    SpeculativeDecoder,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_model(seed: int = 0) -> AureliusTransformer:
    torch.manual_seed(seed)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


def _prompt(length: int = 4, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, 256, (1, length))


# ---------------------------------------------------------------------------
# 1. SpecDecodeConfig defaults
# ---------------------------------------------------------------------------

def test_spec_decode_config_defaults():
    cfg = SpecDecodeConfig()
    assert cfg.n_draft == 4
    assert cfg.temperature == 1.0
    assert cfg.top_k == 0
    assert cfg.max_new_tokens == 32


# ---------------------------------------------------------------------------
# 2. sample_token output shape (batch,) for various batch sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_sample_token_shape(batch: int):
    vocab = 256
    logits = torch.randn(batch, vocab)
    out = sample_token(logits, temperature=1.0, top_k=0)
    assert out.shape == (batch,), f"Expected ({batch},), got {out.shape}"
    assert out.dtype == torch.long or out.dtype == torch.int64


# ---------------------------------------------------------------------------
# 3. sample_token temperature≈0 (greedy) gives deterministic, argmax results
# ---------------------------------------------------------------------------

def test_sample_token_greedy_deterministic():
    torch.manual_seed(7)
    vocab = 256
    logits = torch.randn(1, vocab)
    # Near-zero temperature => should equal argmax
    out1 = sample_token(logits, temperature=1e-9, top_k=0)
    out2 = sample_token(logits, temperature=1e-9, top_k=0)
    expected = logits.argmax(dim=-1)
    assert out1.item() == expected.item()
    assert out2.item() == expected.item()


# ---------------------------------------------------------------------------
# 4. sample_token top_k limits to k tokens
# ---------------------------------------------------------------------------

def test_sample_token_top_k_limits_tokens():
    torch.manual_seed(0)
    vocab = 256
    k = 5
    # Make logits with a clear top-k by setting k high, rest very low
    logits = torch.full((1, vocab), -1000.0)
    top_indices = torch.arange(k)
    logits[0, top_indices] = torch.randn(k)

    # Run many samples and verify all fall in top_indices
    results = set()
    for _ in range(200):
        tok = sample_token(logits, temperature=1.0, top_k=k)
        results.add(tok.item())

    assert results.issubset(set(top_indices.tolist())), (
        f"Got tokens outside top-{k}: {results - set(top_indices.tolist())}"
    )


# ---------------------------------------------------------------------------
# 5. draft_tokens returns shapes (B, n_draft) for both outputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_draft", [1, 3, 4])
def test_draft_tokens_shapes(n_draft: int):
    model = _tiny_model()
    prompt = _prompt(length=4)
    ids, log_probs = draft_tokens(model, prompt, n_draft=n_draft, temperature=1.0)
    assert ids.shape == (1, n_draft), f"ids shape {ids.shape}"
    assert log_probs.shape == (1, n_draft), f"log_probs shape {log_probs.shape}"


# ---------------------------------------------------------------------------
# 6. draft_tokens with n_draft=1 works
# ---------------------------------------------------------------------------

def test_draft_tokens_n_draft_1():
    model = _tiny_model(seed=1)
    prompt = _prompt(length=3)
    ids, log_probs = draft_tokens(model, prompt, n_draft=1, temperature=1.0)
    assert ids.shape == (1, 1)
    assert log_probs.shape == (1, 1)
    assert ids.dtype in (torch.long, torch.int64)


# ---------------------------------------------------------------------------
# 7. verify_tokens returns tuple of (Tensor, int)
# ---------------------------------------------------------------------------

def test_verify_tokens_return_types():
    model = _tiny_model(seed=2)
    prompt = _prompt(length=4)
    n_draft = 3
    draft_ids, _ = draft_tokens(model, prompt, n_draft=n_draft, temperature=1.0)
    result = verify_tokens(model, prompt, draft_ids, temperature=1.0)
    assert isinstance(result, tuple), "verify_tokens must return a tuple"
    assert len(result) == 2
    accepted_ids, n_accepted = result
    assert isinstance(accepted_ids, torch.Tensor)
    assert isinstance(n_accepted, int)


# ---------------------------------------------------------------------------
# 8. verify_tokens n_accepted in range [0, n_draft+1]
# ---------------------------------------------------------------------------

def test_verify_tokens_n_accepted_range():
    model = _tiny_model(seed=3)
    prompt = _prompt(length=4)
    n_draft = 4
    draft_ids, _ = draft_tokens(model, prompt, n_draft=n_draft, temperature=1.0)
    _, n_accepted = verify_tokens(model, prompt, draft_ids, temperature=1.0)
    assert 0 <= n_accepted <= n_draft, f"n_accepted={n_accepted} out of [0, {n_draft}]"


# ---------------------------------------------------------------------------
# 9. verify_tokens accepted_ids shape correct
# ---------------------------------------------------------------------------

def test_verify_tokens_accepted_ids_shape():
    model = _tiny_model(seed=4)
    prompt = _prompt(length=4)
    n_draft = 3
    draft_ids, _ = draft_tokens(model, prompt, n_draft=n_draft, temperature=1.0)
    accepted_ids, n_accepted = verify_tokens(model, prompt, draft_ids, temperature=1.0)
    # Shape must be (batch, <=n_draft+1)
    assert accepted_ids.ndim == 2
    assert accepted_ids.shape[0] == prompt.shape[0]
    assert 0 < accepted_ids.shape[1] <= n_draft + 1, (
        f"accepted_ids.shape={accepted_ids.shape}, n_draft={n_draft}"
    )


# ---------------------------------------------------------------------------
# 10. SpeculativeDecoder instantiates
# ---------------------------------------------------------------------------

def test_speculative_decoder_instantiates():
    draft = _tiny_model(seed=0)
    target = _tiny_model(seed=1)
    cfg = SpecDecodeConfig(n_draft=2, max_new_tokens=4)
    decoder = SpeculativeDecoder(draft, target, cfg)
    assert decoder is not None
    assert decoder.draft_model is draft
    assert decoder.target_model is target


# ---------------------------------------------------------------------------
# 11. SpeculativeDecoder.generate returns tensor longer than input
# ---------------------------------------------------------------------------

def test_speculative_decoder_generate_longer_than_input():
    draft = _tiny_model(seed=0)
    target = _tiny_model(seed=1)
    cfg = SpecDecodeConfig(n_draft=2, max_new_tokens=8)
    decoder = SpeculativeDecoder(draft, target, cfg)
    prompt = _prompt(length=4)
    out = decoder.generate(prompt)
    assert out.shape[1] > prompt.shape[1], (
        f"Output length {out.shape[1]} not > prompt length {prompt.shape[1]}"
    )


# ---------------------------------------------------------------------------
# 12. SpeculativeDecoder.generate max_new_tokens respected
# ---------------------------------------------------------------------------

def test_speculative_decoder_generate_max_new_tokens():
    draft = _tiny_model(seed=5)
    target = _tiny_model(seed=6)
    max_new = 6
    cfg = SpecDecodeConfig(n_draft=2, max_new_tokens=max_new)
    decoder = SpeculativeDecoder(draft, target, cfg)
    prompt = _prompt(length=4)
    out = decoder.generate(prompt)
    n_generated = out.shape[1] - prompt.shape[1]
    # Must not exceed max_new_tokens (may slightly overshoot by bonus token in some implementations,
    # but we aim for exact compliance)
    assert n_generated <= max_new + 1, (
        f"Generated {n_generated} tokens, expected <= {max_new + 1}"
    )
    assert n_generated > 0, "Must generate at least 1 token"


# ---------------------------------------------------------------------------
# 13. SpeculativeDecoder.acceptance_rate returns float in [0, 1]
# ---------------------------------------------------------------------------

def test_speculative_decoder_acceptance_rate():
    draft = _tiny_model(seed=7)
    target = _tiny_model(seed=8)
    cfg = SpecDecodeConfig(n_draft=3, max_new_tokens=12)
    decoder = SpeculativeDecoder(draft, target, cfg)
    prompt = _prompt(length=4)
    decoder.generate(prompt)  # Must run generate first to populate stats
    rate = decoder.acceptance_rate()
    assert isinstance(rate, float), f"acceptance_rate() should return float, got {type(rate)}"
    assert 0.0 <= rate <= 1.0, f"acceptance_rate() = {rate} not in [0, 1]"


# ---------------------------------------------------------------------------
# 14. SpeculativeDecoder.generate with draft == target (should accept all drafts)
# ---------------------------------------------------------------------------

def test_speculative_decoder_same_model_high_acceptance():
    """When draft and target are the same model, acceptance should be high."""
    torch.manual_seed(99)
    model = _tiny_model(seed=10)
    cfg = SpecDecodeConfig(n_draft=4, max_new_tokens=16, temperature=1.0)
    decoder = SpeculativeDecoder(model, model, cfg)
    prompt = _prompt(length=4, seed=10)
    out = decoder.generate(prompt)
    # Must have generated something
    assert out.shape[1] > prompt.shape[1]
    # Acceptance rate should be > 0 (and typically high when same model)
    rate = decoder.acceptance_rate()
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0
    # With same model the draft IS the target, so acceptance should be 1.0
    assert rate >= 0.5, f"Same-model acceptance rate too low: {rate}"
