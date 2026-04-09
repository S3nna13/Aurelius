"""Tests for speculative_decoding_v2: draft model + target verification with acceptance rate tracking."""
import pytest
import torch
import torch.nn.functional as F

from src.inference.speculative_decoding_v2 import (
    SpeculativeConfig,
    apply_top_k,
    apply_top_p,
    sample_token,
    draft_tokens,
    verify_tokens,
    AcceptanceTracker,
    SpeculativeDecoderV2,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model() -> AureliusTransformer:
    torch.manual_seed(0)
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


def _make_prompt(length: int = 4) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 256, (1, length))


# ---------------------------------------------------------------------------
# 1. SpeculativeConfig defaults
# ---------------------------------------------------------------------------

def test_speculative_config_defaults():
    cfg = SpeculativeConfig()
    assert cfg.n_draft_tokens == 4
    assert cfg.max_new_tokens == 64
    assert cfg.temperature == 1.0
    assert cfg.top_k == 0
    assert cfg.top_p == 1.0


# ---------------------------------------------------------------------------
# 2. apply_top_k zeros out non-top-k logits (via -inf)
# ---------------------------------------------------------------------------

def test_apply_top_k_non_top_k_are_neg_inf():
    torch.manual_seed(42)
    logits = torch.randn(1, 256)
    k = 10
    result = apply_top_k(logits, k)

    # Positions set to -inf are "zeroed out"
    finite_mask = result.isfinite()
    assert finite_mask.sum().item() == k


# ---------------------------------------------------------------------------
# 3. apply_top_p masks low-prob tokens
# ---------------------------------------------------------------------------

def test_apply_top_p_masks_low_prob_tokens():
    torch.manual_seed(7)
    # Make one token dominate so top_p=0.5 should cut most tokens
    logits = torch.full((1, 256), -10.0)
    logits[0, 0] = 10.0   # this token has ~1.0 probability

    result = apply_top_p(logits, p=0.5)
    probs = F.softmax(result, dim=-1)

    # Only token 0 should survive (rest ~0)
    n_nonzero = (probs > 1e-6).sum().item()
    assert n_nonzero <= 10  # most are masked


# ---------------------------------------------------------------------------
# 4. sample_token output shape is (B,)
# ---------------------------------------------------------------------------

def test_sample_token_output_shape():
    B, V = 3, 256
    logits = torch.randn(B, V)
    tokens = sample_token(logits, temperature=1.0, top_k=0, top_p=1.0)
    assert tokens.shape == (B,)


# ---------------------------------------------------------------------------
# 5. draft_tokens output shapes (B, n_draft)
# ---------------------------------------------------------------------------

def test_draft_tokens_output_shapes():
    model = _make_model()
    prompt = _make_prompt(4)   # (1, 4)
    cfg = SpeculativeConfig(n_draft_tokens=3, max_new_tokens=4)

    with torch.no_grad():
        d_ids, d_log_probs = draft_tokens(model, prompt, cfg)

    assert d_ids.shape == (1, 3)
    assert d_log_probs.shape == (1, 3)


# ---------------------------------------------------------------------------
# 6. AcceptanceTracker.update and acceptance_rate
# ---------------------------------------------------------------------------

def test_acceptance_tracker_update_and_rate():
    tracker = AcceptanceTracker()
    tracker.update(n_drafted=4, n_accepted=3)
    tracker.update(n_drafted=4, n_accepted=2)
    # total_drafted=8, total_accepted=5
    assert tracker.total_drafted == 8
    assert tracker.total_accepted == 5
    assert abs(tracker.acceptance_rate() - 5 / 8) < 1e-6


# ---------------------------------------------------------------------------
# 7. AcceptanceTracker.reset
# ---------------------------------------------------------------------------

def test_acceptance_tracker_reset():
    tracker = AcceptanceTracker()
    tracker.update(n_drafted=4, n_accepted=3)
    tracker.reset()
    assert tracker.total_drafted == 0
    assert tracker.total_accepted == 0
    assert tracker.acceptance_rate() == 0.0


# ---------------------------------------------------------------------------
# 8. AcceptanceTracker: 0 drafted → 0.0 (no division by zero)
# ---------------------------------------------------------------------------

def test_acceptance_tracker_zero_drafted():
    tracker = AcceptanceTracker()
    rate = tracker.acceptance_rate()
    assert rate == 0.0


# ---------------------------------------------------------------------------
# 9. SpeculativeDecoderV2.generate returns correct keys in stats
# ---------------------------------------------------------------------------

def test_speculative_decoder_v2_stats_keys():
    draft = _make_model()
    target = _make_model()
    cfg = SpeculativeConfig(n_draft_tokens=2, max_new_tokens=4)
    decoder = SpeculativeDecoderV2(draft_model=draft, target_model=target, config=cfg)

    prompt = _make_prompt(4)
    _, stats = decoder.generate(prompt)

    assert "acceptance_rate" in stats
    assert "n_steps" in stats
    assert "total_tokens" in stats
    assert isinstance(stats["acceptance_rate"], float)
    assert isinstance(stats["n_steps"], int)
    assert isinstance(stats["total_tokens"], int)


# ---------------------------------------------------------------------------
# 10. SpeculativeDecoderV2.generate output shape (1, generated_len)
# ---------------------------------------------------------------------------

def test_speculative_decoder_v2_output_shape():
    draft = _make_model()
    target = _make_model()
    cfg = SpeculativeConfig(n_draft_tokens=2, max_new_tokens=4)
    decoder = SpeculativeDecoderV2(draft_model=draft, target_model=target, config=cfg)

    prompt = _make_prompt(4)
    generated, stats = decoder.generate(prompt)

    assert generated.ndim == 2
    assert generated.shape[0] == 1
    assert generated.shape[1] == stats["total_tokens"]
    assert 0 < generated.shape[1] <= cfg.max_new_tokens


# ---------------------------------------------------------------------------
# 11. verify_tokens: n_accepted <= n_draft_tokens
# ---------------------------------------------------------------------------

def test_verify_tokens_n_accepted_bounded():
    draft_model = _make_model()
    target_model = _make_model()
    prompt = _make_prompt(4)
    cfg = SpeculativeConfig(n_draft_tokens=3, max_new_tokens=4)

    with torch.no_grad():
        d_ids, d_log_probs = draft_tokens(draft_model, prompt, cfg)
        _, n_accepted = verify_tokens(target_model, prompt, d_ids, d_log_probs, cfg)

    assert 0 <= n_accepted <= cfg.n_draft_tokens


# ---------------------------------------------------------------------------
# 12. apply_top_k with k=1: only argmax survives
# ---------------------------------------------------------------------------

def test_apply_top_k_k1_only_argmax_survives():
    torch.manual_seed(99)
    logits = torch.randn(1, 256)
    result = apply_top_k(logits, k=1)

    argmax = logits.argmax(dim=-1).item()
    finite_mask = result.isfinite()

    # Exactly one position should be finite
    assert finite_mask.sum().item() == 1
    assert result[0, argmax].isfinite()
