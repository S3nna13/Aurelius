"""Tests for speculative decoding v2 (full rejection sampling)."""
import pytest
import torch
from src.inference.speculative_v2 import (
    SpeculativeConfig,
    sample_from_logits,
    rejection_sample,
    speculative_decode_step,
    AcceptanceTracker,
    SpeculativeDecoderV2,
    estimate_draft_quality,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


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
    assert cfg.draft_temperature == 1.0
    assert cfg.track_acceptance is True


# ---------------------------------------------------------------------------
# 2. sample_from_logits — valid token
# ---------------------------------------------------------------------------

def test_sample_from_logits_valid_token():
    V = 256
    logits = torch.randn(V)
    token_id, probs = sample_from_logits(logits, temperature=1.0)
    assert isinstance(token_id, int)
    assert 0 <= token_id < V


# ---------------------------------------------------------------------------
# 3. sample_from_logits — probabilities sum to ~1
# ---------------------------------------------------------------------------

def test_sample_from_logits_probabilities_sum():
    V = 256
    logits = torch.randn(V)
    _, probs = sample_from_logits(logits, temperature=1.0)
    assert probs.shape == (V,)
    assert abs(probs.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 4. rejection_sample — identical distributions always accept
# ---------------------------------------------------------------------------

def test_rejection_sample_accept_identical():
    V = 256
    torch.manual_seed(42)
    probs = torch.randn(V).abs()
    probs = probs / probs.sum()
    draft_token = int(probs.argmax().item())

    # When target == draft, acceptance prob = min(1, p/p) = 1, always accept
    n_trials = 50
    accepted_count = 0
    for _ in range(n_trials):
        token, was_accepted = rejection_sample(probs.clone(), probs.clone(), draft_token)
        if was_accepted:
            accepted_count += 1

    # With identical distributions all should be accepted
    assert accepted_count == n_trials


# ---------------------------------------------------------------------------
# 5. rejection_sample — returns valid token
# ---------------------------------------------------------------------------

def test_rejection_sample_returns_valid_token():
    V = 256
    torch.manual_seed(7)
    target_prob = torch.rand(V)
    target_prob /= target_prob.sum()
    draft_prob = torch.rand(V)
    draft_prob /= draft_prob.sum()
    draft_token = int(draft_prob.argmax().item())

    token, was_accepted = rejection_sample(target_prob, draft_prob, draft_token)
    assert isinstance(token, int)
    assert 0 <= token < V
    assert isinstance(was_accepted, bool)


# ---------------------------------------------------------------------------
# 6. speculative_decode_step — returns list
# ---------------------------------------------------------------------------

def test_speculative_decode_step_accepted_list():
    target = _make_model()
    draft = _make_model()
    prompt = _make_prompt(4)

    accepted_tokens, n_accepted, n_proposed = speculative_decode_step(
        target_model=target,
        draft_model=draft,
        prompt_ids=prompt,
        n_draft=2,
        temperature=1.0,
    )
    assert isinstance(accepted_tokens, list)
    assert len(accepted_tokens) >= 1


# ---------------------------------------------------------------------------
# 7. speculative_decode_step — n_proposed == n_draft
# ---------------------------------------------------------------------------

def test_speculative_decode_step_counts():
    target = _make_model()
    draft = _make_model()
    prompt = _make_prompt(4)
    n_draft = 2

    accepted_tokens, n_accepted, n_proposed = speculative_decode_step(
        target_model=target,
        draft_model=draft,
        prompt_ids=prompt,
        n_draft=n_draft,
        temperature=1.0,
    )
    assert n_proposed == n_draft
    assert 0 <= n_accepted <= n_draft


# ---------------------------------------------------------------------------
# 8. AcceptanceTracker — tracks correctly
# ---------------------------------------------------------------------------

def test_acceptance_tracker_rate():
    tracker = AcceptanceTracker()
    tracker.record(3, 4)
    tracker.record(2, 4)
    # n_accepted = 5, n_proposed = 8
    assert abs(tracker.acceptance_rate() - 5 / 8) < 1e-6
    assert tracker.n_accepted == 5
    assert tracker.n_proposed == 8


# ---------------------------------------------------------------------------
# 9. AcceptanceTracker — speedup positive
# ---------------------------------------------------------------------------

def test_acceptance_tracker_speedup_positive():
    tracker = AcceptanceTracker()
    tracker.record(2, 4)
    tracker.record(3, 4)
    speedup = tracker.speedup_estimate()
    assert speedup > 0.0


# ---------------------------------------------------------------------------
# 10. SpeculativeDecoderV2.generate — returns tokens
# ---------------------------------------------------------------------------

def test_speculative_decoder_v2_generate_returns_tokens():
    target = _make_model()
    draft = _make_model()
    cfg = SpeculativeConfig(n_draft_tokens=2, max_new_tokens=4)
    decoder = SpeculativeDecoderV2(target_model=target, draft_model=draft, config=cfg)

    prompt = _make_prompt(4)
    generated, stats = decoder.generate(prompt)

    assert isinstance(generated, list)
    assert len(generated) > 0
    assert len(generated) <= cfg.max_new_tokens


# ---------------------------------------------------------------------------
# 11. SpeculativeDecoderV2.generate — stats keys
# ---------------------------------------------------------------------------

def test_speculative_decoder_v2_stats_keys():
    target = _make_model()
    draft = _make_model()
    cfg = SpeculativeConfig(n_draft_tokens=2, max_new_tokens=4)
    decoder = SpeculativeDecoderV2(target_model=target, draft_model=draft, config=cfg)

    prompt = _make_prompt(4)
    generated, stats = decoder.generate(prompt)

    assert "acceptance_rate" in stats
    assert "n_steps" in stats
    assert "total_tokens" in stats
    assert isinstance(stats["acceptance_rate"], float)
    assert isinstance(stats["n_steps"], int)
    assert isinstance(stats["total_tokens"], int)


# ---------------------------------------------------------------------------
# 12. estimate_draft_quality — keys
# ---------------------------------------------------------------------------

def test_estimate_draft_quality_keys():
    target = _make_model()
    draft = _make_model()
    prompt = _make_prompt(4)

    result = estimate_draft_quality(target, draft, prompt, n_eval=4)

    assert "agreement_rate" in result
    assert "mean_prob_ratio" in result
    assert 0.0 <= result["agreement_rate"] <= 1.0
    assert result["mean_prob_ratio"] > 0.0
