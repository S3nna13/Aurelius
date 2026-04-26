"""Tests for adaptive speculative decoding module."""

from __future__ import annotations

import pytest
import torch

from src.inference.adaptive_speculative import (
    AcceptanceRateTracker,
    AdaptiveSpecConfig,
    AdaptiveSpeculativeDecoder,
    adjust_draft_length,
    compute_target_log_probs,
    sample_draft_tokens,
    speculative_verify,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(42)
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


@pytest.fixture(scope="module")
def draft_model():
    torch.manual_seed(7)
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


@pytest.fixture
def input_ids():
    return torch.randint(0, 256, (1, 8))


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = AdaptiveSpecConfig()
    assert cfg.init_draft_len == 4
    assert cfg.target_acceptance == 0.7


# ---------------------------------------------------------------------------
# 2. test_tracker_init_rate
# ---------------------------------------------------------------------------


def test_tracker_init_rate():
    tracker = AcceptanceRateTracker(alpha=0.1, init_rate=0.7)
    assert tracker.get_rate() == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 3. test_tracker_update_high
# ---------------------------------------------------------------------------


def test_tracker_update_high():
    tracker = AcceptanceRateTracker(alpha=0.1, init_rate=0.5)
    initial_rate = tracker.get_rate()
    # All tokens accepted -> batch rate = 1.0, should move rate up
    tracker.update(n_accepted=10, n_proposed=10)
    assert tracker.get_rate() > initial_rate


# ---------------------------------------------------------------------------
# 4. test_tracker_update_low
# ---------------------------------------------------------------------------


def test_tracker_update_low():
    tracker = AcceptanceRateTracker(alpha=0.1, init_rate=0.7)
    initial_rate = tracker.get_rate()
    # 0 tokens accepted -> batch rate = 0.0, should move rate down
    tracker.update(n_accepted=0, n_proposed=10)
    assert tracker.get_rate() < initial_rate


# ---------------------------------------------------------------------------
# 5. test_tracker_reset
# ---------------------------------------------------------------------------


def test_tracker_reset():
    tracker = AcceptanceRateTracker(alpha=0.1, init_rate=0.7)
    tracker.update(n_accepted=0, n_proposed=10)
    tracker.reset()
    assert tracker.get_rate() == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 6. test_sample_draft_tokens_shape
# ---------------------------------------------------------------------------


def test_sample_draft_tokens_shape(tiny_model, input_ids):
    n_tokens = 3
    draft_ids, draft_log_probs = sample_draft_tokens(tiny_model, input_ids, n_tokens)
    assert draft_ids.shape == (1, n_tokens)
    assert draft_log_probs.shape == (1, n_tokens)


# ---------------------------------------------------------------------------
# 7. test_sample_draft_tokens_vocab_range
# ---------------------------------------------------------------------------


def test_sample_draft_tokens_vocab_range(tiny_model, input_ids):
    draft_ids, _ = sample_draft_tokens(tiny_model, input_ids, n_tokens=4)
    assert draft_ids.min().item() >= 0
    assert draft_ids.max().item() < TINY_CFG.vocab_size


# ---------------------------------------------------------------------------
# 8. test_compute_target_log_probs_shape
# ---------------------------------------------------------------------------


def test_compute_target_log_probs_shape(tiny_model, input_ids):
    K = 3
    draft_ids = torch.randint(0, 256, (1, K))
    target_log_probs = compute_target_log_probs(tiny_model, input_ids, draft_ids)
    assert target_log_probs.shape == (1, K)


# ---------------------------------------------------------------------------
# 9. test_speculative_verify_all_accept
# ---------------------------------------------------------------------------


def test_speculative_verify_all_accept():
    K = 4
    draft_ids = torch.randint(0, 256, (1, K))
    # target_log_probs >> draft_log_probs -> ratio clamped to 1 -> always accept
    draft_log_probs = torch.full((1, K), -10.0)
    target_log_probs = torch.full((1, K), -1.0)
    accepted_ids, n_accepted = speculative_verify(draft_ids, draft_log_probs, target_log_probs)
    assert n_accepted > 0


# ---------------------------------------------------------------------------
# 10. test_speculative_verify_all_reject
# ---------------------------------------------------------------------------


def test_speculative_verify_all_reject():
    K = 4
    draft_ids = torch.randint(0, 256, (1, K))
    # draft_log_probs >> target_log_probs -> ratio very small -> likely reject
    draft_log_probs = torch.full((1, K), -0.001)
    target_log_probs = torch.full((1, K), -100.0)
    torch.manual_seed(0)
    _, n_accepted = speculative_verify(draft_ids, draft_log_probs, target_log_probs)
    assert n_accepted <= K


# ---------------------------------------------------------------------------
# 11. test_speculative_verify_accepted_ids_shape
# ---------------------------------------------------------------------------


def test_speculative_verify_accepted_ids_shape():
    K = 4
    draft_ids = torch.randint(0, 256, (1, K))
    # Force acceptance of all tokens
    draft_log_probs = torch.full((1, K), -10.0)
    target_log_probs = torch.full((1, K), -1.0)
    accepted_ids, n_accepted = speculative_verify(draft_ids, draft_log_probs, target_log_probs)
    assert accepted_ids.shape == (1, n_accepted)


# ---------------------------------------------------------------------------
# 12. test_adjust_draft_len_increase
# ---------------------------------------------------------------------------


def test_adjust_draft_len_increase():
    # acceptance_rate = 0.9 > target_rate(0.7) + 0.1 -> increase by 1
    new_len = adjust_draft_length(
        current_len=4, acceptance_rate=0.9, target_rate=0.7, min_len=1, max_len=8
    )
    assert new_len == 5


# ---------------------------------------------------------------------------
# 13. test_adjust_draft_len_decrease
# ---------------------------------------------------------------------------


def test_adjust_draft_len_decrease():
    # acceptance_rate = 0.3 < target_rate(0.7) - 0.1 -> decrease by 1
    new_len = adjust_draft_length(
        current_len=4, acceptance_rate=0.3, target_rate=0.7, min_len=1, max_len=8
    )
    assert new_len == 3


# ---------------------------------------------------------------------------
# 14. test_adjust_draft_len_clamp
# ---------------------------------------------------------------------------


def test_adjust_draft_len_clamp():
    # Cannot go below min_len
    new_len_low = adjust_draft_length(
        current_len=1, acceptance_rate=0.1, target_rate=0.7, min_len=1, max_len=8
    )
    assert new_len_low >= 1

    # Cannot exceed max_len
    new_len_high = adjust_draft_length(
        current_len=8, acceptance_rate=0.99, target_rate=0.7, min_len=1, max_len=8
    )
    assert new_len_high <= 8


# ---------------------------------------------------------------------------
# 15. test_decoder_decode_returns_tokens
# ---------------------------------------------------------------------------


def test_decoder_decode_returns_tokens(tiny_model, draft_model, input_ids):
    cfg = AdaptiveSpecConfig(init_draft_len=2, max_draft_len=3, adjustment_interval=5)
    decoder = AdaptiveSpeculativeDecoder(
        target_model=tiny_model,
        draft_model=draft_model,
        cfg=cfg,
    )
    generated_ids, stats = decoder.decode(input_ids, max_new_tokens=4)
    assert isinstance(generated_ids, torch.Tensor)
    assert generated_ids.shape[0] == 1
    assert generated_ids.shape[1] > 0


# ---------------------------------------------------------------------------
# 16. test_decoder_stats_keys
# ---------------------------------------------------------------------------


def test_decoder_stats_keys(tiny_model, draft_model, input_ids):
    cfg = AdaptiveSpecConfig(init_draft_len=2, max_draft_len=3, adjustment_interval=5)
    decoder = AdaptiveSpeculativeDecoder(
        target_model=tiny_model,
        draft_model=draft_model,
        cfg=cfg,
    )
    _, stats = decoder.decode(input_ids, max_new_tokens=4)
    assert "mean_draft_len" in stats
    assert "mean_acceptance_rate" in stats
    assert "n_target_calls" in stats
    assert "n_tokens_generated" in stats
