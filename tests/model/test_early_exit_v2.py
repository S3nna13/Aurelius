"""Tests for early_exit_v2.py — patience-based token-level early exit."""

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.early_exit_v2 import (
    PatienceConfig,
    PatienceTracker,
    TokenExitDecision,
    TokenLevelEarlyExit,
)
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture
def base_model(cfg):
    torch.manual_seed(42)
    return AureliusTransformer(cfg)


@pytest.fixture
def patience_cfg():
    return PatienceConfig(
        patience=3,
        min_layers=2,
        confidence_threshold=0.9,
        exit_on_confidence=True,
        exit_on_patience=True,
    )


@pytest.fixture
def model(base_model, patience_cfg):
    return TokenLevelEarlyExit(base_model, patience_cfg)


@pytest.fixture
def input_ids(cfg):
    torch.manual_seed(0)
    return torch.randint(0, cfg.vocab_size, (1, 8))


# ---------------------------------------------------------------------------
# Test 1 — PatienceConfig defaults
# ---------------------------------------------------------------------------


def test_patience_config_defaults():
    pc = PatienceConfig()
    assert pc.patience == 3
    assert pc.min_layers == 2
    assert pc.confidence_threshold == 0.9
    assert pc.exit_on_confidence is True
    assert pc.exit_on_patience is True


# ---------------------------------------------------------------------------
# Test 2 — TokenExitDecision fields
# ---------------------------------------------------------------------------


def test_token_exit_decision_fields():
    d = TokenExitDecision(token_pos=3, exit_layer=1, exit_reason="confidence", confidence=0.95)
    assert d.token_pos == 3
    assert d.exit_layer == 1
    assert d.exit_reason == "confidence"
    assert d.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Test 3 — PatienceTracker.update returns boolean tensor
# ---------------------------------------------------------------------------


def test_patience_tracker_update_returns_bool():
    tracker = PatienceTracker(patience=2, seq_len=4)
    logits = torch.zeros(1, 4, 16)
    result = tracker.update(logits)
    assert result.dtype == torch.bool


# ---------------------------------------------------------------------------
# Test 4 — PatienceTracker.update shape is (T,)
# ---------------------------------------------------------------------------


def test_patience_tracker_update_shape():
    T = 5
    tracker = PatienceTracker(patience=2, seq_len=T)
    logits = torch.zeros(1, T, 16)
    result = tracker.update(logits)
    assert result.shape == (T,)


# ---------------------------------------------------------------------------
# Test 5 — PatienceTracker.update increments counter for repeated argmax
# ---------------------------------------------------------------------------


def test_patience_tracker_increments_for_repeated_argmax():
    tracker = PatienceTracker(patience=3, seq_len=2)
    # Same logits → same argmax → counter should grow
    logits = torch.zeros(1, 2, 8)
    logits[0, :, 0] = 1.0  # argmax always 0
    tracker.update(logits)  # count = 1
    tracker.update(logits)  # count = 2
    # Counter should be 2 now, not yet at patience=3
    assert (tracker._consecutive_same == 2).all()


# ---------------------------------------------------------------------------
# Test 6 — PatienceTracker.update resets counter when argmax changes
# ---------------------------------------------------------------------------


def test_patience_tracker_resets_on_argmax_change():
    tracker = PatienceTracker(patience=3, seq_len=2)
    logits_a = torch.zeros(1, 2, 8)
    logits_a[0, :, 0] = 1.0  # argmax = 0
    logits_b = torch.zeros(1, 2, 8)
    logits_b[0, :, 1] = 1.0  # argmax = 1

    tracker.update(logits_a)  # count = 1
    tracker.update(logits_a)  # count = 2
    tracker.update(logits_b)  # argmax changed → count resets to 1
    assert (tracker._consecutive_same == 1).all()


# ---------------------------------------------------------------------------
# Test 7 — PatienceTracker.reset zeros all counters
# ---------------------------------------------------------------------------


def test_patience_tracker_reset():
    tracker = PatienceTracker(patience=3, seq_len=4)
    logits = torch.zeros(1, 4, 8)
    logits[0, :, 0] = 1.0
    tracker.update(logits)
    tracker.update(logits)
    tracker.reset()
    assert (tracker._consecutive_same == 0).all()
    assert (tracker._last_argmax == -1).all()


# ---------------------------------------------------------------------------
# Test 8 — TokenLevelEarlyExit.forward returns 3-tuple
# ---------------------------------------------------------------------------


def test_forward_returns_3_tuple(model, input_ids):
    with torch.no_grad():
        out = model(input_ids)
    assert isinstance(out, tuple)
    assert len(out) == 3


# ---------------------------------------------------------------------------
# Test 9 — logits shape is (B, T, V)
# ---------------------------------------------------------------------------


def test_forward_logits_shape(model, input_ids, cfg):
    B, T = input_ids.shape
    with torch.no_grad():
        _, logits, _ = model(input_ids)
    assert logits.shape == (B, T, cfg.vocab_size)


# ---------------------------------------------------------------------------
# Test 10 — decisions is a list
# ---------------------------------------------------------------------------


def test_forward_decisions_is_list(model, input_ids):
    with torch.no_grad():
        _, _, decisions = model(input_ids)
    assert isinstance(decisions, list)


# ---------------------------------------------------------------------------
# Test 11 — decisions length equals T (one decision per token position)
# ---------------------------------------------------------------------------


def test_forward_decisions_length(model, input_ids):
    _, T = input_ids.shape
    with torch.no_grad():
        _, _, decisions = model(input_ids)
    assert len(decisions) >= 1
    assert len(decisions) == T


# ---------------------------------------------------------------------------
# Test 12 — get_exit_stats returns required keys
# ---------------------------------------------------------------------------


def test_get_exit_stats_keys(model, input_ids):
    with torch.no_grad():
        _, _, decisions = model(input_ids)
    stats = model.get_exit_stats(decisions)
    assert "mean_exit_layer" in stats
    assert "exit_reason_counts" in stats
    assert "efficiency" in stats


# ---------------------------------------------------------------------------
# Test 13 — get_exit_stats efficiency is in [0, 1]
# ---------------------------------------------------------------------------


def test_get_exit_stats_efficiency_range(model, input_ids):
    with torch.no_grad():
        _, _, decisions = model(input_ids)
    stats = model.get_exit_stats(decisions)
    assert 0.0 <= stats["efficiency"] <= 1.0


# ---------------------------------------------------------------------------
# Test 14 — patience=1 exits after first repeated token
# ---------------------------------------------------------------------------


def test_patience_one_exits_after_first_repeat():
    tracker = PatienceTracker(patience=1, seq_len=3)
    logits = torch.zeros(1, 3, 8)
    logits[0, :, 2] = 1.0  # argmax = 2

    # First call: last_argmax was -1, so argmax changes → counter = 1 = patience → exited
    result = tracker.update(logits)
    # With patience=1, a single observation (counter reaches 1) should trigger exit
    assert result.dtype == torch.bool
    assert result.shape == (3,)
    assert result.all(), "With patience=1, all tokens should exit after first repeated argmax"


# ---------------------------------------------------------------------------
# Test 15 — exit_classifiers length equals n_layers
# ---------------------------------------------------------------------------


def test_exit_classifiers_length(model, cfg):
    assert len(model.exit_classifiers) == cfg.n_layers
