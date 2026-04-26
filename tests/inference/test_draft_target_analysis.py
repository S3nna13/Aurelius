"""Tests for draft_target_analysis module."""

from __future__ import annotations

import torch

from src.inference.draft_target_analysis import (
    AcceptanceStats,
    DraftQualityMonitor,
    acceptance_rate,
    calibrate_draft_temperature,
    compute_acceptance_stats,
    compute_expected_speedup,
    compute_token_acceptance,
    kl_divergence_analysis,
    top_k_overlap,
)

# Standard test dimensions
VOCAB = 64
BATCH = 2
T_DRAFT = 4


def make_logits(B: int, T: int, V: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T, V)


def make_tokens(B: int, T: int, V: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, V, (B, T))


# ----- Test 1: compute_token_acceptance returns boolean tensor of shape (B, T_draft) -----


def test_compute_token_acceptance_shape():
    draft_tokens = make_tokens(BATCH, T_DRAFT, VOCAB)
    target_logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=1)
    draft_logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=2)

    result = compute_token_acceptance(draft_tokens, target_logits, draft_logits)

    assert result.shape == (BATCH, T_DRAFT)
    assert result.dtype == torch.bool


# ----- Test 2: identical draft and target → acceptance rate ≈ 1.0 -----


def test_compute_token_acceptance_identical_high_rate():
    # When draft and target are identical, ratio = 1.0, always accepted
    logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=5)
    draft_tokens = make_tokens(BATCH, T_DRAFT, VOCAB)

    result = compute_token_acceptance(draft_tokens, logits, logits)

    # All should be accepted since ratio = min(1, p/p) = 1 and uniform < 1
    rate = result.float().mean().item()
    assert rate > 0.9, f"Expected high acceptance rate, got {rate}"


# ----- Test 3: very different draft and target → some rejections -----


def test_compute_token_acceptance_different_some_rejections():
    torch.manual_seed(99)
    # Make target put all mass on token 0, draft put all mass on token 1
    target_logits = torch.full((BATCH, T_DRAFT, VOCAB), -100.0)
    target_logits[:, :, 0] = 100.0

    draft_logits = torch.full((BATCH, T_DRAFT, VOCAB), -100.0)
    draft_logits[:, :, 1] = 100.0

    # Draft proposes token 1 which has near-zero target probability
    draft_tokens = torch.ones(BATCH, T_DRAFT, dtype=torch.long)

    result = compute_token_acceptance(draft_tokens, target_logits, draft_logits)

    # With p_target(1) ≈ 0 and p_draft(1) ≈ 1, ratio ≈ 0, nearly all rejected
    rate = result.float().mean().item()
    assert rate < 0.1, f"Expected low acceptance rate, got {rate}"


# ----- Test 4: acceptance_rate returns float in [0, 1] -----


def test_acceptance_rate_returns_float_in_range():
    draft_tokens = make_tokens(BATCH, T_DRAFT, VOCAB)
    target_logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=10)
    draft_logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=11)

    rate = acceptance_rate(draft_tokens, target_logits, draft_logits)

    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0


# ----- Test 5: acceptance_rate: identical logits → ≈ 1.0 -----


def test_acceptance_rate_identical_logits():
    logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=7)
    draft_tokens = make_tokens(BATCH, T_DRAFT, VOCAB)

    rate = acceptance_rate(draft_tokens, logits, logits)

    assert rate > 0.9, f"Expected high acceptance rate with identical logits, got {rate}"


# ----- Test 6: compute_acceptance_stats returns AcceptanceStats -----


def test_compute_acceptance_stats_returns_dataclass():
    draft_tokens = make_tokens(BATCH, T_DRAFT, VOCAB)
    target_logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=20)
    draft_logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=21)

    stats = compute_acceptance_stats(draft_tokens, target_logits, draft_logits)

    assert isinstance(stats, AcceptanceStats)


# ----- Test 7: AcceptanceStats.acceptance_rate in [0, 1] -----


def test_acceptance_stats_rate_in_range():
    draft_tokens = make_tokens(BATCH, T_DRAFT, VOCAB)
    target_logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=30)
    draft_logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=31)

    stats = compute_acceptance_stats(draft_tokens, target_logits, draft_logits)

    assert 0.0 <= stats.acceptance_rate <= 1.0


# ----- Test 8: AcceptanceStats.token_acceptance_dist length == T_draft -----


def test_acceptance_stats_dist_length():
    draft_tokens = make_tokens(BATCH, T_DRAFT, VOCAB)
    target_logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=40)
    draft_logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=41)

    stats = compute_acceptance_stats(draft_tokens, target_logits, draft_logits)

    assert len(stats.token_acceptance_dist) == T_DRAFT


# ----- Test 9: calibrate_draft_temperature returns float in [0.5, 2.0] -----


def test_calibrate_draft_temperature_range():
    torch.manual_seed(0)
    N = 50
    draft_logits = torch.randn(N, VOCAB)
    target_logits = torch.randn(N, VOCAB)

    temp = calibrate_draft_temperature(draft_logits, target_logits, n_steps=10)

    assert isinstance(temp, float)
    assert 0.5 <= temp <= 2.0


# ----- Test 10: kl_divergence_analysis returns dict with 'mean_kl' key -----


def test_kl_divergence_analysis_returns_dict():
    torch.manual_seed(0)
    N = 20
    draft_logits = torch.randn(N, VOCAB)
    target_logits = torch.randn(N, VOCAB)

    result = kl_divergence_analysis(draft_logits, target_logits)

    assert isinstance(result, dict)
    assert "mean_kl" in result
    assert "max_kl" in result
    assert "frac_high_kl" in result


# ----- Test 11: kl_divergence_analysis: identical logits → mean_kl ≈ 0 -----


def test_kl_divergence_identical_logits():
    torch.manual_seed(0)
    N = 20
    logits = torch.randn(N, VOCAB)

    result = kl_divergence_analysis(logits, logits)

    assert result["mean_kl"] < 1e-4, f"Expected mean_kl ≈ 0, got {result['mean_kl']}"


# ----- Test 12: compute_expected_speedup: acceptance=1.0 gives speedup > 1.0 -----


def test_expected_speedup_full_acceptance():
    speedup = compute_expected_speedup(acceptance_rate=1.0, n_draft_tokens=4, draft_overhead=0.1)

    assert speedup > 1.0, f"Expected speedup > 1.0 with full acceptance, got {speedup}"


# ----- Test 13: compute_expected_speedup: acceptance=0.0 gives speedup <= 1.0 -----


def test_expected_speedup_zero_acceptance():
    speedup = compute_expected_speedup(acceptance_rate=0.0, n_draft_tokens=4, draft_overhead=0.1)

    # With zero acceptance: numerator = 1 (only alpha^0), denominator = 1 + 4*0.1 = 1.4
    assert speedup <= 1.0, f"Expected speedup <= 1.0 with zero acceptance, got {speedup}"


# ----- Test 14: DraftQualityMonitor.update and get_stats run without error -----


def test_draft_quality_monitor_update_get_stats():
    monitor = DraftQualityMonitor(window_size=10, ema_decay=0.9, alert_threshold=0.5)

    monitor.update(accepted=8, total=10)
    monitor.update(accepted=7, total=10)

    stats = monitor.get_stats()

    assert "ema_acceptance" in stats
    assert "window_acceptance" in stats
    assert "total_accepted" in stats


# ----- Test 15: DraftQualityMonitor.should_fallback False when acceptance is high -----


def test_draft_quality_monitor_no_fallback_high_acceptance():
    monitor = DraftQualityMonitor(window_size=10, ema_decay=0.5, alert_threshold=0.5)

    # Feed high acceptance rates
    for _ in range(10):
        monitor.update(accepted=9, total=10)

    assert not monitor.should_fallback(), "Should not fallback with high acceptance rate"


# ----- Test 16: top_k_overlap: identical logits → 1.0 -----


def test_top_k_overlap_identical_logits():
    torch.manual_seed(0)
    logits = torch.randn(VOCAB)

    overlap = top_k_overlap(logits, logits, k=10)

    assert overlap == 1.0, f"Expected overlap 1.0 with identical logits, got {overlap}"


# ----- Bonus tests -----


def test_top_k_overlap_disjoint_logits():
    """Logits that put top-k in completely different positions → overlap near 0."""
    logits_a = torch.zeros(VOCAB)
    logits_b = torch.zeros(VOCAB)

    k = 5
    # a: top-k are tokens 0..4
    logits_a[:k] = 100.0
    # b: top-k are tokens (VOCAB-k)..(VOCAB-1) — disjoint
    logits_b[VOCAB - k :] = 100.0

    overlap = top_k_overlap(logits_a, logits_b, k=k)

    assert overlap == 0.0, f"Expected 0.0 overlap, got {overlap}"


def test_draft_quality_monitor_reset():
    monitor = DraftQualityMonitor()
    monitor.update(accepted=5, total=10)
    monitor.reset()

    stats = monitor.get_stats()
    assert stats["total_accepted"] == 0.0
    assert stats["window_acceptance"] == 0.0


def test_compute_acceptance_stats_total_tokens():
    draft_tokens = make_tokens(BATCH, T_DRAFT, VOCAB)
    target_logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=50)
    draft_logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=51)

    stats = compute_acceptance_stats(draft_tokens, target_logits, draft_logits)

    assert stats.total_tokens == BATCH * T_DRAFT
    assert 0 <= stats.accepted_tokens <= stats.total_tokens


def test_acceptance_stats_mean_acceptance_length_range():
    draft_tokens = make_tokens(BATCH, T_DRAFT, VOCAB)
    target_logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=60)
    draft_logits = make_logits(BATCH, T_DRAFT, VOCAB, seed=61)

    stats = compute_acceptance_stats(draft_tokens, target_logits, draft_logits)

    assert 0.0 <= stats.mean_acceptance_length <= T_DRAFT
