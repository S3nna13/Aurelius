"""Unit + integration tests for AdaptiveKVEvictionManager.

Run with:
    pytest tests/inference/test_adaptive_kv_eviction.py -v
"""

from __future__ import annotations

import pytest
import torch

from src.inference.adaptive_kv_eviction import (
    AdaptiveKVConfig,
    AdaptiveKVEvictionManager,
    KVCacheState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_manager(**kwargs) -> AdaptiveKVEvictionManager:
    cfg = AdaptiveKVConfig(**kwargs)
    return AdaptiveKVEvictionManager(cfg)


def make_kv(n_heads: int, T: int, head_dim: int, seed: int = 0) -> tuple:
    g = torch.Generator()
    g.manual_seed(seed)
    keys   = torch.rand(n_heads, T, head_dim, generator=g)
    values = torch.rand(n_heads, T, head_dim, generator=g)
    return keys, values


def uniform_attn(n_heads: int, T: int) -> torch.Tensor:
    """Uniform attention weights summing to 1 per head."""
    return torch.full((n_heads, T), 1.0 / T)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_config_defaults(self):
        cfg = AdaptiveKVConfig()
        assert cfg.n_layers      == 24
        assert cfg.n_heads       == 16
        assert cfg.head_dim      == 128
        assert cfg.max_seq_len   == 8192
        assert cfg.budget_ratio  == pytest.approx(0.3)
        assert cfg.min_budget    == 64
        assert cfg.recent_window == 32
        assert cfg.accumulate_steps == 1


# ---------------------------------------------------------------------------
# 2. Compute budget — ratio
# ---------------------------------------------------------------------------

class TestComputeBudgetRatio:
    def test_compute_budget_ratio(self):
        mgr = make_manager(budget_ratio=0.4, min_budget=10)
        T = 200
        expected = max(10, int(200 * 0.4))   # 80
        assert mgr.compute_budget(T) == expected

    def test_compute_budget_fractional_floor(self):
        mgr = make_manager(budget_ratio=0.3, min_budget=10)
        # 101 * 0.3 = 30.3 → floor to 30
        assert mgr.compute_budget(101) == 30


# ---------------------------------------------------------------------------
# 3. Compute budget — minimum
# ---------------------------------------------------------------------------

class TestComputeBudgetMin:
    def test_compute_budget_min_respected(self):
        mgr = make_manager(budget_ratio=0.01, min_budget=64)
        # 1% of 10 = 0, but min_budget clamps to 64; T=10 so result = 10
        assert mgr.compute_budget(10) == 10  # can't exceed T

    def test_compute_budget_min_when_ratio_small(self):
        mgr = make_manager(budget_ratio=0.01, min_budget=64)
        # 1% of 1000 = 10 → min_budget(64) wins
        assert mgr.compute_budget(1000) == 64

    def test_compute_budget_never_exceeds_current_len(self):
        mgr = make_manager(budget_ratio=2.0, min_budget=9999)
        assert mgr.compute_budget(50) == 50


# ---------------------------------------------------------------------------
# 4. should_evict — over budget
# ---------------------------------------------------------------------------

class TestShouldEvictOverBudget:
    def test_should_evict_over_budget(self):
        mgr = make_manager(budget_ratio=0.3, min_budget=10)
        T = 200
        keys, values = make_kv(4, T, 16)
        state = mgr.new_state(keys, values)
        # budget = max(10, 60) = 60 < 200 → should evict
        assert mgr.should_evict(state) is True


# ---------------------------------------------------------------------------
# 5. should_evict — under budget
# ---------------------------------------------------------------------------

class TestShouldEvictUnderBudget:
    def test_should_evict_under_budget(self):
        mgr = make_manager(budget_ratio=0.9, min_budget=10)
        T = 20
        keys, values = make_kv(4, T, 16)
        state = mgr.new_state(keys, values)
        # budget = max(10, 18) = 18 < 20 → should still evict
        # Use a very high ratio so budget == T
        mgr2 = make_manager(budget_ratio=1.0, min_budget=10)
        state2 = mgr2.new_state(keys, values)
        assert mgr2.should_evict(state2) is False

    def test_should_evict_at_exactly_min_budget(self):
        mgr = make_manager(budget_ratio=0.01, min_budget=20)
        T = 20
        keys, values = make_kv(4, T, 16)
        state = mgr.new_state(keys, values)
        # budget = max(20, 0) = 20 = T → no eviction needed
        assert mgr.should_evict(state) is False


# ---------------------------------------------------------------------------
# 6. evict — reduces size
# ---------------------------------------------------------------------------

class TestEvictReducesSize:
    def test_evict_reduces_size(self):
        mgr = make_manager(budget_ratio=0.3, min_budget=10, recent_window=5)
        T = 100
        keys, values = make_kv(4, T, 16)
        state = mgr.new_state(keys, values)
        # Give uniform attention so eviction is based on budget only.
        state = mgr.update_scores(state, uniform_attn(4, T))
        state = mgr.evict(state)
        budget = mgr.compute_budget(T)
        assert len(state.kept_positions) <= budget


# ---------------------------------------------------------------------------
# 7. evict — always keeps recent window
# ---------------------------------------------------------------------------

class TestEvictKeepsRecent:
    def test_evict_keeps_recent(self):
        rw = 10
        mgr = make_manager(budget_ratio=0.3, min_budget=5, recent_window=rw)
        T = 100
        keys, values = make_kv(4, T, 16)
        state = mgr.new_state(keys, values)
        # Give near-zero attention to recent tokens to ensure score doesn't
        # rescue them — they should be kept by the recency rule regardless.
        attn = torch.zeros(4, T)
        attn[:, :T - rw] = 1.0 / (T - rw)  # all mass on older tokens
        state = mgr.update_scores(state, attn)
        state = mgr.evict(state)
        # The last `rw` original positions must all be retained.
        original_recent = set(range(T - rw, T))
        kept_set = set(state.kept_positions)
        assert original_recent.issubset(kept_set), (
            f"Recent positions not all kept. Missing: {original_recent - kept_set}"
        )


# ---------------------------------------------------------------------------
# 8. evict — keeps high-score older tokens
# ---------------------------------------------------------------------------

class TestEvictKeepsHighScore:
    def test_evict_keeps_high_score(self):
        rw = 5
        mgr = make_manager(budget_ratio=0.3, min_budget=5, recent_window=rw)
        T = 50
        n_heads = 4
        head_dim = 16
        keys, values = make_kv(n_heads, T, head_dim)
        state = mgr.new_state(keys, values)

        # Give all attention mass to token 0 (oldest) → it should survive.
        attn = torch.zeros(n_heads, T)
        attn[:, 0] = 1.0
        state = mgr.update_scores(state, attn)
        state = mgr.evict(state)

        budget = mgr.compute_budget(T)
        # If budget allows any old tokens at all, token 0 must be kept.
        n_recent = min(rw, budget)
        n_keep_old = budget - n_recent
        if n_keep_old > 0:
            assert 0 in state.kept_positions, (
                "Highest-scored token 0 was incorrectly evicted."
            )


# ---------------------------------------------------------------------------
# 9. evict — KV shape consistent with kept count
# ---------------------------------------------------------------------------

class TestEvictKVShapeConsistent:
    def test_evict_kv_shape_consistent(self):
        n_heads, T, head_dim = 4, 80, 16
        mgr = make_manager(n_heads=n_heads, head_dim=head_dim,
                           budget_ratio=0.3, min_budget=10, recent_window=8)
        keys, values = make_kv(n_heads, T, head_dim)
        state = mgr.new_state(keys, values)
        state = mgr.update_scores(state, uniform_attn(n_heads, T))
        state = mgr.evict(state)

        k = len(state.kept_positions)
        assert state.keys.shape   == (n_heads, k, head_dim)
        assert state.values.shape == (n_heads, k, head_dim)
        assert state.attn_scores_acc.shape == (n_heads, k)


# ---------------------------------------------------------------------------
# 10. update_scores — monotonically accumulates
# ---------------------------------------------------------------------------

class TestUpdateScoresAccumulates:
    def test_update_scores_accumulates(self):
        n_heads, T, head_dim = 4, 40, 16
        mgr = make_manager(budget_ratio=1.0, min_budget=1)  # no eviction
        keys, values = make_kv(n_heads, T, head_dim)
        state = mgr.new_state(keys, values)

        attn = uniform_attn(n_heads, T)
        prev_acc = state.attn_scores_acc.clone()

        for _ in range(5):
            state = mgr.update_scores(state, attn)
            assert (state.attn_scores_acc >= prev_acc).all(), (
                "Accumulated scores decreased after update."
            )
            prev_acc = state.attn_scores_acc.clone()


# ---------------------------------------------------------------------------
# 11. new_state — zero initial scores
# ---------------------------------------------------------------------------

class TestNewStateZeroScores:
    def test_new_state_zero_scores(self):
        n_heads, T, head_dim = 4, 60, 16
        mgr = make_manager()
        keys, values = make_kv(n_heads, T, head_dim)
        state = mgr.new_state(keys, values)

        assert state.attn_scores_acc.shape == (n_heads, T)
        assert (state.attn_scores_acc == 0).all()
        assert state.eviction_count == 0
        assert state.kept_positions == list(range(T))


# ---------------------------------------------------------------------------
# 12. efficiency_stats — ratio in (0, 1]
# ---------------------------------------------------------------------------

class TestEfficiencyStatsRatio:
    def test_efficiency_stats_ratio(self):
        n_heads, T, head_dim = 4, 100, 16
        mgr = make_manager(n_heads=n_heads, head_dim=head_dim,
                           budget_ratio=0.3, min_budget=10, recent_window=5)
        keys, values = make_kv(n_heads, T, head_dim)
        state = mgr.new_state(keys, values)
        state = mgr.update_scores(state, uniform_attn(n_heads, T))
        state = mgr.evict(state)

        stats = mgr.efficiency_stats(state, original_len=T)
        assert 0 < stats["cache_size_ratio"] <= 1.0
        assert stats["evictions"] == 1
        assert stats["memory_saved"] >= 0.0

    def test_efficiency_stats_no_eviction(self):
        n_heads, T, head_dim = 4, 20, 16
        mgr = make_manager(n_heads=n_heads, head_dim=head_dim,
                           budget_ratio=1.0, min_budget=1)
        keys, values = make_kv(n_heads, T, head_dim)
        state = mgr.new_state(keys, values)
        stats = mgr.efficiency_stats(state, original_len=T)
        assert stats["cache_size_ratio"] == pytest.approx(1.0)
        assert stats["memory_saved"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 13. No eviction below budget
# ---------------------------------------------------------------------------

class TestNoEvictionBelowBudget:
    def test_no_eviction_below_budget(self):
        """evict() on an already-within-budget cache must return it unchanged."""
        n_heads, T, head_dim = 4, 10, 16
        # budget_ratio=1.0 → budget = T; no eviction should occur.
        mgr = make_manager(budget_ratio=1.0, min_budget=1, recent_window=2)
        keys, values = make_kv(n_heads, T, head_dim)
        state = mgr.new_state(keys, values)
        state = mgr.update_scores(state, uniform_attn(n_heads, T))

        original_positions = list(state.kept_positions)
        state_after = mgr.evict(state)

        assert state_after.kept_positions == original_positions
        assert state_after.eviction_count == 0  # unchanged — not incremented


# ---------------------------------------------------------------------------
# 14. Eviction count increments
# ---------------------------------------------------------------------------

class TestEvictionCountIncrements:
    def test_eviction_count_increments(self):
        n_heads, T, head_dim = 4, 100, 16
        mgr = make_manager(budget_ratio=0.5, min_budget=10, recent_window=5)
        keys, values = make_kv(n_heads, T, head_dim)
        state = mgr.new_state(keys, values)

        for expected_count in range(1, 4):
            # Replenish attention weights matching current cache size.
            cur_T = len(state.kept_positions)
            state = mgr.update_scores(state, uniform_attn(n_heads, cur_T))
            if mgr.should_evict(state):
                state = mgr.evict(state)
                assert state.eviction_count == expected_count, (
                    f"Expected eviction_count={expected_count}, "
                    f"got {state.eviction_count}"
                )
            else:
                # Cache converged — stop early.
                break


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline(self):
        """n_heads=4, head_dim=16, T=100, budget_ratio=0.3.

        Run: new_state → update_scores x 5 → evict.
        Verify: kept <= budget, shapes consistent, eviction_count == 1.
        """
        n_heads  = 4
        head_dim = 16
        T        = 100
        cfg = AdaptiveKVConfig(
            n_layers=2,
            n_heads=n_heads,
            head_dim=head_dim,
            max_seq_len=512,
            budget_ratio=0.3,
            min_budget=10,
            recent_window=8,
        )
        mgr = AdaptiveKVEvictionManager(cfg)

        keys   = torch.randn(n_heads, T, head_dim)
        values = torch.randn(n_heads, T, head_dim)

        # Initialise.
        state = mgr.new_state(keys, values)
        assert state.attn_scores_acc.sum() == 0

        # Simulate 5 decoding steps; each step the query attends to all T tokens.
        for step in range(5):
            # Softmax-like weights (positive, sum to 1 per head).
            raw = torch.rand(n_heads, T)
            attn = raw / raw.sum(dim=-1, keepdim=True)
            state = mgr.update_scores(state, attn)

        assert (state.attn_scores_acc > 0).all(), (
            "Scores should be positive after 5 update steps."
        )

        # Evict.
        assert mgr.should_evict(state)
        state = mgr.evict(state)

        budget = mgr.compute_budget(T)
        kept   = len(state.kept_positions)

        assert kept <= budget, f"kept={kept} > budget={budget}"
        assert state.keys.shape   == (n_heads, kept, head_dim)
        assert state.values.shape == (n_heads, kept, head_dim)
        assert state.attn_scores_acc.shape == (n_heads, kept)
        assert state.eviction_count == 1

        # Efficiency stats sanity.
        stats = mgr.efficiency_stats(state, original_len=T)
        assert 0 < stats["cache_size_ratio"] <= 1.0
        assert stats["memory_saved"] > 0
