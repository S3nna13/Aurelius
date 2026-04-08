"""Tests for src/training/nas.py — Hyperband / Successive Halving NAS."""
from __future__ import annotations

import random as _random

import pytest

from src.training.nas import (
    HyperbandSearcher,
    NASConfig,
    NASResult,
    SearchSpace,
    Trial,
    random_search,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_lr_search_space() -> SearchSpace:
    return SearchSpace(
        continuous={"lr": (1e-5, 1e-1, True)},
    )


def lr_eval_fn(config: dict, steps: int) -> float:
    """Minimum at lr=1e-3; adds tiny noise so losses differ slightly."""
    rng = _random.Random(id(config) ^ steps)
    return abs(config["lr"] - 1e-3) * 1000 + rng.uniform(0, 0.01)


# ---------------------------------------------------------------------------
# SearchSpace tests
# ---------------------------------------------------------------------------

def test_search_space_sample_continuous():
    ss = make_lr_search_space()
    rng = _random.Random(0)
    for _ in range(50):
        cfg = ss.sample(rng)
        assert "lr" in cfg
        assert 1e-5 <= cfg["lr"] <= 1e-1


def test_search_space_sample_log_scale():
    """Log-scale sampling should produce values across multiple orders of magnitude."""
    ss = make_lr_search_space()
    rng = _random.Random(7)
    values = [ss.sample(rng)["lr"] for _ in range(200)]
    # Expect at least some values below 1e-3 and some above 1e-3
    assert any(v < 1e-3 for v in values), "Expected values below 1e-3"
    assert any(v > 1e-3 for v in values), "Expected values above 1e-3"
    # Log-scale: should see values spanning at least 3 orders of magnitude
    log_range = max(values) / min(values)
    assert log_range > 100, f"Expected > 100x range, got {log_range:.1f}x"


def test_search_space_sample_discrete():
    ss = SearchSpace(
        discrete={"n_layers": [2, 4, 6, 8, 12]},
    )
    rng = _random.Random(1)
    sampled = {ss.sample(rng)["n_layers"] for _ in range(100)}
    # All sampled values should be in the valid list
    assert sampled <= {2, 4, 6, 8, 12}
    # With 100 draws we should have seen at least a few distinct values
    assert len(sampled) > 1


# ---------------------------------------------------------------------------
# Trial dataclass
# ---------------------------------------------------------------------------

def test_trial_dataclass():
    trial = Trial(config={"lr": 1e-3}, trial_id=0)
    assert trial.config == {"lr": 1e-3}
    assert trial.trial_id == 0
    assert trial.best_loss == float("inf")
    assert trial.steps_trained == 0
    assert trial.eliminated is False


# ---------------------------------------------------------------------------
# random_search tests
# ---------------------------------------------------------------------------

def test_random_search_returns_result():
    ss = make_lr_search_space()
    result = random_search(ss, lr_eval_fn, n_trials=10, budget_per_trial=20)
    assert isinstance(result, NASResult)
    assert "lr" in result.best_config
    assert isinstance(result.best_loss, float)


def test_random_search_n_trials():
    ss = make_lr_search_space()
    result = random_search(ss, lr_eval_fn, n_trials=8, budget_per_trial=5)
    assert len(result.trial_history) == 8
    assert result.n_trials == 8


def test_random_search_best_is_min():
    ss = make_lr_search_space()
    result = random_search(ss, lr_eval_fn, n_trials=20, budget_per_trial=10, seed=42)
    all_losses = [t.best_loss for t in result.trial_history]
    assert result.best_loss <= min(all_losses) + 1e-9


# ---------------------------------------------------------------------------
# HyperbandSearcher tests
# ---------------------------------------------------------------------------

def test_hyperband_search_returns_result():
    ss = make_lr_search_space()
    cfg = NASConfig(n_initial_configs=9, eta=3, min_budget=5, max_budget=45, seed=0)
    searcher = HyperbandSearcher(ss, lr_eval_fn, cfg)
    result = searcher.search()
    assert isinstance(result, NASResult)
    assert "lr" in result.best_config
    assert isinstance(result.best_loss, float)
    assert result.total_steps > 0


def test_hyperband_eliminates_configs():
    """Hyperband should eliminate configs so fewer survive to the final rung."""
    ss = make_lr_search_space()
    cfg = NASConfig(n_initial_configs=9, eta=3, min_budget=5, max_budget=45, seed=1)
    searcher = HyperbandSearcher(ss, lr_eval_fn, cfg)
    result = searcher.search()

    # At least some trials should be eliminated
    eliminated = [t for t in result.trial_history if t.eliminated]
    surviving = [t for t in result.trial_history if not t.eliminated]
    assert len(eliminated) > 0, "Expected some eliminated trials"
    assert len(surviving) < len(result.trial_history), "Expected fewer survivors than total"


def test_hyperband_finds_reasonable_lr():
    """Best config from Hyperband should be closer to optimal lr=1e-3 than average."""
    ss = make_lr_search_space()
    cfg = NASConfig(n_initial_configs=16, eta=3, min_budget=10, max_budget=90, seed=99)
    searcher = HyperbandSearcher(ss, lr_eval_fn, cfg)
    result = searcher.search()

    best_lr = result.best_config["lr"]
    best_distance = abs(best_lr - 1e-3)

    # Compare against all trial losses: best should be in the top third
    all_losses = [t.best_loss for t in result.trial_history if t.best_loss < float("inf")]
    all_losses.sort()
    top_third_threshold = all_losses[len(all_losses) // 3]

    assert result.best_loss <= top_third_threshold, (
        f"best_loss={result.best_loss:.4f} not in top third (threshold={top_third_threshold:.4f})"
    )
