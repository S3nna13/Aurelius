"""Tests for TIES model merging (arXiv 2306.01708)."""
from __future__ import annotations

import pytest
import torch

from src.compression.dare_merge import MergeResult
from src.compression.ties_merge import TIESConfig, TIESMerger


def _sd(val: float = 0.0, seed: int | None = None) -> dict[str, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
        return {"w1": torch.randn(4, 4), "w2": torch.randn(4, 4)}
    return {
        "w1": torch.full((4, 4), val),
        "w2": torch.full((4, 4), val),
    }


# --- TIESConfig ---

def test_config_defaults():
    cfg = TIESConfig()
    assert cfg.trim_ratio == 0.2
    assert cfg.elect_sign == "majority"


def test_config_magnitude_weighted():
    cfg = TIESConfig(elect_sign="magnitude_weighted")
    assert cfg.elect_sign == "magnitude_weighted"


def test_invalid_elect_sign_raises():
    with pytest.raises(ValueError):
        TIESMerger(TIESConfig(elect_sign="bad_method"))


# --- compute_task_vectors ---

def test_task_vectors_zero_for_identical():
    base = _sd(seed=0)
    merger = TIESMerger()
    tvs = merger.compute_task_vectors(base, [base, base])
    for tv in tvs:
        for v in tv.values():
            assert v.abs().max().item() == pytest.approx(0.0)


def test_task_vectors_count_matches_models():
    base = _sd(seed=0)
    models = [_sd(seed=i) for i in range(3)]
    merger = TIESMerger()
    tvs = merger.compute_task_vectors(base, models)
    assert len(tvs) == 3


# --- trim ---

def test_trim_zeros_small_values():
    base = {"w1": torch.zeros(4, 4)}
    ft = {"w1": torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 4)}
    merger = TIESMerger(TIESConfig(trim_ratio=0.5))
    tv = merger.compute_task_vectors(base, [ft])
    trimmed = merger.trim(tv)
    assert (trimmed[0]["w1"] != 0).sum().item() >= 8


def test_trim_ratio_zero_keeps_all():
    base = _sd(0.0)
    ft = _sd(1.0)
    merger = TIESMerger(TIESConfig(trim_ratio=0.0))
    tv = merger.compute_task_vectors(base, [ft])
    trimmed = merger.trim(tv)
    assert trimmed[0]["w1"].nonzero().shape[0] == 16


# --- elect_sign ---

def test_elect_sign_majority_positive():
    tv_pos = {"w1": torch.ones(4, 4)}
    tv_neg = {"w1": -torch.ones(4, 4) * 0.1}
    merger = TIESMerger(TIESConfig(elect_sign="majority"))
    signs = merger.elect_sign([tv_pos, tv_pos, tv_neg])
    assert (signs["w1"] > 0).all()


def test_elect_sign_magnitude_weighted():
    tv_pos = {"w1": torch.ones(4, 4) * 10}
    tv_neg = {"w1": -torch.ones(4, 4)}
    merger = TIESMerger(TIESConfig(elect_sign="magnitude_weighted"))
    signs = merger.elect_sign([tv_pos, tv_neg])
    assert (signs["w1"] > 0).all()


def test_elect_sign_values_are_pm1():
    tvs = [{"w1": torch.randn(4, 4)} for _ in range(3)]
    merger = TIESMerger()
    signs = merger.elect_sign(tvs)
    uniq = signs["w1"].abs().unique()
    assert (uniq == 1.0).all()


# --- disjoint_merge ---

def test_disjoint_merge_agrees_with_elected():
    tv1 = {"w1": torch.ones(4, 4)}
    tv2 = {"w1": torch.ones(4, 4)}
    elected = {"w1": torch.ones(4, 4)}
    merger = TIESMerger()
    result = merger.disjoint_merge([tv1, tv2], elected)
    assert result["w1"].allclose(torch.ones(4, 4))


# --- merge (full pipeline) ---

def test_merge_returns_merge_result():
    base = _sd(seed=0)
    models = [_sd(seed=1), _sd(seed=2)]
    merger = TIESMerger()
    result = merger.merge(base, models)
    assert isinstance(result, MergeResult)


def test_merge_n_params_correct():
    base = _sd(seed=0)
    ft = _sd(seed=1)
    merger = TIESMerger()
    result = merger.merge(base, [ft])
    expected = sum(t.numel() for t in base.values())
    assert result.n_params_merged == expected


def test_merge_drop_rate_recorded():
    base = _sd(seed=0)
    ft = _sd(seed=1)
    cfg = TIESConfig(trim_ratio=0.3)
    merger = TIESMerger(cfg)
    result = merger.merge(base, [ft])
    assert result.drop_rate_applied == 0.3


def test_merge_empty_models_raises():
    base = _sd(seed=0)
    merger = TIESMerger()
    with pytest.raises(ValueError):
        merger.merge(base, [])


def test_merge_state_dict_keys_match_base():
    base = _sd(seed=0)
    ft = _sd(seed=1)
    merger = TIESMerger()
    result = merger.merge(base, [ft])
    assert set(result.merged_state_dict.keys()) == set(base.keys())


def test_merge_magnitude_weighted_elect():
    base = _sd(seed=0)
    ft = _sd(seed=1)
    merger = TIESMerger(TIESConfig(elect_sign="magnitude_weighted"))
    result = merger.merge(base, [ft])
    assert isinstance(result, MergeResult)
