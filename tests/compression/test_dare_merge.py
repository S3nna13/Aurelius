"""Tests for DARE model merging (arXiv 2311.03099)."""
from __future__ import annotations

import pytest
import torch

from src.compression.dare_merge import DAREConfig, DAREMerger, MergeResult


def _sd(seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {"w1": torch.randn(4, 4), "w2": torch.randn(4, 4)}


# --- DAREConfig ---

def test_config_defaults():
    cfg = DAREConfig()
    assert cfg.drop_rate == 0.9
    assert cfg.rescale is True
    assert cfg.seed == 42


def test_config_custom():
    cfg = DAREConfig(drop_rate=0.5, rescale=False, seed=7)
    assert cfg.drop_rate == 0.5
    assert cfg.rescale is False
    assert cfg.seed == 7


# --- MergeResult ---

def test_merge_result_fields():
    sd = _sd()
    mr = MergeResult(merged_state_dict=sd, n_params_merged=32, drop_rate_applied=0.9)
    assert mr.n_params_merged == 32
    assert mr.drop_rate_applied == 0.9


# --- DAREMerger init ---

def test_merger_default_config():
    m = DAREMerger()
    assert m.config.drop_rate == 0.9


def test_merger_custom_config():
    cfg = DAREConfig(drop_rate=0.5)
    m = DAREMerger(cfg)
    assert m.config.drop_rate == 0.5


# --- compute_task_vector ---

def test_task_vector_zero_for_identical():
    base = _sd(0)
    merger = DAREMerger()
    tv = merger.compute_task_vector(base, base)
    for v in tv.values():
        assert v.abs().max().item() == pytest.approx(0.0)


def test_task_vector_correct_delta():
    base = {"w1": torch.zeros(4, 4), "w2": torch.zeros(4, 4)}
    ft = {"w1": torch.ones(4, 4), "w2": torch.ones(4, 4) * 2}
    merger = DAREMerger()
    tv = merger.compute_task_vector(base, ft)
    assert tv["w1"].allclose(torch.ones(4, 4))
    assert tv["w2"].allclose(torch.ones(4, 4) * 2)


def test_task_vector_keys_match_base():
    base = _sd(0)
    ft = _sd(1)
    merger = DAREMerger()
    tv = merger.compute_task_vector(base, ft)
    assert set(tv.keys()) == set(base.keys())


# --- sparsify ---

def test_sparsify_reduces_nonzero():
    base = {"w1": torch.randn(4, 4), "w2": torch.randn(4, 4)}
    ft = {"w1": torch.randn(4, 4) + 5, "w2": torch.randn(4, 4) + 5}
    merger = DAREMerger(DAREConfig(drop_rate=0.9, rescale=False, seed=0))
    tv = merger.compute_task_vector(base, ft)
    sparse = merger.sparsify(tv)
    total = sum(t.numel() for t in sparse.values())
    nonzero = sum(t.nonzero().shape[0] for t in sparse.values())
    assert nonzero < total


def test_sparsify_no_rescale_preserves_magnitude_scale():
    tv = {"w1": torch.ones(4, 4)}
    merger = DAREMerger(DAREConfig(drop_rate=0.5, rescale=False, seed=0))
    sparse = merger.sparsify(tv)
    assert sparse["w1"].max().item() <= 1.0 + 1e-6


def test_sparsify_rescale_amplifies():
    tv = {"w1": torch.ones(100, 100)}
    merger_no = DAREMerger(DAREConfig(drop_rate=0.5, rescale=False, seed=1))
    merger_yes = DAREMerger(DAREConfig(drop_rate=0.5, rescale=True, seed=1))
    no_mean = merger_no.sparsify(tv)["w1"].abs().mean().item()
    yes_mean = merger_yes.sparsify(tv)["w1"].abs().mean().item()
    assert yes_mean > no_mean


def test_sparsify_seed_deterministic():
    tv = {"w1": torch.randn(4, 4)}
    cfg = DAREConfig(drop_rate=0.5, seed=99)
    m1 = DAREMerger(cfg)
    m2 = DAREMerger(cfg)
    s1 = m1.sparsify(tv)["w1"]
    s2 = m2.sparsify(tv)["w1"]
    assert s1.allclose(s2)


# --- merge ---

def test_merge_returns_merge_result():
    base = _sd(0)
    ft = _sd(1)
    merger = DAREMerger(DAREConfig(drop_rate=0.5))
    result = merger.merge(base, [ft])
    assert isinstance(result, MergeResult)


def test_merge_n_params_correct():
    base = _sd(0)
    ft = _sd(1)
    merger = DAREMerger()
    result = merger.merge(base, [ft])
    expected = sum(t.numel() for t in base.values())
    assert result.n_params_merged == expected


def test_merge_drop_rate_recorded():
    base = _sd(0)
    ft = _sd(1)
    cfg = DAREConfig(drop_rate=0.7)
    merger = DAREMerger(cfg)
    result = merger.merge(base, [ft])
    assert result.drop_rate_applied == 0.7


def test_merge_equal_weights_default():
    base = {"w1": torch.zeros(4, 4)}
    ft1 = {"w1": torch.ones(4, 4) * 2}
    ft2 = {"w1": torch.ones(4, 4) * 2}
    merger = DAREMerger(DAREConfig(drop_rate=0.0, rescale=False, seed=0))
    result = merger.merge(base, [ft1, ft2])
    assert result.merged_state_dict["w1"].allclose(torch.ones(4, 4) * 2)


def test_merge_empty_finetuned_raises():
    base = _sd(0)
    merger = DAREMerger()
    with pytest.raises(ValueError):
        merger.merge(base, [])


def test_merge_weight_mismatch_raises():
    base = _sd(0)
    ft = _sd(1)
    merger = DAREMerger()
    with pytest.raises(ValueError):
        merger.merge(base, [ft, ft], weights=[0.5])


def test_merge_zero_drop_identity():
    base = {"w1": torch.zeros(4, 4)}
    ft = {"w1": torch.ones(4, 4)}
    merger = DAREMerger(DAREConfig(drop_rate=0.0, rescale=False, seed=0))
    result = merger.merge(base, [ft])
    assert result.merged_state_dict["w1"].allclose(torch.ones(4, 4))
