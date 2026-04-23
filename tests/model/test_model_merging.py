"""Unit tests for :mod:`src.model.model_merging`."""

from __future__ import annotations

import pytest
import torch

from src.model.model_merging import (
    MergeError,
    MergeResult,
    MergeStrategy,
    ModelMerger,
    dare_merge,
    linear_merge,
    slerp_merge,
    ties_merge,
)


def _sd(**kw: torch.Tensor) -> dict:
    return dict(kw)


def test_linear_merge_two_states_shape_preserved() -> None:
    a = _sd(w=torch.ones(4, 3), b=torch.zeros(3))
    b = _sd(w=torch.zeros(4, 3), b=torch.ones(3))
    out = linear_merge([a, b])
    assert out["w"].shape == (4, 3)
    assert out["b"].shape == (3,)
    assert torch.allclose(out["w"], torch.full((4, 3), 0.5))
    assert torch.allclose(out["b"], torch.full((3,), 0.5))


def test_linear_weighted_sums_correctly() -> None:
    a = _sd(w=torch.ones(2))
    b = _sd(w=torch.full((2,), 3.0))
    out = linear_merge([a, b], weights=[1.0, 3.0])  # normalised → 0.25, 0.75
    expected = 0.25 * 1.0 + 0.75 * 3.0
    assert torch.allclose(out["w"], torch.full((2,), expected))


def test_slerp_t_zero_returns_a() -> None:
    a = _sd(w=torch.randn(8))
    b = _sd(w=torch.randn(8))
    out = slerp_merge(a, b, t=0.0)
    assert torch.allclose(out["w"], a["w"])


def test_slerp_t_one_returns_b() -> None:
    a = _sd(w=torch.randn(8))
    b = _sd(w=torch.randn(8))
    out = slerp_merge(a, b, t=1.0)
    assert torch.allclose(out["w"], b["w"])


def test_slerp_half_preserves_unit_norm_approximately() -> None:
    torch.manual_seed(0)
    x = torch.randn(32)
    y = torch.randn(32)
    x = x / torch.linalg.vector_norm(x)
    y = y / torch.linalg.vector_norm(y)
    a = _sd(w=x)
    b = _sd(w=y)
    out = slerp_merge(a, b, t=0.5)
    n = torch.linalg.vector_norm(out["w"])
    # SLERP on unit vectors preserves norm exactly (to float tolerance).
    assert abs(float(n) - 1.0) < 1e-4


def test_ties_trims_small_magnitude_entries() -> None:
    base = _sd(w=torch.zeros(10))
    delta = _sd(w=torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05, 10.0, 11.0, 12.0, 13.0, 14.0]))
    out = ties_merge(base, [delta], trim_ratio=0.5)
    # the 5 smallest entries should have been zeroed in the delta contribution
    for i in range(5):
        assert float(out["w"][i]) == pytest.approx(0.0)
    for i in range(5, 10):
        assert float(out["w"][i]) > 0.0


def test_ties_respects_sign_majority() -> None:
    base = _sd(w=torch.zeros(3))
    d1 = _sd(w=torch.tensor([1.0, 1.0, 1.0]))
    d2 = _sd(w=torch.tensor([1.0, 1.0, -5.0]))
    d3 = _sd(w=torch.tensor([1.0, -1.0, -5.0]))
    out = ties_merge(base, [d1, d2, d3], trim_ratio=0.0)
    # entry 0: all positive → +1.0
    assert float(out["w"][0]) == pytest.approx(1.0)
    # entry 1: two positive, one negative → positive mean = (1+1)/2 = 1.0
    assert float(out["w"][1]) == pytest.approx(1.0)
    # entry 2: two negative (-5,-5) vs one positive (1) → negative wins, mean -5
    assert float(out["w"][2]) == pytest.approx(-5.0)


def test_dare_drop_rate_zero_equals_base_plus_delta() -> None:
    base = _sd(w=torch.ones(5))
    delta = _sd(w=torch.full((5,), 2.0))
    out = dare_merge(base, delta, drop_rate=0.0)
    assert torch.allclose(out["w"], torch.full((5,), 3.0))


def test_dare_drop_rate_one_equals_base() -> None:
    base = _sd(w=torch.ones(5))
    delta = _sd(w=torch.full((5,), 2.0))
    out = dare_merge(base, delta, drop_rate=1.0)
    assert torch.allclose(out["w"], torch.ones(5))


def test_dare_seeded_deterministic() -> None:
    base = _sd(w=torch.zeros(64))
    delta = _sd(w=torch.arange(64, dtype=torch.float32))
    torch.manual_seed(1234)
    out_a = dare_merge(base, delta, drop_rate=0.5)
    torch.manual_seed(1234)
    out_b = dare_merge(base, delta, drop_rate=0.5)
    assert torch.allclose(out_a["w"], out_b["w"])
    # and rescaling preserves expected magnitude roughly
    torch.manual_seed(0)
    out_c = dare_merge(base, delta, drop_rate=0.5, scale_mode="rescale")
    # mean of rescaled delta should be close to mean of original delta for large N
    assert abs(float(out_c["w"].mean()) - float(delta["w"].mean())) < float(delta["w"].mean()) * 0.5


def test_key_mismatch_raises() -> None:
    a = _sd(w=torch.zeros(2))
    b = _sd(v=torch.zeros(2))
    with pytest.raises(MergeError):
        linear_merge([a, b])


def test_shape_mismatch_raises() -> None:
    a = _sd(w=torch.zeros(2))
    b = _sd(w=torch.zeros(3))
    with pytest.raises(MergeError):
        linear_merge([a, b])


def test_modelmerger_wrapper_all_strategies() -> None:
    a = _sd(w=torch.ones(4))
    b = _sd(w=torch.full((4,), 3.0))

    # LINEAR
    r = ModelMerger(MergeStrategy.LINEAR).merge([a, b], names=("A", "B"))
    assert isinstance(r, MergeResult)
    assert r.contributors == ("A", "B")
    assert torch.allclose(r.state_dict["w"], torch.full((4,), 2.0))

    # SLERP
    r = ModelMerger(MergeStrategy.SLERP, t=0.0).merge([a, b])
    assert torch.allclose(r.state_dict["w"], a["w"])

    # TIES
    base = _sd(w=torch.zeros(4))
    d1 = _sd(w=torch.tensor([1.0, 2.0, 3.0, 4.0]))
    r = ModelMerger(MergeStrategy.TIES, trim_ratio=0.0).merge([base, d1])
    assert torch.allclose(r.state_dict["w"], d1["w"])

    # DARE
    r = ModelMerger(MergeStrategy.DARE, drop_rate=0.0).merge([base, d1])
    assert torch.allclose(r.state_dict["w"], d1["w"])
    assert r.metadata["drop_rate"] == 0.0


def test_mergeresult_contributors_populated() -> None:
    a = _sd(w=torch.zeros(2))
    b = _sd(w=torch.ones(2))
    r = ModelMerger(MergeStrategy.LINEAR).merge([a, b])
    assert r.contributors == ("model_0", "model_1")
    assert r.strategy is MergeStrategy.LINEAR
    assert r.metadata["n_inputs"] == 2


def test_slerp_requires_valid_t() -> None:
    a = _sd(w=torch.ones(2))
    b = _sd(w=torch.zeros(2))
    with pytest.raises(MergeError):
        slerp_merge(a, b, t=1.5)


def test_complex_tensor_rejected() -> None:
    a = _sd(w=torch.ones(2, dtype=torch.complex64))
    b = _sd(w=torch.ones(2, dtype=torch.complex64))
    with pytest.raises(MergeError):
        linear_merge([a, b])
