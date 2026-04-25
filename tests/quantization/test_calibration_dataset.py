"""Tests for src/quantization/calibration_dataset.py — ≥28 tests, stdlib-only."""

from __future__ import annotations

import pytest

from src.quantization.calibration_dataset import (
    CalibrationSample,
    CalibrationDataset,
    CALIBRATION_DATASET_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

def make_sample(text: str = "hello", tokens: list[int] | None = None, source: str = "") -> CalibrationSample:
    if tokens is None:
        tokens = [1, 2, 3]
    return CalibrationSample(text=text, tokens=tokens, source=source)


def populated_ds(n: int = 5, max_samples: int = 64) -> CalibrationDataset:
    ds = CalibrationDataset(max_samples=max_samples)
    for i in range(n):
        ds.add(CalibrationSample(text=f"sample {i}", tokens=list(range(i + 1))))
    return ds


# ---------------------------------------------------------------------------
# CalibrationSample
# ---------------------------------------------------------------------------

class TestCalibrationSample:
    def test_fields_stored(self):
        s = CalibrationSample(text="hi", tokens=[1, 2], source="wiki")
        assert s.text == "hi"
        assert s.tokens == [1, 2]
        assert s.source == "wiki"

    def test_default_source_empty(self):
        s = CalibrationSample(text="x", tokens=[])
        assert s.source == ""

    def test_frozen_text(self):
        s = CalibrationSample(text="abc", tokens=[1])
        with pytest.raises((AttributeError, TypeError)):
            s.text = "xyz"  # type: ignore[misc]

    def test_frozen_tokens(self):
        s = CalibrationSample(text="abc", tokens=[1])
        with pytest.raises((AttributeError, TypeError)):
            s.tokens = [2]  # type: ignore[misc]

    def test_frozen_source(self):
        s = CalibrationSample(text="abc", tokens=[1], source="a")
        with pytest.raises((AttributeError, TypeError)):
            s.source = "b"  # type: ignore[misc]

    def test_equality(self):
        s1 = CalibrationSample(text="a", tokens=[1, 2])
        s2 = CalibrationSample(text="a", tokens=[1, 2])
        assert s1 == s2


# ---------------------------------------------------------------------------
# CalibrationDataset — basics
# ---------------------------------------------------------------------------

class TestCalibrationDatasetBasics:
    def test_empty_len(self):
        ds = CalibrationDataset()
        assert len(ds) == 0

    def test_add_increases_len(self):
        ds = CalibrationDataset()
        ds.add(make_sample())
        assert len(ds) == 1

    def test_add_multiple(self):
        ds = CalibrationDataset()
        for _ in range(5):
            ds.add(make_sample())
        assert len(ds) == 5

    def test_max_samples_enforced(self):
        ds = CalibrationDataset(max_samples=2)
        ds.add(make_sample())
        ds.add(make_sample())
        with pytest.raises(ValueError):
            ds.add(make_sample())

    def test_getitem(self):
        ds = CalibrationDataset()
        s = make_sample(text="unique")
        ds.add(s)
        assert ds[0] == s

    def test_getitem_index(self):
        ds = populated_ds(3)
        assert ds[1].text == "sample 1"

    def test_iter_all_samples(self):
        ds = populated_ds(4)
        collected = list(ds)
        assert len(collected) == 4

    def test_iter_order_preserved(self):
        ds = CalibrationDataset()
        texts = ["alpha", "beta", "gamma"]
        for t in texts:
            ds.add(CalibrationSample(text=t, tokens=[]))
        assert [s.text for s in ds] == texts

    def test_default_max_samples(self):
        ds = CalibrationDataset()
        assert ds._max_samples == 512


# ---------------------------------------------------------------------------
# CalibrationDataset — token_counts
# ---------------------------------------------------------------------------

class TestTokenCounts:
    def test_token_counts_empty(self):
        ds = CalibrationDataset()
        assert ds.token_counts() == []

    def test_token_counts_lengths(self):
        ds = populated_ds(3)
        # sample 0 → [0] (1 tok), sample 1 → [0,1] (2 tok), sample 2 → [0,1,2] (3 tok)
        assert ds.token_counts() == [1, 2, 3]

    def test_token_counts_type(self):
        ds = populated_ds(2)
        counts = ds.token_counts()
        assert isinstance(counts, list)
        for c in counts:
            assert isinstance(c, int)


# ---------------------------------------------------------------------------
# CalibrationDataset — stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_empty_raises(self):
        ds = CalibrationDataset()
        with pytest.raises((ValueError, ZeroDivisionError)):
            ds.stats()

    def test_stats_count(self):
        ds = populated_ds(4)
        assert ds.stats()["count"] == 4

    def test_stats_max_tokens(self):
        # sample i has (i+1) tokens → max is 5 for n=5
        ds = populated_ds(5)
        assert ds.stats()["max_tokens"] == 5

    def test_stats_min_tokens(self):
        ds = populated_ds(5)
        assert ds.stats()["min_tokens"] == 1

    def test_stats_mean_tokens(self):
        ds = populated_ds(4)
        # token counts: [1, 2, 3, 4] → mean = 2.5
        assert ds.stats()["mean_tokens"] == pytest.approx(2.5)

    def test_stats_single_sample(self):
        ds = CalibrationDataset()
        ds.add(CalibrationSample(text="x", tokens=[10, 20, 30]))
        s = ds.stats()
        assert s["count"] == 1
        assert s["mean_tokens"] == 3.0
        assert s["max_tokens"] == 3
        assert s["min_tokens"] == 3


# ---------------------------------------------------------------------------
# CalibrationDataset — subsample
# ---------------------------------------------------------------------------

class TestSubsample:
    def test_subsample_size(self):
        ds = populated_ds(10)
        sub = ds.subsample(4)
        assert len(sub) == 4

    def test_subsample_returns_calibration_dataset(self):
        ds = populated_ds(10)
        sub = ds.subsample(3)
        assert isinstance(sub, CalibrationDataset)

    def test_subsample_reproducible_same_seed(self):
        ds = populated_ds(20)
        sub1 = ds.subsample(5, seed=42)
        sub2 = ds.subsample(5, seed=42)
        assert [s.text for s in sub1] == [s.text for s in sub2]

    def test_subsample_different_seeds_differ(self):
        ds = populated_ds(20)
        sub1 = ds.subsample(10, seed=0)
        sub2 = ds.subsample(10, seed=99)
        # With high probability two random draws of 10 from 20 will differ
        assert [s.text for s in sub1] != [s.text for s in sub2]

    def test_subsample_n_larger_than_dataset(self):
        ds = populated_ds(5)
        sub = ds.subsample(100)
        assert len(sub) == 5

    def test_subsample_zero(self):
        ds = populated_ds(5)
        sub = ds.subsample(0)
        assert len(sub) == 0

    def test_subsample_samples_are_from_original(self):
        ds = populated_ds(10)
        original_texts = {s.text for s in ds}
        sub = ds.subsample(5)
        for s in sub:
            assert s.text in original_texts


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_exists(self):
        assert isinstance(CALIBRATION_DATASET_REGISTRY, dict)

    def test_default_key_present(self):
        assert "default" in CALIBRATION_DATASET_REGISTRY

    def test_default_maps_to_class(self):
        assert CALIBRATION_DATASET_REGISTRY["default"] is CalibrationDataset

    def test_registry_instantiable(self):
        cls = CALIBRATION_DATASET_REGISTRY["default"]
        obj = cls()
        assert isinstance(obj, CalibrationDataset)
