"""Tests for model_warmer.py."""

from __future__ import annotations

import threading

import pytest
import torch

from src.serving.model_warmer import ModelWarmer, WARMER_REGISTRY


def _make_pt_file(path: str) -> dict[str, torch.Tensor]:
    state = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
    torch.save(state, path)
    return state


def _make_safetensors_file(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import save_file

    state = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
    save_file(state, path)
    return state


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_init_defaults(self, tmp_path):
        pt = tmp_path / "model.pt"
        _make_pt_file(str(pt))
        warmer = ModelWarmer(str(pt))
        assert warmer._device == "cpu"

    def test_init_custom_device(self, tmp_path):
        pt = tmp_path / "model.pt"
        _make_pt_file(str(pt))
        warmer = ModelWarmer(str(pt), device="cuda")
        assert warmer._device == "cuda"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ModelWarmer("nonexistent_file.pt")

    def test_invalid_model_path_type(self):
        with pytest.raises(TypeError):
            ModelWarmer(123)  # type: ignore[arg-type]

    def test_invalid_device_type(self, tmp_path):
        pt = tmp_path / "model.pt"
        _make_pt_file(str(pt))
        with pytest.raises(TypeError):
            ModelWarmer(str(pt), device=123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Warm
# ---------------------------------------------------------------------------


class TestWarm:
    def test_warm_pt_returns_state_dict(self, tmp_path):
        pt = tmp_path / "model.pt"
        expected = _make_pt_file(str(pt))
        warmer = ModelWarmer(str(pt))
        state = warmer.warm()
        assert isinstance(state, dict)
        assert set(state.keys()) == set(expected.keys())
        for k in expected:
            assert torch.equal(state[k], expected[k])

    def test_warm_bin_returns_state_dict(self, tmp_path):
        bin_file = tmp_path / "model.bin"
        expected = _make_pt_file(str(bin_file))
        warmer = ModelWarmer(str(bin_file))
        state = warmer.warm()
        assert isinstance(state, dict)
        for k in expected:
            assert torch.equal(state[k], expected[k])

    def test_warm_safetensors_returns_state_dict(self, tmp_path):
        st = tmp_path / "model.safetensors"
        expected = _make_safetensors_file(str(st))
        warmer = ModelWarmer(str(st))
        state = warmer.warm()
        assert isinstance(state, dict)
        for k in expected:
            assert torch.equal(state[k], expected[k])

    def test_warm_idempotent(self, tmp_path):
        pt = tmp_path / "model.pt"
        _make_pt_file(str(pt))
        warmer = ModelWarmer(str(pt))
        s1 = warmer.warm()
        s2 = warmer.warm()
        assert s1 is s2


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestStateTransitions:
    def test_is_warmed_false_before_warm(self, tmp_path):
        pt = tmp_path / "model.pt"
        _make_pt_file(str(pt))
        warmer = ModelWarmer(str(pt))
        assert warmer.is_warmed() is False

    def test_is_warmed_true_after_warm(self, tmp_path):
        pt = tmp_path / "model.pt"
        _make_pt_file(str(pt))
        warmer = ModelWarmer(str(pt))
        warmer.warm()
        assert warmer.is_warmed() is True

    def test_unload_clears_state(self, tmp_path):
        pt = tmp_path / "model.pt"
        _make_pt_file(str(pt))
        warmer = ModelWarmer(str(pt))
        warmer.warm()
        assert warmer.is_warmed() is True
        warmer.unload()
        assert warmer.is_warmed() is False


# ---------------------------------------------------------------------------
# Unload
# ---------------------------------------------------------------------------


class TestUnload:
    def test_unload_reduces_memory_to_zero(self, tmp_path):
        pt = tmp_path / "model.pt"
        _make_pt_file(str(pt))
        warmer = ModelWarmer(str(pt))
        warmer.warm()
        assert warmer.memory_footprint_mb() > 0.0
        warmer.unload()
        assert warmer.memory_footprint_mb() == 0.0


# ---------------------------------------------------------------------------
# Memory footprint
# ---------------------------------------------------------------------------


class TestMemoryFootprint:
    def test_memory_footprint_mb_before_warm(self, tmp_path):
        pt = tmp_path / "model.pt"
        _make_pt_file(str(pt))
        warmer = ModelWarmer(str(pt))
        assert warmer.memory_footprint_mb() == 0.0

    def test_memory_footprint_mb_after_warm(self, tmp_path):
        pt = tmp_path / "model.pt"
        expected = _make_pt_file(str(pt))
        warmer = ModelWarmer(str(pt))
        warmer.warm()
        total_bytes = sum(t.numel() * t.element_size() for t in expected.values())
        expected_mb = total_bytes / (1024 * 1024)
        assert warmer.memory_footprint_mb() == pytest.approx(expected_mb, rel=1e-5)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_warmer_registry_is_dict(self):
        assert isinstance(WARMER_REGISTRY, dict)

    def test_manual_registration(self, tmp_path):
        pt = tmp_path / "model.pt"
        _make_pt_file(str(pt))
        warmer = ModelWarmer(str(pt))
        WARMER_REGISTRY["test_warmer"] = warmer
        assert WARMER_REGISTRY["test_warmer"] is warmer
        del WARMER_REGISTRY["test_warmer"]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_warm_and_unload(self, tmp_path):
        pt = tmp_path / "model.pt"
        _make_pt_file(str(pt))
        warmer = ModelWarmer(str(pt))

        errors: list[Exception] = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(50):
                try:
                    warmer.warm()
                    warmer.is_warmed()
                    warmer.memory_footprint_mb()
                    warmer.unload()
                except Exception as exc:
                    with lock:
                        errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
