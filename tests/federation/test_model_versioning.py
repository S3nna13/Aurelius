"""Tests for src/federation/model_versioning.py"""

import torch

from src.federation.model_versioning import ModelVersion, ModelVersionRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fresh() -> ModelVersionRegistry:
    return ModelVersionRegistry()


def params(keys=("w1", "w2")) -> dict[str, torch.Tensor]:
    return {k: torch.randn(3, 3) for k in keys}


# ---------------------------------------------------------------------------
# ModelVersion dataclass
# ---------------------------------------------------------------------------


def test_model_version_auto_uuid():
    v1 = ModelVersion(round_number=1, param_hash="abc")
    v2 = ModelVersion(round_number=2, param_hash="def")
    assert v1.version_id != v2.version_id


def test_model_version_defaults():
    v = ModelVersion()
    assert v.metadata == {}
    assert v.round_number == 0


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def test_register_returns_version():
    reg = fresh()
    v = reg.register(1, params())
    assert isinstance(v, ModelVersion)
    assert v.round_number == 1
    assert len(v.param_hash) == 64  # sha256 hex


def test_register_deterministic_hash():
    """Same params produce same hash."""
    p = {"x": torch.tensor([1.0, 2.0, 3.0])}
    reg = fresh()
    v1 = reg.register(1, p)
    v2 = reg.register(2, p)
    assert v1.param_hash == v2.param_hash


def test_register_different_params_different_hash():
    reg = fresh()
    v1 = reg.register(1, {"x": torch.tensor([1.0])})
    v2 = reg.register(2, {"x": torch.tensor([2.0])})
    assert v1.param_hash != v2.param_hash


def test_register_stores_metadata():
    reg = fresh()
    v = reg.register(1, params(), metadata={"info": "test"})
    assert v.metadata["info"] == "test"


# ---------------------------------------------------------------------------
# get / get_by_round
# ---------------------------------------------------------------------------


def test_get_by_id():
    reg = fresh()
    v = reg.register(1, params())
    assert reg.get(v.version_id) is v


def test_get_nonexistent_returns_none():
    reg = fresh()
    assert reg.get("no-such-id") is None


def test_get_by_round():
    reg = fresh()
    v = reg.register(7, params())
    assert reg.get_by_round(7) is v


def test_get_by_round_nonexistent():
    reg = fresh()
    assert reg.get_by_round(99) is None


# ---------------------------------------------------------------------------
# list_versions
# ---------------------------------------------------------------------------


def test_list_versions_sorted():
    reg = fresh()
    reg.register(3, params())
    reg.register(1, params())
    reg.register(2, params())
    versions = reg.list_versions()
    rounds = [v.round_number for v in versions]
    assert rounds == [1, 2, 3]


def test_list_versions_empty():
    reg = fresh()
    assert reg.list_versions() == []


# ---------------------------------------------------------------------------
# diff_rounds
# ---------------------------------------------------------------------------


def test_diff_rounds_added():
    reg = fresh()
    reg.register(1, params(), metadata={"param_names": ["w1"]})
    reg.register(2, params(), metadata={"param_names": ["w1", "w2"]})
    diff = reg.diff_rounds(1, 2)
    assert "w2" in diff["added"]
    assert diff["removed"] == []


def test_diff_rounds_removed():
    reg = fresh()
    reg.register(1, params(), metadata={"param_names": ["w1", "w2"]})
    reg.register(2, params(), metadata={"param_names": ["w1"]})
    diff = reg.diff_rounds(1, 2)
    assert "w2" in diff["removed"]
    assert diff["added"] == []


def test_diff_rounds_missing_round():
    reg = fresh()
    reg.register(1, params(), metadata={"param_names": ["w1"]})
    diff = reg.diff_rounds(1, 99)
    # Round 99 doesn't exist, so all params from round 1 appear as removed
    assert "w1" in diff["removed"]
