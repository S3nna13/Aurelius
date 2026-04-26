"""Tests for src/compression/weight_sharing.py (≥28 tests)."""

from __future__ import annotations

import dataclasses

import pytest

from src.compression.weight_sharing import (
    WEIGHT_SHARING_REGISTRY,
    SharingGroup,
    WeightSharing,
    WeightSharingConfig,
)

# ---------------------------------------------------------------------------
# WeightSharingConfig
# ---------------------------------------------------------------------------


class TestWeightSharingConfig:
    def test_defaults(self):
        cfg = WeightSharingConfig()
        assert cfg.num_clusters == 256
        assert cfg.bits == 8

    def test_custom(self):
        cfg = WeightSharingConfig(num_clusters=16, bits=4)
        assert cfg.num_clusters == 16
        assert cfg.bits == 4

    def test_frozen(self):
        cfg = WeightSharingConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.num_clusters = 99  # type: ignore[misc]

    def test_frozen_bits(self):
        cfg = WeightSharingConfig(bits=4)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.bits = 8  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SharingGroup
# ---------------------------------------------------------------------------


class TestSharingGroup:
    def test_fields_stored(self):
        sg = SharingGroup(group_id="g1", param_names=["w1", "w2"], shared_value=[0.5, -0.5])
        assert sg.group_id == "g1"
        assert sg.param_names == ["w1", "w2"]
        assert sg.shared_value == [0.5, -0.5]

    def test_mutable_dataclass(self):
        sg = SharingGroup(group_id="g2", param_names=[], shared_value=[])
        sg.group_id = "g2_updated"
        assert sg.group_id == "g2_updated"


# ---------------------------------------------------------------------------
# WeightSharing.cluster_weights
# ---------------------------------------------------------------------------


class TestClusterWeights:
    def _ws(self, k: int = 4) -> WeightSharing:
        return WeightSharing(WeightSharingConfig(num_clusters=k, bits=8))

    def test_length_matches_input(self):
        ws = self._ws(4)
        weights = [float(i) for i in range(20)]
        result = ws.cluster_weights(weights)
        assert len(result) == len(weights)

    def test_assignments_in_range(self):
        ws = self._ws(4)
        weights = [float(i) for i in range(20)]
        result = ws.cluster_weights(weights)
        assert all(0 <= a < 4 for a in result)

    def test_single_weight(self):
        ws = self._ws(4)
        result = ws.cluster_weights([3.14])
        assert len(result) == 1
        assert result[0] == 0

    def test_empty_weights(self):
        ws = self._ws(4)
        assert ws.cluster_weights([]) == []

    def test_deterministic_same_seed(self):
        ws = self._ws(8)
        weights = [float(i) * 0.1 for i in range(50)]
        r1 = ws.cluster_weights(weights, seed=7)
        r2 = ws.cluster_weights(weights, seed=7)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        ws = self._ws(8)
        weights = list(range(100))
        r1 = ws.cluster_weights(weights, seed=0)
        r2 = ws.cluster_weights(weights, seed=999)
        # Not guaranteed to differ, but usually will with 8 clusters over 100 spread values
        # Just ensure both are valid
        assert len(r1) == len(r2) == 100

    def test_all_same_values_one_unique_centroid(self):
        ws = self._ws(4)
        weights = [5.0] * 20
        result = ws.cluster_weights(weights)
        # All should map to the same cluster
        assert len(set(result)) == 1

    def test_k_capped_at_weight_count(self):
        ws = WeightSharing(WeightSharingConfig(num_clusters=256, bits=8))
        weights = [1.0, 2.0, 3.0]
        result = ws.cluster_weights(weights)
        assert len(result) == 3
        assert all(0 <= a < 3 for a in result)

    def test_cluster_count_at_most_num_clusters(self):
        ws = self._ws(4)
        weights = [float(i) for i in range(100)]
        result = ws.cluster_weights(weights)
        assert max(result) < 4


# ---------------------------------------------------------------------------
# WeightSharing.reconstruct
# ---------------------------------------------------------------------------


class TestReconstruct:
    def test_replaces_with_centroid(self):
        ws = WeightSharing()
        centroids = [0.0, 1.0, 2.0]
        assignments = [0, 1, 2, 0, 1]
        result = ws.reconstruct(assignments, centroids)
        assert result == [0.0, 1.0, 2.0, 0.0, 1.0]

    def test_empty(self):
        ws = WeightSharing()
        assert ws.reconstruct([], [1.0, 2.0]) == []

    def test_all_same_centroid(self):
        ws = WeightSharing()
        centroids = [3.14]
        result = ws.reconstruct([0, 0, 0, 0], centroids)
        assert result == [3.14, 3.14, 3.14, 3.14]

    def test_length_preserved(self):
        ws = WeightSharing()
        centroids = [1.0, 2.0]
        assignments = [0, 1, 0, 0, 1, 1]
        result = ws.reconstruct(assignments, centroids)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# WeightSharing.compression_ratio
# ---------------------------------------------------------------------------


class TestCompressionRatio:
    def test_ratio_below_one_for_large_count(self):
        ws = WeightSharing(WeightSharingConfig(num_clusters=256, bits=8))
        # original_bits=32 per weight, many weights → ratio < 1
        ratio = ws.compression_ratio(original_bits=32 * 10_000, weight_count=10_000)
        assert ratio < 1.0

    def test_formula(self):
        ws = WeightSharing(WeightSharingConfig(bits=8))
        # bits_needed = 100 * 8 / 8 = 100; original_bits = 3200
        ratio = ws.compression_ratio(original_bits=3200, weight_count=100)
        assert abs(ratio - 100 / 3200) < 1e-9

    def test_bits_4(self):
        ws = WeightSharing(WeightSharingConfig(bits=4))
        ratio = ws.compression_ratio(original_bits=3200, weight_count=100)
        expected = (100 * 4 / 8) / 3200
        assert abs(ratio - expected) < 1e-9


# ---------------------------------------------------------------------------
# WeightSharing.create_group
# ---------------------------------------------------------------------------


class TestCreateGroup:
    def test_group_id_stored(self):
        ws = WeightSharing(WeightSharingConfig(num_clusters=4))
        sg = ws.create_group("layer0", ["w.0", "w.1"], [1.0, 2.0, 3.0, 4.0])
        assert sg.group_id == "layer0"

    def test_param_names_stored(self):
        ws = WeightSharing(WeightSharingConfig(num_clusters=4))
        sg = ws.create_group("g", ["a", "b", "c"], [0.1, 0.2, 0.3])
        assert sg.param_names == ["a", "b", "c"]

    def test_shared_value_is_list_of_floats(self):
        ws = WeightSharing(WeightSharingConfig(num_clusters=4))
        sg = ws.create_group("g", ["p"], [1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(sg.shared_value, list)
        assert all(isinstance(v, float) for v in sg.shared_value)

    def test_returns_sharing_group(self):
        ws = WeightSharing(WeightSharingConfig(num_clusters=2))
        sg = ws.create_group("g", [], [1.0, 2.0, 3.0])
        assert isinstance(sg, SharingGroup)

    def test_shared_value_nonempty_for_nonempty_weights(self):
        ws = WeightSharing(WeightSharingConfig(num_clusters=4))
        sg = ws.create_group("g", [], [1.0, 2.0])
        assert len(sg.shared_value) > 0


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in WEIGHT_SHARING_REGISTRY

    def test_registry_default_is_class(self):
        assert WEIGHT_SHARING_REGISTRY["default"] is WeightSharing

    def test_registry_instantiable(self):
        cls = WEIGHT_SHARING_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, WeightSharing)
