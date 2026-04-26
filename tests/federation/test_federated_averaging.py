"""Tests for federated averaging."""

from __future__ import annotations

import pytest

from src.federation.federated_averaging import FederatedAveraging


class TestFederatedAveraging:
    """Test FederatedAveraging weight aggregation."""

    def test_average_equal_weights(self):
        """Test averaging with equal client weights."""
        client_weights = [
            {"layer1": [[1.0, 2.0], [3.0, 4.0]]},
            {"layer1": [[5.0, 6.0], [7.0, 8.0]]},
        ]
        client_counts = [10, 10]

        result = FederatedAveraging().average(client_weights, client_counts)

        assert result["layer1"][0][0] == pytest.approx(3.0)

    def test_average_weighted_by_counts(self):
        """Test averaging weighted by sample counts."""
        client_weights = [
            {"layer1": [[1.0]]},
            {"layer1": [[9.0]]},
        ]
        client_counts = [1, 3]

        result = FederatedAveraging().average(client_weights, client_counts)

        assert result["layer1"][0][0] == pytest.approx(7.0)

    def test_empty_inputs(self):
        """Test empty client weights."""
        result = FederatedAveraging().average([], [0])
        assert result == {}
