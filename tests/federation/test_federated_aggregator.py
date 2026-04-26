"""Tests for FederatedAggregator."""

from __future__ import annotations

import pytest
import torch

from src.federation.federated_aggregator import (
    FEDERATED_AGGREGATOR_REGISTRY,
    FederatedAggregator,
)


class TestFederatedAvg:
    """Test FedAvg strategy."""

    def test_fedavg_equal_weights(self):
        """Average with equal weights."""
        updates = [
            {"w": torch.tensor([1.0, 2.0])},
            {"w": torch.tensor([3.0, 4.0])},
        ]
        result = FederatedAggregator("fedavg").aggregate(updates)
        assert torch.allclose(result["w"], torch.tensor([2.0, 3.0]))

    def test_fedavg_custom_weights(self):
        """Average with custom client weights."""
        updates = [
            {"w": torch.tensor([1.0, 0.0])},
            {"w": torch.tensor([0.0, 1.0])},
        ]
        result = FederatedAggregator("fedavg").aggregate(updates, client_weights=[3.0, 1.0])
        assert torch.allclose(result["w"], torch.tensor([0.75, 0.25]))

    def test_fedavg_single_client(self):
        """Single client returns its own update."""
        updates = [{"w": torch.tensor([5.0, 10.0])}]
        result = FederatedAggregator("fedavg").aggregate(updates)
        assert torch.allclose(result["w"], torch.tensor([5.0, 10.0]))

    def test_fedavg_multiple_keys(self):
        """FedAvg handles multiple parameter keys."""
        updates = [
            {"a": torch.tensor([1.0]), "b": torch.tensor([[2.0, 3.0]])},
            {"a": torch.tensor([3.0]), "b": torch.tensor([[4.0, 5.0]])},
        ]
        result = FederatedAggregator("fedavg").aggregate(updates)
        assert torch.allclose(result["a"], torch.tensor([2.0]))
        assert torch.allclose(result["b"], torch.tensor([[3.0, 4.0]]))


class TestMedian:
    """Test coordinate-wise median strategy."""

    def test_median_odd_clients(self):
        """Median with odd number of clients."""
        updates = [
            {"w": torch.tensor([1.0, 10.0])},
            {"w": torch.tensor([2.0, 20.0])},
            {"w": torch.tensor([3.0, 30.0])},
        ]
        result = FederatedAggregator("median").aggregate(updates)
        assert torch.allclose(result["w"], torch.tensor([2.0, 20.0]))

    def test_median_even_clients(self):
        """Median with even number of clients returns lower middle per torch.median."""
        updates = [
            {"w": torch.tensor([1.0, 10.0])},
            {"w": torch.tensor([2.0, 20.0])},
            {"w": torch.tensor([3.0, 30.0])},
            {"w": torch.tensor([4.0, 40.0])},
        ]
        result = FederatedAggregator("median").aggregate(updates)
        assert torch.allclose(result["w"], torch.tensor([2.0, 20.0]))

    def test_median_single_client(self):
        """Median with one client returns that client's values."""
        updates = [{"w": torch.tensor([7.0, 8.0])}]
        result = FederatedAggregator("median").aggregate(updates)
        assert torch.allclose(result["w"], torch.tensor([7.0, 8.0]))

    def test_median_ignores_weights(self):
        """Median strategy ignores client_weights."""
        updates = [
            {"w": torch.tensor([1.0])},
            {"w": torch.tensor([3.0])},
            {"w": torch.tensor([5.0])},
        ]
        result = FederatedAggregator("median").aggregate(updates, client_weights=[100.0, 1.0, 1.0])
        assert torch.allclose(result["w"], torch.tensor([3.0]))


class TestTrimmedMean:
    """Test trimmed mean strategy."""

    def test_trimmed_mean_basic(self):
        """Trimmed mean removes min and max per coordinate."""
        updates = [
            {"w": torch.tensor([1.0, 10.0])},
            {"w": torch.tensor([2.0, 20.0])},
            {"w": torch.tensor([3.0, 30.0])},
        ]
        result = FederatedAggregator("trimmed_mean").aggregate(updates)
        assert torch.allclose(result["w"], torch.tensor([2.0, 20.0]))

    def test_trimmed_mean_four_clients(self):
        """Trimmed mean with four clients removes one min and one max."""
        updates = [
            {"w": torch.tensor([1.0, 100.0])},
            {"w": torch.tensor([2.0, 200.0])},
            {"w": torch.tensor([3.0, 300.0])},
            {"w": torch.tensor([4.0, 400.0])},
        ]
        result = FederatedAggregator("trimmed_mean").aggregate(updates)
        assert torch.allclose(result["w"], torch.tensor([2.5, 250.0]))

    def test_trimmed_mean_single_client(self):
        """Trimmed mean with one client falls back to mean."""
        updates = [{"w": torch.tensor([5.0])}]
        result = FederatedAggregator("trimmed_mean").aggregate(updates)
        assert torch.allclose(result["w"], torch.tensor([5.0]))

    def test_trimmed_mean_two_clients(self):
        """Trimmed mean with two clients falls back to mean."""
        updates = [
            {"w": torch.tensor([1.0, 4.0])},
            {"w": torch.tensor([3.0, 2.0])},
        ]
        result = FederatedAggregator("trimmed_mean").aggregate(updates)
        assert torch.allclose(result["w"], torch.tensor([2.0, 3.0]))


class TestValidation:
    """Test input validation and error handling."""

    def test_mismatched_keys(self):
        """Raises ValueError when client updates have different keys."""
        updates = [
            {"a": torch.tensor([1.0])},
            {"b": torch.tensor([2.0])},
        ]
        with pytest.raises(ValueError, match="mismatched keys"):
            FederatedAggregator("fedavg").aggregate(updates)

    def test_mismatched_shapes(self):
        """Raises ValueError when tensor shapes differ for the same key."""
        updates = [
            {"w": torch.tensor([1.0, 2.0])},
            {"w": torch.tensor([1.0])},
        ]
        with pytest.raises(ValueError, match="shape"):
            FederatedAggregator("fedavg").aggregate(updates)

    def test_weights_length_mismatch(self):
        """Raises ValueError when client_weights length does not match updates."""
        updates = [
            {"w": torch.tensor([1.0])},
            {"w": torch.tensor([2.0])},
        ]
        with pytest.raises(ValueError, match="client_weights length"):
            FederatedAggregator("fedavg").aggregate(updates, client_weights=[0.5])

    def test_empty_list(self):
        """Empty client_updates returns empty dict."""
        result = FederatedAggregator("fedavg").aggregate([])
        assert result == {}

    def test_unknown_strategy(self):
        """Unknown strategy raises ValueError at construction."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            FederatedAggregator("unknown")


class TestRegistry:
    """Test FEDERATED_AGGREGATOR_REGISTRY."""

    def test_registry_has_default(self):
        assert "default" in FEDERATED_AGGREGATOR_REGISTRY
        assert FEDERATED_AGGREGATOR_REGISTRY["default"] is FederatedAggregator
