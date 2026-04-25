"""Federated averaging for model weight synchronization.

Aurelius LLM Project — clean-room impl per FedAvg (McMahan 2017).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class WeightAverager(Protocol):
    """Protocol for federated averaging strategies."""
    
    def average(self, client_weights: list[dict], client_counts: list[int]) -> dict: ...


@dataclass
class FederatedAveraging:
    """Federated Averaging (FedAvg) weight aggregation.
    
    Aggregates client model weights proportional to client dataset sizes.
    """
    
    def average(self, client_weights: list[dict], client_counts: list[int]) -> dict:
        """Average weights weighted by client sample counts.
        
        Args:
            client_weights: List of model weight dicts from clients
            client_counts: Number of samples on each client
            
        Returns:
            Aggregated weight dict
        """
        total = sum(client_counts)
        if total == 0:
            return client_weights[0] if client_weights else {}
        
        averaged = {}
        for key in client_weights[0]:
            arr = np.asarray(client_weights[0][key], dtype=np.float64)
            weighted_sum = np.zeros_like(arr, dtype=np.float64)
            for weights, count in zip(client_weights, client_counts):
                if key in weights:
                    weighted_sum += np.asarray(weights[key], dtype=np.float64) * (count / total)
            averaged[key] = weighted_sum
        return averaged