"""Federated/Edge/NAS: Federated Learning, Spiking NN, NAS, TinyML.

Papers: McMahan 2017, Konecny 2016, Tavanaei 2018, Zoph 2016, Warden 2020.
"""

from __future__ import annotations

import random
from typing import Any

from .foundational import MLP
from .registry import register


class FederatedAveraging:
    """Federated Averaging (McMahan et al. 2017)."""

    def __init__(self, model_factory: Any = None, n_clients: int = 10) -> None:
        self.global_model = model_factory() if model_factory else MLP([10, 10, 2])
        self.client_models = [
            model_factory() if model_factory else MLP([10, 10, 2]) for _ in range(n_clients)
        ]

    def aggregate(self, client_weights: list[list[list[list[float]]]]) -> None:
        n = len(client_weights)
        for layer in range(len(self.global_model.weights)):
            for neuron in range(len(self.global_model.weights[layer])):
                for w in range(len(self.global_model.weights[layer][neuron])):
                    avg_w = sum(cw[layer][neuron][w] for cw in client_weights) / n
                    self.global_model.weights[layer][neuron][w] = avg_w

    def distribute(self) -> None:
        for client_model in self.client_models:
            client_model.weights = [list(layer) for layer in self.global_model.weights]
            client_model.biases = list(self.global_model.biases)


register("edge.fedavg", FederatedAveraging)


class SpikingNeuron:
    """Leaky Integrate-and-Fire spiking neuron (Tavanaei et al. 2018)."""

    def __init__(self, threshold: float = 1.0, decay: float = 0.9, reset: float = 0.0) -> None:
        self.threshold = threshold
        self.decay = decay
        self.reset_potential = reset
        self.membrane = 0.0

    def forward(self, input_current: float) -> int:
        self.membrane = self.decay * self.membrane + input_current
        if self.membrane >= self.threshold:
            self.membrane = self.reset_potential
            return 1
        return 0

    def reset(self) -> None:
        self.membrane = 0.0


register("edge.spiking", SpikingNeuron)


class NASLayer:
    """Neural Architecture Search (Zoph & Le 2016). RL-based search."""

    def __init__(self, search_space: list[dict[str, Any]] | None = None) -> None:
        self.search_space = search_space or [
            {"type": "dense", "units": [32, 64, 128, 256]},
            {"type": "conv", "filters": [16, 32, 64], "kernel": [3, 5]},
            {"type": "lstm", "units": [32, 64, 128]},
        ]
        self._controller: list[float] = [random.gauss(0, 0.1) for _ in range(10)]

    def sample_architecture(self) -> list[dict[str, Any]]:
        arch: list[dict[str, Any]] = []
        for _ in range(random.randint(2, 5)):
            choice = random.choice(self.search_space)
            if choice["type"] == "dense":
                arch.append({"type": "dense", "units": random.choice(choice["units"])})
            elif choice["type"] == "conv":
                arch.append(
                    {
                        "type": "conv",
                        "filters": random.choice(choice["filters"]),
                        "kernel": random.choice(choice["kernel"]),
                    }
                )
        return arch


register("edge.nas", NASLayer)


class TinyMLModel:
    """TinyML: quantized, pruned model for edge deployment (Warden & Situnayake 2020)."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 16, n_classes: int = 3) -> None:
        self.W1 = [[random.gauss(0, 0.1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(n_classes)]
        self.b2 = [0.0] * n_classes
        # 8-bit quantization
        self.scale = 1.0

    def quantize(self) -> None:
        for layer in [self.W1, self.W2]:
            for i in range(len(layer)):
                for j in range(len(layer[i])):
                    layer[i][j] = round(layer[i][j] * 128) / 128.0

    def prune(self, threshold: float = 0.01) -> int:
        n_pruned = 0
        for layer in [self.W1, self.W2]:
            for i in range(len(layer)):
                for j in range(len(layer[i])):
                    if abs(layer[i][j]) < threshold:
                        layer[i][j] = 0.0
                        n_pruned += 1
        return n_pruned

    def forward(self, x: list[float]) -> list[float]:
        h = [
            max(0.0, sum(self.W1[i][j] * x[j] for j in range(len(x))) + self.b1[i])
            for i in range(len(self.W1))
        ]
        return [
            sum(self.W2[o][k] * h[k] for k in range(len(h))) + self.b2[o]
            for o in range(len(self.W2))
        ]


register("edge.tinyml", TinyMLModel)
