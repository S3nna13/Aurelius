"""Split learning protocol: model split between client and server.

The client computes forward activations up to a cut layer, sends the
"smash data" to the server, and receives gradients back. This module
provides a coordinator for orchestrating the protocol between multiple
clients and a shared server.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class SplitConfig:
    """Configuration for split learning."""

    cut_layer: int = 6
    client_lr: float = 1e-3
    server_lr: float = 1e-3
    num_clients: int = 3


def _new_batch_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass(frozen=True)
class SmashData:
    """Activations sent from client to server at the cut layer."""

    client_id: str
    layer_index: int
    activations: list[float]
    labels: list[int]
    batch_id: str = field(default_factory=_new_batch_id)


@dataclass(frozen=True)
class GradientPacket:
    """Gradient packet returned from server to client."""

    client_id: str
    batch_id: str
    gradients: list[float]


class SplitLearningCoordinator:
    """Coordinates split learning between multiple clients and one server."""

    def __init__(self, config: SplitConfig) -> None:
        self.config = config
        self._clients: list[str] = []
        self._smash: dict[tuple[str, str], SmashData] = {}

    def register_client(self, client_id: str) -> None:
        if client_id not in self._clients:
            self._clients.append(client_id)

    def receive_smash(self, data: SmashData) -> None:
        self._smash[(data.client_id, data.batch_id)] = data

    def compute_server_gradient(
        self, data: SmashData, server_fn: Callable[[list[float]], list[float]]
    ) -> GradientPacket:
        output = server_fn(data.activations)
        gradients = [o - a for o, a in zip(output, data.activations[: len(output)])]
        return GradientPacket(
            client_id=data.client_id,
            batch_id=data.batch_id,
            gradients=gradients,
        )

    def send_gradient(self, packet: GradientPacket) -> GradientPacket:
        """Stub for actual network send; returns the packet as-is."""
        return packet

    def client_ids(self) -> list[str]:
        return list(self._clients)

    def pending_batches(self) -> int:
        return len(self._smash)


SPLIT_LEARNING_REGISTRY = {"default": SplitLearningCoordinator}
