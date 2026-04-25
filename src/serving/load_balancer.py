import random
from dataclasses import dataclass, field
from enum import Enum


class LBStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"


@dataclass
class BackendNode:
    name: str
    url: str
    weight: int = 1
    active_connections: int = 0
    healthy: bool = True


class LoadBalancer:
    def __init__(self, strategy: LBStrategy = LBStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self._nodes: list[BackendNode] = []
        self._rr_index: int = 0

    def add_backend(self, node: BackendNode) -> None:
        self._nodes.append(node)

    def remove_backend(self, name: str) -> None:
        self._nodes = [n for n in self._nodes if n.name != name]

    def next_backend(self) -> BackendNode | None:
        healthy = [n for n in self._nodes if n.healthy]
        if not healthy:
            return None

        if self.strategy == LBStrategy.ROUND_ROBIN:
            all_healthy_names = {n.name for n in healthy}
            start = self._rr_index % len(self._nodes)
            for i in range(len(self._nodes)):
                node = self._nodes[(start + i) % len(self._nodes)]
                if node.healthy:
                    self._rr_index = (self._nodes.index(node) + 1) % len(self._nodes)
                    return node
            return None

        elif self.strategy == LBStrategy.LEAST_CONNECTIONS:
            return min(healthy, key=lambda n: n.active_connections)

        elif self.strategy == LBStrategy.WEIGHTED_ROUND_ROBIN:
            total = sum(n.weight for n in healthy)
            r = random.uniform(0, total)
            cumulative = 0.0
            for node in healthy:
                cumulative += node.weight
                if r <= cumulative:
                    return node
            return healthy[-1]

        elif self.strategy == LBStrategy.RANDOM:
            return random.choice(healthy)

        return None

    def mark_connection_open(self, name: str) -> None:
        for n in self._nodes:
            if n.name == name:
                n.active_connections += 1
                return

    def mark_connection_closed(self, name: str) -> None:
        for n in self._nodes:
            if n.name == name:
                n.active_connections = max(0, n.active_connections - 1)
                return

    def mark_unhealthy(self, name: str) -> None:
        for n in self._nodes:
            if n.name == name:
                n.healthy = False
                return

    def mark_healthy(self, name: str) -> None:
        for n in self._nodes:
            if n.name == name:
                n.healthy = True
                return

    def healthy_count(self) -> int:
        return sum(1 for n in self._nodes if n.healthy)


LOAD_BALANCER_REGISTRY: dict[str, LBStrategy] = {s.value: s for s in LBStrategy}
