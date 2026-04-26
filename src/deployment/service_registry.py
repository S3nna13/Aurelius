"""Service discovery registry for microservice coordination."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServiceInstance:
    name: str
    host: str
    port: int
    health_url: str = "/healthz"
    metadata: dict[str, Any] = field(default_factory=dict)
    _last_heartbeat: float = 0.0

    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_healthy(self, timeout: float = 30.0) -> bool:
        if self._last_heartbeat == 0.0:
            return True
        return time.monotonic() - self._last_heartbeat < timeout


@dataclass
class ServiceRegistry:
    services: dict[str, list[ServiceInstance]] = field(default_factory=dict, repr=False)

    def register(self, instance: ServiceInstance) -> None:
        self.services.setdefault(instance.name, []).append(instance)

    def unregister(self, name: str, host: str, port: int) -> None:
        instances = self.services.get(name, [])
        self.services[name] = [i for i in instances if not (i.host == host and i.port == port)]

    def discover(self, name: str) -> list[ServiceInstance]:
        return self.services.get(name, [])

    def heartbeat(self, name: str, host: str, port: int) -> bool:
        for instance in self.services.get(name, []):
            if instance.host == host and instance.port == port:
                instance._last_heartbeat = time.monotonic()
                return True
        return False

    def healthy_instances(self, name: str) -> list[ServiceInstance]:
        return [i for i in self.discover(name) if i.is_healthy()]


SERVICE_REGISTRY = ServiceRegistry()
