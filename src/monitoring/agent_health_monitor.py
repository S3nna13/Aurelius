"""Agent health monitor: heartbeat tracking, timeout detection, thread-safe."""
from __future__ import annotations

import threading
import time


class AgentHealthMonitor:
    def __init__(self, check_interval_sec: float = 30.0) -> None:
        self._check_interval_sec = check_interval_sec
        self._agents: dict[str, dict[str, float | None]] = {}
        self._lock = threading.Lock()

    def register_agent(self, agent_id: str, heartbeat_timeout_sec: float = 60.0) -> None:
        with self._lock:
            self._agents[agent_id] = {
                "heartbeat_timeout_sec": heartbeat_timeout_sec,
                "last_seen": None,
            }

    def record_heartbeat(self, agent_id: str) -> None:
        with self._lock:
            if agent_id not in self._agents:
                raise KeyError(f"Agent {agent_id!r} is not registered")
            self._agents[agent_id]["last_seen"] = time.time()

    def check_health(self) -> dict[str, str]:
        now = time.time()
        with self._lock:
            result: dict[str, str] = {}
            for agent_id, state in self._agents.items():
                last_seen = state["last_seen"]
                timeout = state["heartbeat_timeout_sec"]
                if last_seen is None:
                    result[agent_id] = "unknown"
                elif now - last_seen > timeout:
                    result[agent_id] = "unhealthy"
                else:
                    result[agent_id] = "healthy"
            return result

    def get_unhealthy(self) -> list[str]:
        health = self.check_health()
        return [agent_id for agent_id, status in health.items() if status == "unhealthy"]

    def remove_agent(self, agent_id: str) -> None:
        with self._lock:
            if agent_id not in self._agents:
                raise KeyError(f"Agent {agent_id!r} is not registered")
            del self._agents[agent_id]


AGENT_HEALTH_MONITOR_REGISTRY: dict[str, AgentHealthMonitor] = {}
DEFAULT_AGENT_HEALTH_MONITOR = AgentHealthMonitor()
AGENT_HEALTH_MONITOR_REGISTRY["default"] = DEFAULT_AGENT_HEALTH_MONITOR
