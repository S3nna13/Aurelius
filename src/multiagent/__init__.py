"""Aurelius multi-agent surface: orchestration, routing, and agent pool."""
__all__ = [
    "AgentSpec", "Orchestrator", "ORCHESTRATOR",
    "TaskRouter", "RoutingStrategy", "TASK_ROUTER",
    "AgentPool", "AGENT_POOL",
    "MULTIAGENT_REGISTRY",
]
from .orchestrator import AgentSpec, Orchestrator, ORCHESTRATOR
from .task_router import TaskRouter, RoutingStrategy, TASK_ROUTER
from .agent_pool import AgentPool, AGENT_POOL

MULTIAGENT_REGISTRY: dict[str, object] = {
    "orchestrator": ORCHESTRATOR,
    "router": TASK_ROUTER,
    "pool": AGENT_POOL,
}
