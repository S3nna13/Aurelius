"""Aurelius multi-agent surface: orchestration, routing, and agent pool."""

__all__ = [
    "AgentSpec",
    "Orchestrator",
    "ORCHESTRATOR",
    "TaskRouter",
    "RoutingStrategy",
    "TASK_ROUTER",
    "AgentPool",
    "AGENT_POOL",
    "MULTIAGENT_REGISTRY",
]
from .agent_pool import AGENT_POOL, AgentPool
from .orchestrator import ORCHESTRATOR, AgentSpec, Orchestrator
from .task_router import TASK_ROUTER, RoutingStrategy, TaskRouter

MULTIAGENT_REGISTRY: dict[str, object] = {
    "orchestrator": ORCHESTRATOR,
    "router": TASK_ROUTER,
    "pool": AGENT_POOL,
}

# --- Message bus (cycle-197) -------------------------------------------------
from .message_bus import MESSAGE_BUS, AgentMessage, MessageBus  # noqa: F401

MULTIAGENT_REGISTRY["message_bus"] = MESSAGE_BUS

# --- Broadcast dispatcher (cycle-197) ----------------------------------------
from .broadcast_dispatcher import (  # noqa: F401
    BROADCAST_DISPATCHER_REGISTRY,
    DEFAULT_BROADCAST_DISPATCHER,
    BroadcastDispatcher,
)

MULTIAGENT_REGISTRY["broadcast_dispatcher"] = DEFAULT_BROADCAST_DISPATCHER

# --- Debate framework (additive, cycle-146, Du et al. 2023) --------------------
from .debate_framework import (  # noqa: F401
    DEBATE_REGISTRY,
    DebateConfig,
    DebateRound,
    DebateSession,
)

MULTIAGENT_REGISTRY["debate"] = DebateSession

# --- Consensus engine (additive, cycle-146) ------------------------------------
from .consensus_engine import (  # noqa: F401
    CONSENSUS_REGISTRY,
    ConsensusConfig,
    ConsensusEngine,
    ConsensusMethod,
    ConsensusResult,
)

MULTIAGENT_REGISTRY["consensus"] = ConsensusEngine

# --- Role-play manager (additive, cycle-146) -----------------------------------
from .role_play_manager import (  # noqa: F401
    ROLE_PLAY_REGISTRY,
    AgentRole,
    RolePlayConfig,
    RolePlayManager,
    Utterance,
)

MULTIAGENT_REGISTRY["role_play"] = RolePlayManager

# --- Consensus voter (cycle-201) ---------------------------------------------
from .consensus_voter import (  # noqa: F401
    CONSENSUS_VOTER_REGISTRY,
    DEFAULT_CONSENSUS_VOTER,
    Ballot,
    ConsensusVoter,
    TieBreak,
    Vote,
)

MULTIAGENT_REGISTRY["consensus_voter"] = DEFAULT_CONSENSUS_VOTER
