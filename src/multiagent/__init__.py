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

# --- Message bus (cycle-197) -------------------------------------------------
from .message_bus import AgentMessage, MessageBus, MESSAGE_BUS  # noqa: F401
MULTIAGENT_REGISTRY["message_bus"] = MESSAGE_BUS

# --- Broadcast dispatcher (cycle-197) ----------------------------------------
from .broadcast_dispatcher import (  # noqa: F401
    BroadcastDispatcher,
    BROADCAST_DISPATCHER_REGISTRY,
    DEFAULT_BROADCAST_DISPATCHER,
)
MULTIAGENT_REGISTRY["broadcast_dispatcher"] = DEFAULT_BROADCAST_DISPATCHER

# --- Debate framework (additive, cycle-146, Du et al. 2023) --------------------
from .debate_framework import (  # noqa: F401
    DebateConfig,
    DebateRound,
    DebateSession,
    DEBATE_REGISTRY,
)
MULTIAGENT_REGISTRY["debate"] = DebateSession

# --- Consensus engine (additive, cycle-146) ------------------------------------
from .consensus_engine import (  # noqa: F401
    ConsensusConfig,
    ConsensusEngine,
    ConsensusMethod,
    ConsensusResult,
    CONSENSUS_REGISTRY,
)
MULTIAGENT_REGISTRY["consensus"] = ConsensusEngine

# --- Role-play manager (additive, cycle-146) -----------------------------------
from .role_play_manager import (  # noqa: F401
    AgentRole,
    RolePlayConfig,
    RolePlayManager,
    Utterance,
    ROLE_PLAY_REGISTRY,
)
MULTIAGENT_REGISTRY["role_play"] = RolePlayManager

# --- Consensus voter (cycle-201) ---------------------------------------------
from .consensus_voter import (  # noqa: F401
    Ballot,
    ConsensusResult,
    ConsensusVoter,
    TieBreak,
    Vote,
    CONSENSUS_VOTER_REGISTRY,
    DEFAULT_CONSENSUS_VOTER,
)
MULTIAGENT_REGISTRY["consensus_voter"] = DEFAULT_CONSENSUS_VOTER
