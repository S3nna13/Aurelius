"""Aurelius reasoning surface: chain-of-thought, tree-of-thought, scratchpad, MCTS."""
__all__ = [
    "ChainOfThought", "COT_REGISTRY",
    "ThoughtNode", "ToTPlanner", "TOT_PLANNER",
    "Scratchpad", "SCRATCHPAD",
    "MCTSNode", "MCTSReasoner", "MCTS_REASONER",
    "ChainStrategy", "ChainStep", "ReasoningChainManager",
    "CHAIN_MANAGER_REGISTRY", "DEFAULT_CHAIN_MANAGER",
    "REASONING_REGISTRY",
]
from .chain_of_thought import ChainOfThought, COT_REGISTRY
from .tot_planner import ThoughtNode, ToTPlanner, TOT_PLANNER
from .scratchpad import Scratchpad, SCRATCHPAD
from .mcts_reasoner import MCTSNode, MCTSReasoner, MCTS_REASONER
from .reasoning_chain_manager import (
    ChainStrategy, ChainStep, ReasoningChainManager,
    CHAIN_MANAGER_REGISTRY, DEFAULT_CHAIN_MANAGER,
)

REASONING_REGISTRY: dict[str, object] = {
    "cot": COT_REGISTRY,
    "tot": TOT_PLANNER,
    "scratchpad": SCRATCHPAD,
    "mcts": MCTS_REASONER,
    "chain_manager": DEFAULT_CHAIN_MANAGER,
}
