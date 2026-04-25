"""Aurelius reasoning surface: chain-of-thought, tree-of-thought, scratchpad, MCTS."""
__all__ = [
    "ChainOfThought", "COT_REGISTRY",
    "ThoughtNode", "ToTPlanner", "TOT_PLANNER",
    "Scratchpad", "SCRATCHPAD",
    "MCTSNode", "MCTSReasoner", "MCTS_REASONER",
    "ChainStrategy", "ChainStep", "ReasoningChainManager",
    "CHAIN_MANAGER_REGISTRY", "DEFAULT_CHAIN_MANAGER",
    "REASONING_REGISTRY",
    # Cycle-146 step verification + beam search (Lightman et al. 2305.20050)
    "VerificationLabel", "StepScore", "StepVerifier", "STEP_VERIFIER",
    "BeamHypothesis", "BeamSearchReasoner", "BEAM_SEARCH_REASONER",
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

# --- Cycle-146 step verification + beam search (Lightman et al. 2305.20050) --
from .step_verifier import VerificationLabel, StepScore, StepVerifier, STEP_VERIFIER  # noqa: F401
from .beam_search_reasoner import BeamHypothesis, BeamSearchReasoner, BEAM_SEARCH_REASONER  # noqa: F401
REASONING_REGISTRY.update({"step_verifier": STEP_VERIFIER, "beam": BEAM_SEARCH_REASONER})

# --- Cycle-147 reasoning deepening (Wang 2022, Chen 2022, Zhou 2022) -----------
from .self_consistency import SelfConsistency, SelfConsistencyConfig, ConsistencyResult, SELF_CONSISTENCY_REGISTRY  # noqa: F401
from .program_of_thought import ProgramOfThought, PoTConfig, PoTResult, POT_REGISTRY  # noqa: F401
from .least_to_most import LeastToMost, L2MConfig, L2MResult, SubProblem, L2M_REGISTRY  # noqa: F401
REASONING_REGISTRY.update({"self_consistency": SelfConsistency, "pot": ProgramOfThought, "l2m": LeastToMost})

# --- Cycle-158 GNN layers + trainer ------------------------------------
from .gnn_layer import GNNConfig, GCNLayer, GATLayer, GraphSAGELayer, GNNStack, GNN_PROFILER_REGISTRY  # noqa: F401
from .gnn_trainer import GNNTrainConfig, GNNTrainResult, GNNTrainer  # noqa: F401
REASONING_REGISTRY.update({"gnn": GNNStack})

__all__ = [
    "GNNConfig", "GCNLayer", "GATLayer", "GraphSAGELayer", "GNNStack", "GNN_PROFILER_REGISTRY",
    "GNNTrainConfig", "GNNTrainResult", "GNNTrainer",
] + __all__
