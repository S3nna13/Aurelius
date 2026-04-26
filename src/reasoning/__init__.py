"""Aurelius reasoning surface: chain-of-thought, tree-of-thought, scratchpad, MCTS."""

__all__ = [
    "ChainOfThought",
    "COT_REGISTRY",
    "ThoughtNode",
    "ToTPlanner",
    "TOT_PLANNER",
    "Scratchpad",
    "SCRATCHPAD",
    "MCTSNode",
    "MCTSReasoner",
    "MCTS_REASONER",
    "ChainStrategy",
    "ChainStep",
    "ReasoningChainManager",
    "CHAIN_MANAGER_REGISTRY",
    "DEFAULT_CHAIN_MANAGER",
    "REASONING_REGISTRY",
    # Cycle-146 step verification + beam search (Lightman et al. 2305.20050)
    "VerificationLabel",
    "StepScore",
    "StepVerifier",
    "STEP_VERIFIER",
    "BeamHypothesis",
    "BeamSearchReasoner",
    "BEAM_SEARCH_REASONER",
]
from .chain_of_thought import COT_REGISTRY, ChainOfThought
from .mcts_reasoner import MCTS_REASONER, MCTSNode, MCTSReasoner
from .reasoning_chain_manager import (
    CHAIN_MANAGER_REGISTRY,
    DEFAULT_CHAIN_MANAGER,
    ChainStep,
    ChainStrategy,
    ReasoningChainManager,
)
from .scratchpad import SCRATCHPAD, Scratchpad
from .tot_planner import TOT_PLANNER, ThoughtNode, ToTPlanner

REASONING_REGISTRY: dict[str, object] = {
    "cot": COT_REGISTRY,
    "tot": TOT_PLANNER,
    "scratchpad": SCRATCHPAD,
    "mcts": MCTS_REASONER,
    "chain_manager": DEFAULT_CHAIN_MANAGER,
}

# --- Cycle-146 step verification + beam search (Lightman et al. 2305.20050) --
from .beam_search_reasoner import (  # noqa: F401
    BEAM_SEARCH_REASONER,
    BeamHypothesis,
    BeamSearchReasoner,
)
from .step_verifier import STEP_VERIFIER, StepScore, StepVerifier, VerificationLabel  # noqa: F401

REASONING_REGISTRY.update({"step_verifier": STEP_VERIFIER, "beam": BEAM_SEARCH_REASONER})

# --- Cycle-147 reasoning deepening (Wang 2022, Chen 2022, Zhou 2022) -----------
from .least_to_most import L2M_REGISTRY, L2MConfig, L2MResult, LeastToMost, SubProblem  # noqa: F401
from .program_of_thought import POT_REGISTRY, PoTConfig, PoTResult, ProgramOfThought  # noqa: F401
from .self_consistency import (  # noqa: F401
    SELF_CONSISTENCY_REGISTRY,
    ConsistencyResult,
    SelfConsistency,
    SelfConsistencyConfig,
)

REASONING_REGISTRY.update(
    {"self_consistency": SelfConsistency, "pot": ProgramOfThought, "l2m": LeastToMost}
)
