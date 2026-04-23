"""Aurelius reasoning surface: chain-of-thought, tree-of-thought, scratchpad."""
__all__ = [
    "ChainOfThought", "COT_REGISTRY",
    "ThoughtNode", "ToTPlanner", "TOT_PLANNER",
    "Scratchpad", "SCRATCHPAD",
    "REASONING_REGISTRY",
]
from .chain_of_thought import ChainOfThought, COT_REGISTRY
from .tot_planner import ThoughtNode, ToTPlanner, TOT_PLANNER
from .scratchpad import Scratchpad, SCRATCHPAD

REASONING_REGISTRY: dict[str, object] = {
    "cot": COT_REGISTRY,
    "tot": TOT_PLANNER,
    "scratchpad": SCRATCHPAD,
}
