"""Aurelius memory surface: episodic, working, and semantic memory."""

__all__ = [
    "MemoryEntry", "EpisodicMemory",
    "WorkingMemory", "WORKING_MEMORY",
    "MemoryIndex", "MEMORY_INDEX",
    "RelationType", "Concept", "Relation", "SemanticMemory",
    "ConsolidationPolicy", "ConsolidationResult", "MemoryConsolidator",
    "MEMORY_REGISTRY",
    # Cycle-146 long-term memory deepening (Park et al. 2303.17580)
    "LTMEntry", "LongTermMemory", "LONG_TERM_MEMORY",
    "RetrievalResult", "MemoryRetriever", "MEMORY_RETRIEVER",
]
from .episodic_memory import MemoryEntry, EpisodicMemory
from .working_memory import WorkingMemory, WORKING_MEMORY
from .memory_index import MemoryIndex, MEMORY_INDEX
from .semantic_memory import RelationType, Concept, Relation, SemanticMemory
from .memory_consolidation import ConsolidationPolicy, ConsolidationResult, MemoryConsolidator

MEMORY_REGISTRY: dict[str, object] = {
    "episodic": EpisodicMemory(),
    "working": WORKING_MEMORY,
    "index": MEMORY_INDEX,
    "semantic": SemanticMemory(),
    "consolidator": MemoryConsolidator(),
}

# --- Cycle-146 long-term memory deepening (Park et al. 2303.17580) -----------
from .long_term_memory import LTMEntry, LongTermMemory, LONG_TERM_MEMORY  # noqa: F401
from .memory_retriever import RetrievalResult, MemoryRetriever, MEMORY_RETRIEVER  # noqa: F401
MEMORY_REGISTRY.update({"ltm": LONG_TERM_MEMORY, "retriever": MEMORY_RETRIEVER})
