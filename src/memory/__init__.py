"""Aurelius memory surface: episodic, working, and semantic memory."""

__all__ = [
    "MemoryEntry",
    "EpisodicMemory",
    "WorkingMemory",
    "WORKING_MEMORY",
    "MemoryIndex",
    "MEMORY_INDEX",
    "RelationType",
    "Concept",
    "Relation",
    "SemanticMemory",
    "ConsolidationPolicy",
    "ConsolidationResult",
    "MemoryConsolidator",
    "MEMORY_REGISTRY",
    # Cycle-146 long-term memory deepening (Park et al. 2303.17580)
    "LTMEntry",
    "LongTermMemory",
    "LONG_TERM_MEMORY",
    "RetrievalResult",
    "MemoryRetriever",
    "MEMORY_RETRIEVER",
]
from .episodic_memory import EpisodicMemory, MemoryEntry
from .memory_consolidation import (
    ConsolidationPolicy,
    ConsolidationResult,
    MemoryConsolidator,
)
from .memory_index import MEMORY_INDEX, MemoryIndex
from .semantic_memory import Concept, Relation, RelationType, SemanticMemory
from .working_memory import WORKING_MEMORY, WorkingMemory

MEMORY_REGISTRY: dict[str, object] = {
    "episodic": EpisodicMemory(),
    "working": WORKING_MEMORY,
    "index": MEMORY_INDEX,
    "semantic": SemanticMemory(),
    "consolidator": MemoryConsolidator(),
}

# --- Cycle-146 long-term memory deepening (Park et al. 2303.17580) -----------
from .long_term_memory import LONG_TERM_MEMORY, LongTermMemory, LTMEntry  # noqa: F401
from .memory_retriever import (
    MEMORY_RETRIEVER,
    MemoryRetriever,
    RetrievalResult,
)  # noqa: F401

MEMORY_REGISTRY.update({"ltm": LONG_TERM_MEMORY, "retriever": MEMORY_RETRIEVER})

# --- Cycle-210 layered memory (GenericAgent-inspired) ------------------------
from .layered_memory import (  # noqa: F401
    DEFAULT_LAYERED_MEMORY,
    LAYERED_MEMORY_REGISTRY,
    LayeredMemory,
    LayeredMemoryEntry,
    LayeredMemoryError,
    MemoryLayer,
)

# Register layered memory in the combined registry.
MEMORY_REGISTRY["layered"] = DEFAULT_LAYERED_MEMORY

# --- Cycle-210 progressive search (claude-mem-inspired) ----------------------
from .progressive_search import (  # noqa: F401
    DEFAULT_PROGRESSIVE_SEARCHER,
    PROGRESSIVE_SEARCH_REGISTRY,
    IndexEntry,
    ProgressiveSearcher,
    ProgressiveSearchError,
    SearchResult,
)

MEMORY_REGISTRY["progressive_search"] = DEFAULT_PROGRESSIVE_SEARCHER

from .manager import MemoryManager, MemoryQueryResult  # noqa: E402
from .unified_orchestrator import (  # noqa: E402
    InferenceDecision,
    InferenceResult,
    InferenceTier,
    UnifiedInferenceOrchestrator,
)
from .composite import (  # noqa: E402
    ArchitectureDecomposer,
    CompositeInferenceEngine,
    CompositeResult,
    SubTask,
)

__all__ += [
    "MemoryManager",
    "MemoryQueryResult",
]

__all__ += [
    "UnifiedInferenceOrchestrator",
    "InferenceDecision",
    "InferenceResult",
    "InferenceTier",
    "CompositeInferenceEngine",
    "ArchitectureDecomposer",
    "CompositeResult",
    "SubTask",
]

__all__ += [
    "ProgressiveSearcher",
    "IndexEntry",
    "SearchResult",
    "ProgressiveSearchError",
    "DEFAULT_PROGRESSIVE_SEARCHER",
    "PROGRESSIVE_SEARCH_REGISTRY",
]
