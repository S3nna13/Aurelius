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
from .episodic_memory import MemoryEntry, EpisodicMemory
from .working_memory import WorkingMemory, WORKING_MEMORY
from .memory_index import MemoryIndex, MEMORY_INDEX
from .semantic_memory import RelationType, Concept, Relation, SemanticMemory
from .memory_consolidation import (
    ConsolidationPolicy,
    ConsolidationResult,
    MemoryConsolidator,
)

MEMORY_REGISTRY: dict[str, object] = {
    "episodic": EpisodicMemory(),
    "working": WORKING_MEMORY,
    "index": MEMORY_INDEX,
    "semantic": SemanticMemory(),
    "consolidator": MemoryConsolidator(),
}

# --- Cycle-146 long-term memory deepening (Park et al. 2303.17580) -----------
from .long_term_memory import LTMEntry, LongTermMemory, LONG_TERM_MEMORY  # noqa: F401
from .memory_retriever import (
    RetrievalResult,
    MemoryRetriever,
    MEMORY_RETRIEVER,
)  # noqa: F401

MEMORY_REGISTRY.update({"ltm": LONG_TERM_MEMORY, "retriever": MEMORY_RETRIEVER})

# --- Cycle-210 layered memory (GenericAgent-inspired) ------------------------
from .layered_memory import (  # noqa: F401
    LayeredMemory,
    LayeredMemoryEntry,
    MemoryLayer,
    LayeredMemoryError,
    DEFAULT_LAYERED_MEMORY,
    LAYERED_MEMORY_REGISTRY,
)

# Register layered memory in the combined registry.
MEMORY_REGISTRY["layered"] = DEFAULT_LAYERED_MEMORY

# --- Cycle-210 progressive search (claude-mem-inspired) ----------------------
from .progressive_search import (  # noqa: F401
    ProgressiveSearcher,
    IndexEntry,
    SearchResult,
    ProgressiveSearchError,
    DEFAULT_PROGRESSIVE_SEARCHER,
    PROGRESSIVE_SEARCH_REGISTRY,
)

MEMORY_REGISTRY["progressive_search"] = DEFAULT_PROGRESSIVE_SEARCHER

__all__ += [
    "LayeredMemory",
    "LayeredMemoryEntry",
    "MemoryLayer",
    "LayeredMemoryError",
    "DEFAULT_LAYERED_MEMORY",
    "LAYERED_MEMORY_REGISTRY",
    "ProgressiveSearcher",
    "IndexEntry",
    "SearchResult",
    "ProgressiveSearchError",
    "DEFAULT_PROGRESSIVE_SEARCHER",
    "PROGRESSIVE_SEARCH_REGISTRY",
]
