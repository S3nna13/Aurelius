"""Aurelius memory surface: episodic, working, and semantic memory."""

__all__ = [
    "MemoryEntry", "EpisodicMemory",
    "WorkingMemory", "WORKING_MEMORY",
    "MemoryIndex", "MEMORY_INDEX",
    "MEMORY_REGISTRY",
]
from .episodic_memory import MemoryEntry, EpisodicMemory
from .working_memory import WorkingMemory, WORKING_MEMORY
from .memory_index import MemoryIndex, MEMORY_INDEX

MEMORY_REGISTRY: dict[str, object] = {
    "episodic": EpisodicMemory(),
    "working": WORKING_MEMORY,
    "index": MEMORY_INDEX,
}
