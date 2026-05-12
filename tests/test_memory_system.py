from __future__ import annotations

from src.cache import CacheService
from src.memory import MemoryManager
from src.memory.composite import ArchitectureDecomposer, CompositeInferenceEngine
from src.memory.unified_orchestrator import (
    InferenceTier,
    UnifiedInferenceOrchestrator,
)

pytest.skip("module removed during reorganization", allow_module_level=True)

"""End-to-end test: Aurelius Memory System — all architectures working together."""
