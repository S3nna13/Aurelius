"""End-to-end test: Aurelius Memory System — all architectures working together."""
from __future__ import annotations

from src.memory.composite import CompositeInferenceEngine, ArchitectureDecomposer
from src.memory.unified_orchestrator import (
    UnifiedInferenceOrchestrator,
    InferenceTier,
)
from src.cache import CacheService
from src.memory import MemoryManager


def test_composite_decomposes_query() -> None:
    decomposer = ArchitectureDecomposer()
    patterns = decomposer.list_patterns()
    assert len(patterns) == 6

    tasks = decomposer.decompose("Explain this diagram of a GAN")
    assert len(tasks) == 3
    names = [t.name for t in tasks]
    assert "image_understanding" in names
    assert "language_explanation" in names


def test_composite_runs_multiple_architectures() -> None:
    engine = CompositeInferenceEngine()
    result = engine.run("Compare the graph structure of these two molecules")
    assert len(result.architectures_used) >= 2
    assert result.total_latency_ms >= 0
    assert len(result.sub_tasks) >= 2


def test_composite_graph_comparison() -> None:
    engine = CompositeInferenceEngine()
    result = engine.run("Compare the graph structure of molecules A and B")
    assert "graph.gat" in result.architectures_used
    assert "graph.sage" in result.architectures_used
    assert "transformer.t5" in result.architectures_used


def test_composite_anomaly_detection() -> None:
    engine = CompositeInferenceEngine()
    result = engine.run("Find anomalies in this time series data")
    assert "generative.vae" in result.architectures_used
    assert "generative.flow" in result.architectures_used


def test_composite_image_explanation() -> None:
    engine = CompositeInferenceEngine()
    result = engine.run("Explain this diagram of a GAN architecture")
    assert "cnn.vit" in result.architectures_used
    assert "generative.gan" in result.architectures_used
    assert "transformer.gpt" in result.architectures_used


def test_unified_orchestrator_cache_hit() -> None:
    cache = CacheService()
    cache.set("what is python", "A programming language")
    orch = UnifiedInferenceOrchestrator(cache=cache)
    result = orch.infer("what is python")
    assert result.from_cache is True
    assert result.tier_used == InferenceTier.CACHE
    assert result.cost == 0.0


def test_unified_orchestrator_dense_path() -> None:
    orch = UnifiedInferenceOrchestrator(enable_speculative=False, enable_ensemble=False)
    result = orch.infer("What is the weather today?")
    assert result.tier_used in (InferenceTier.DENSE, InferenceTier.CACHE)
    assert result.latency_ms >= 0


def test_unified_orchestrator_memory_integration() -> None:
    memory = MemoryManager()
    memory.remember("Python is a programming language", tags=["python"])
    orch = UnifiedInferenceOrchestrator(memory=memory, enable_speculative=False)
    result = orch.infer("Tell me about Python")
    assert result.content is not None


def test_unified_orchestrator_reasoning_path() -> None:
    memory = MemoryManager()
    cache = CacheService()
    orch = UnifiedInferenceOrchestrator(
        memory=memory, cache=cache,
        enable_speculative=False, enable_ensemble=False,
        vram_gb=32,
    )
    result = orch.infer(
        "Analyze the implications of quantum computing on cryptography",
        user_id="pro_user",
        tier="pro",
    )
    assert result.model_used == "moe" or result.tier_used == InferenceTier.DENSE


def test_full_pipeline_no_errors() -> None:
    """Smoke test: everything wired together produces a result without raising."""
    cache = CacheService()
    memory = MemoryManager()
    orch = UnifiedInferenceOrchestrator(cache=cache, memory=memory)
    result = orch.infer("Hello world")
    assert result.content is not None
    assert isinstance(result.latency_ms, float)


def test_cost_tracking() -> None:
    orch = UnifiedInferenceOrchestrator(cache=CacheService())
    result = orch.infer("hi")
    assert result.cost >= 0.0
