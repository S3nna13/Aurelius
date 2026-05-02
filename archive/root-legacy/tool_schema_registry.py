"""tool_schema_registry.py — contract surface for ToolSchemaRegistry.

Contract: Each tool module exposes a set of reusable plugin-style components
with well-defined input/output shapes and forward signatures. The registry
maps module names to their path, version, and contract description.

Live path: tool.* — tool-specific torch.nn.Module subclasses.
"""

from __future__ import annotations
import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegistryEntry:
    name: str
    version: str = "0.1.0"
    contract: str = ""
    live: bool = True
    path: str = ""
    test_command: str = ""


TOOL_SCHEMA_REGISTRY: dict[str, RegistryEntry] = {
    "memory_core": RegistryEntry(
        name="memory_core",
        contract="AurelianMemoryCore, SurpriseGate, Episodic slots, LTSMemory, GraphConsolidator — core associative memory with surprise-based gating",
        path="memory_core",
    ),
    "hierarchical_kv_cache": RegistryEntry(
        name="hierarchical_kv_cache",
        contract="3-tier KV cache with importance-based eviction, MultiScaleAttention — HierarchicalKVCache, ImportanceScorer, LearnedEvictionPolicy",
        path="hierarchical_kv_cache",
    ),
    "moe_memory": RegistryEntry(
        name="moe_memory",
        contract="Mixture-of-Experts memory routing — MoEMemoryRouter, MemoryExpert, TopKSchedule. Archived in archive/",
        path="archive.moe_memory",
    ),
    "ntm_memory": RegistryEntry(
        name="ntm_memory",
        contract="Neural Turing Machine addressing — NTMMemory, NTMReadHead, NTMWriteHead, NTMController. Archived in archive/",
        path="archive.ntm_memory",
    ),
    "speculative_decoding": RegistryEntry(
        name="speculative_decoding",
        contract="Draft-verify loop — SpeculativeDecoder, MemoryAwareDraftModel, MemoryContextProjector",
        path="speculative_decoding",
    ),
    "paged_optimizer": RegistryEntry(
        name="paged_optimizer",
        contract="Paged AdamW with CPU offload — PagedAdamW, PagedOptimizerState, GradientBucket, OptimizerStateCompressor",
        path="paged_optimizer",
    ),
    "fp8_allreduce": RegistryEntry(
        name="fp8_allreduce",
        contract="FP8 gradient compression — FP8AllReduce, FP8Compressor, ErrorFeedbackBuffer, FP8DistributedTrainer",
        path="fp8_allreduce",
    ),
    "rlhf_lora": RegistryEntry(
        name="rlhf_lora",
        contract="LoRA-based RLHF — PPOTrainer, RewardModel, LoraMemoryModel, LoraLayer, MemoryOffloadingRLHFCache",
        path="rlhf_lora",
    ),
    "mobile_inference": RegistryEntry(
        name="mobile_inference",
        contract="On-device quantized inference — MobileQuantizer, InferenceOptimizer, MobileMemoryManager, PrunedHead",
        path="mobile_inference",
    ),
    "async_memory": RegistryEntry(
        name="async_memory",
        contract="Async consolidation pipeline — AsyncConsolidationPipeline, PagedLTSMemory, ConsolidationTask",
        path="async_memory",
    ),
    "deduplication": RegistryEntry(
        name="deduplication",
        contract="Cosine deduplicator — CosineDeduplicator, L2Deduplicator, TemporalDeduplicator, PriorityProportionalAllocator, LZ4MemoryCompressor",
        path="deduplication",
    ),
    "prefetch_router": RegistryEntry(
        name="prefetch_router",
        contract="Predictive memory prefetcher — PredictiveMemoryPrefetcher, SparseLTSRouter, PredictiveAttentionRouter, LTSIndexCompressor",
        path="prefetch_router",
    ),
    "recursive_mas": RegistryEntry(
        name="recursive_mas",
        contract="Recursive multi-agent loop — RecursiveLink, RecursiveAgentWrapper, InnerOuterOptimizer",
        path="recursive_mas",
    ),
    "rust_bridge": RegistryEntry(
        name="rust_bridge",
        contract="Rust FFI bridge — get_page_table, save_checkpoint, load_checkpoint; optional aurelius_memory dependency",
        path="rust_bridge",
    ),
}


def get_registry() -> dict[str, RegistryEntry]:
    return TOOL_SCHEMA_REGISTRY


def lookup(name: str) -> RegistryEntry | None:
    return TOOL_SCHEMA_REGISTRY.get(name)


def verify_imports() -> dict[str, bool]:
    results: dict[str, bool] = {}
    _archive_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'archive')
    if os.path.isdir(_archive_path) and _archive_path not in sys.path:
        sys.path.insert(0, _archive_path)
    for key, entry in TOOL_SCHEMA_REGISTRY.items():
        try:
            importlib.import_module(entry.path)
            results[key] = True
        except ImportError:
            results[key] = False
    return results
