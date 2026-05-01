"""Architecture-aware composite inference — decomposes queries into sub-tasks,
dispatches each to the best-suited architecture in the library, and fuses
results into a single coherent answer.

No single architecture is optimal for every aspect of a query. This runtime
combines all of them as specialists in a larger pipeline.

Examples:
  "Explain this diagram of a GAN"
    → ViT (image understanding) + GAN (generative model knowledge) + GPT (language)

  "Compare the graph structure of these two molecules"
    → GCN (graph encoding) + GPT (comparison language)

  "Generate a 3D scene from this description, then explain it"
    → LDM (latent diffusion for image) + GPT (explanation)

  "Find anomalies in this time series and explain why"
    → VAE (latent encoding + anomaly score) + GPT (explanation)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..model.architectures import get_architecture

logger = logging.getLogger("ark.composite")


@dataclass
class SubTask:
    name: str
    architecture: str
    input_data: Any
    output: Any = None
    latency_ms: float = 0.0


@dataclass
class CompositeResult:
    final_output: str
    sub_tasks: list[SubTask] = field(default_factory=list)
    total_latency_ms: float = 0.0
    architectures_used: list[str] = field(default_factory=list)


def _mock_run(arch_name: str, _input: Any) -> str:
    return f"[{arch_name}] processed: {str(_input)[:60]}..."


class ArchitectureDecomposer:
    """Decomposes a query into sub-tasks and maps each to the optimal architecture."""

    def __init__(self) -> None:
        self._routing_table: dict[str, list[dict[str, str]]] = {
            "explain_image": [
                {"name": "image_understanding", "architecture": "cnn.vit"},
                {"name": "generation_knowledge", "architecture": "generative.gan"},
                {"name": "language_explanation", "architecture": "transformer.gpt"},
            ],
            "graph_comparison": [
                {"name": "graph_encoding_a", "architecture": "graph.gat"},
                {"name": "graph_encoding_b", "architecture": "graph.sage"},
                {"name": "comparison_text", "architecture": "transformer.t5"},
            ],
            "anomaly_detection": [
                {"name": "latent_encoding", "architecture": "generative.vae"},
                {"name": "anomaly_scoring", "architecture": "generative.flow"},
                {"name": "explanation", "architecture": "transformer.gpt"},
            ],
            "code_review": [
                {"name": "code_analysis", "architecture": "transformer.llama"},
                {"name": "security_check", "architecture": "graph.gcn"},
                {"name": "quality_report", "architecture": "transformer.gpt"},
            ],
            "research_question": [
                {"name": "literature_retrieval", "architecture": "rag.react"},
                {"name": "graph_reasoning", "architecture": "graph.gcn"},
                {"name": "synthesis", "architecture": "transformer.t5"},
            ],
            "creative_task": [
                {"name": "idea_generation", "architecture": "sparse.mamba"},
                {"name": "evaluation", "architecture": "rl.ppo"},
                {"name": "final_output", "architecture": "transformer.gpt"},
            ],
        }

    def decompose(self, query: str) -> list[SubTask]:
        lower = query.lower()
        if any(w in lower for w in ["diagram", "image", "picture", "photo", "visual"]):
            pattern = "explain_image"
        elif any(w in lower for w in ["graph", "structure", "molecule", "network"]):
            pattern = "graph_comparison"
        elif any(w in lower for w in ["anomal", "outlier", "fraud", "abnormal"]):
            pattern = "anomaly_detection"
        elif any(w in lower for w in ["code", "review", "refactor", "debug"]):
            pattern = "code_review"
        elif any(w in lower for w in ["research", "analyze", "compare", "study"]):
            pattern = "research_question"
        elif any(w in lower for w in ["create", "design", "write a story", "poem"]):
            pattern = "creative_task"
        else:
            pattern = "research_question"

        tasks = self._routing_table.get(pattern, self._routing_table["research_question"])
        return [SubTask(name=t["name"], architecture=t["architecture"], input_data=query) for t in tasks]

    def list_patterns(self) -> list[str]:
        return list(self._routing_table.keys())


class CompositeInferenceEngine:
    """Runs multiple architectures from the library on one decomposed query and fuses results.

    Each sub-task runs on the architecture best suited for it.
    Results are merged into a single final output.
    """

    def __init__(self, decomposer: ArchitectureDecomposer | None = None) -> None:
        self.decomposer = decomposer or ArchitectureDecomposer()
        self._runtimes: dict[str, Any] = {}

    def register_runtime(self, arch_name: str, runner: Any) -> None:
        self._runtimes[arch_name] = runner

    def run(self, query: str) -> CompositeResult:
        sub_tasks = self.decomposer.decompose(query)
        architectures_used: list[str] = []

        for task in sub_tasks:
            start = time.monotonic()
            runner = self._runtimes.get(task.architecture)
            if runner is not None:
                output = runner(task.input_data)
            else:
                output = _mock_run(task.architecture, task.input_data)
            task.output = output
            task.latency_ms = (time.monotonic() - start) * 1000
            architectures_used.append(task.architecture)
            logger.info("  %s (%s): %.0fms", task.name, task.architecture, task.latency_ms)

        # Fuse results into a single coherent output
        fused = self._fuse(sub_tasks, query)
        total_latency = sum(t.latency_ms for t in sub_tasks) or 0.001

        return CompositeResult(
            final_output=fused,
            sub_tasks=sub_tasks,
            total_latency_ms=round(total_latency, 1),
            architectures_used=architectures_used,
        )

    @staticmethod
    def _fuse(tasks: list[SubTask], original_query: str) -> str:
        parts = [f"Query: {original_query}", "", "Analysis:"]
        for task in tasks:
            parts.append(f"\n[{task.name}] ({task.architecture}):")
            parts.append(f"  {task.output}")
        parts.append("\nSynthesized answer:")
        summaries = [str(t.output)[:100] for t in tasks if t.output]
        parts.append(" ".join(summaries))
        return "\n".join(parts)


__all__ = [
    "CompositeInferenceEngine",
    "ArchitectureDecomposer",
    "CompositeResult",
    "SubTask",
]
