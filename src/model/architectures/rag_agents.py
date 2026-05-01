"""RAG/Tool-Use/Agents: RAG, REALM, RETRO, MRKL, ReAct, Toolformer, AutoGen, HuggingGPT.

Papers: Lewis 2020, Guu 2020, Borgeaud 2021, Karpas 2022, Yao 2022, Schick 2023,
Wu 2023, Shen 2023.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

from .registry import register


class RAGRetriever:
    """Retrieval-Augmented Generation retriever (Lewis et al. 2020)."""

    def __init__(self, embed_dim: int = 768, top_k: int = 5) -> None:
        self.top_k = top_k
        self.documents: list[tuple[list[float], str]] = []
        s = 1.0 / math.sqrt(embed_dim)
        self.query_proj = [random.gauss(0, s) for _ in range(embed_dim)]

    def add_document(self, text: str, embedding: list[float] | None = None) -> None:
        self.documents.append(
            (embedding or [random.gauss(0, 0.1) for _ in range(len(self.query_proj))], text)
        )

    def retrieve(self, query_emb: list[float]) -> list[str]:
        projected = [
            sum(self.query_proj[j] * query_emb[j] for j in range(len(query_emb)))
            for _ in range(len(self.query_proj))
        ]
        scored = [
            (
                sum(projected[k] * doc_emb[k] for k in range(len(projected)))
                / (
                    math.sqrt(sum(p**2 for p in projected)) * math.sqrt(sum(d**2 for d in doc_emb))
                    + 1e-8
                ),
                doc_text,
            )
            for doc_emb, doc_text in self.documents
        ]
        scored.sort(key=lambda x: -x[0])
        return [text for _, text in scored[: self.top_k]]


register("rag.retriever", RAGRetriever)


class REALM:
    """REALM: Retrieval-Augmented Language Model Pre-Training (Guu et al. 2020)."""

    def __init__(self, d_model: int = 768) -> None:
        self.retriever = RAGRetriever(d_model)
        from .transformer import BERT

        self.encoder = BERT(30000, d_model)

    def forward(self, input_ids: list[int], query_emb: list[float]) -> list[list[float]]:
        self.retriever.retrieve(query_emb)
        encoded = self.encoder.forward(input_ids)
        return encoded


register("rag.realm", REALM)


class RETRO:
    """RETRO: Retrieval-Enhanced Transformer (Borgeaud et al. 2021)."""

    def __init__(self, d_model: int = 768) -> None:
        self.retriever = RAGRetriever(d_model)
        from .transformer import TransformerBlock

        self.chunked_cross_attn = TransformerBlock(d_model, 12)

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        return self.chunked_cross_attn.forward(x)


register("rag.retro", RETRO)


class MRKLSystem:
    """MRKL: Modular Neuro-Symbolic (Karpas et al. 2022)."""

    def __init__(self) -> None:
        self.modules: dict[str, Callable] = {
            "calculator": lambda x: str(eval(x, {"__builtins__": {}})),  # noqa: S307
            "search": lambda x: f"[search results for: {x}]",
            "translate": lambda x: f"[translation of: {x}]",
        }
        self._router = lambda x: "calculator" if any(c in x for c in "+-*/") else "search"

    def route_and_execute(self, query: str) -> str:
        module = self._router(query)
        fn = self.modules.get(module)
        return fn(query) if fn else f"Unknown module: {module}"

    def add_module(self, name: str, fn: Callable) -> None:
        self.modules[name] = fn


register("rag.mrkl", MRKLSystem)


class ReActAgent:
    """ReAct: Synergizing Reasoning and Acting (Yao et al. 2022)."""

    def __init__(self, tools: dict[str, Callable] | None = None, max_steps: int = 5) -> None:
        self.tools = tools or {"search": lambda q: f"[result: {q}]"}
        self.max_steps = max_steps

    def run(self, task: str) -> list[dict[str, str]]:
        trace: list[dict[str, str]] = []
        thought = f"Starting task: {task}"
        for step in range(self.max_steps):
            trace.append({"role": "thought", "content": thought})
            if "final" in thought.lower() or step == self.max_steps - 1:
                trace.append({"role": "final", "content": thought})
                break
            action = f"search: {task}" if "search" in task else "calculate"
            trace.append({"role": "action", "content": action})
            tool_name = action.split(":")[0].strip()
            tool_input = action.split(":")[1].strip() if ":" in action else ""
            fn = self.tools.get(tool_name)
            result = fn(tool_input) if fn else "tool not found"
            trace.append({"role": "observation", "content": result})
            thought = "Got result, proceeding to final step"
        return trace


register("rag.react", ReActAgent)


class Toolformer:
    """Toolformer (Schick et al. 2023). Self-supervised tool learning."""

    def __init__(self) -> None:
        self.apis: dict[str, Callable] = {}
        self.api_descriptions: dict[str, str] = {}

    def register_api(self, name: str, fn: Callable, description: str = "") -> None:
        self.apis[name] = fn
        self.api_descriptions[name] = description

    def call(self, name: str, **kwargs: Any) -> str:
        fn = self.apis.get(name)
        if fn is None:
            return f"API not found: {name}"
        return str(fn(**kwargs))


register("rag.toolformer", Toolformer)
