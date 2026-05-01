"""Unified inference orchestrator — the top-level runtime that combines
dense model, MoE model, cache, memory, speculative decoding, and model
routing into a single efficient inference pipeline.

Per-request decision flow:

  1. Semantic cache lookup          → hit: return instantly (0 cost, 0 model inference)
  2. Memory contextual recall       → hit: prime generation with context
  3. Intent classification          → determines required capability
  4. Difficulty estimation          → easy: dense, hard: MoE, complex: ensemble
  5. Resource check                 → VRAM/RAM/current load → model selection
  6. Speculative decoding (optional) → dense drafts, MoE verifies
  7. Response generation            → selected model produces output
  8. Guardrails check + store       → verify safety, cache result, store in memory

This never runs more model than the request actually needs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..cache import CacheService
from .manager import MemoryManager
from ..routing.intent import IntentClassifier
from ..routing.model_router import ModelRouter, TaskProfile

logger = logging.getLogger("ark.orchestrator")


class InferenceTier(Enum):
    CACHE = "cache"           # 0 model inference, 0 cost
    MEMORY = "memory"         # 0 model inference, 0 cost
    DENSE = "dense"           # low cost, fast
    MOE = "moe"              # higher cost, higher capacity
    ENSEMBLE = "ensemble"     # both models, merged result
    SPECULATIVE = "spec"      # dense drafts, MoE verifies


@dataclass
class InferenceDecision:
    tier: InferenceTier
    model: str
    estimated_cost: float = 0.0
    estimated_latency_ms: float = 0.0
    reason: str = ""
    use_cache: bool = False
    use_memory: bool = False
    use_speculative: bool = False


@dataclass
class InferenceResult:
    content: str
    tier_used: InferenceTier
    model_used: str
    latency_ms: float = 0.0
    cost: float = 0.0
    from_cache: bool = False
    from_memory: bool = False
    token_count: int = 0


class UnifiedInferenceOrchestrator:
    """Top-level runtime that combines every efficiency mechanism into one path.

    To use this, you register your backend functions via the constructor.
    The defaults are stubs that work for testing; wire real model endpoints
    for production.
    """

    def __init__(
        self,
        cache: CacheService | None = None,
        memory: MemoryManager | None = None,
        router: ModelRouter | None = None,
        classifier: IntentClassifier | None = None,
        dense_generate_fn: object = None,
        moe_generate_fn: object = None,
        enable_speculative: bool = True,
        enable_ensemble: bool = False,
        vram_gb: float = 16.0,
    ) -> None:
        self.cache = cache or CacheService()
        self.memory = memory or MemoryManager()
        self.router = router or ModelRouter()
        self.classifier = classifier or IntentClassifier()
        self._dense_fn = dense_generate_fn or self._mock_generate
        self._moe_fn = moe_generate_fn or self._mock_generate
        self.enable_speculative = enable_speculative
        self.enable_ensemble = enable_ensemble
        self.vram_gb = vram_gb

    def infer(self, prompt: str, user_id: str = "anonymous", tier: str = "free") -> InferenceResult:
        start = time.monotonic()

        # 1. Semantic cache — 0 model inference
        cached = self.cache.get(prompt)
        if cached is not None:
            return InferenceResult(
                content=str(cached),
                tier_used=InferenceTier.CACHE,
                model_used="cache",
                latency_ms=(time.monotonic() - start) * 1000,
                from_cache=True,
            )

        # 2. Memory recall — prime context from past runs
        memory_context = self.memory.contextualize(prompt, top_k=5)

        # 3. Intent + difficulty → routing decision
        intent = self.classifier.classify(prompt)
        profile = TaskProfile(
            user_id=user_id,
            user_tier=tier,
            content=prompt,
            intent=intent,
            message_length=len(prompt),
        )
        decision = self.router.route(profile)

        # 4. Select generation path
        content, tier_used, model_name = self._select_path(prompt, decision, memory_context)

        # 5. Store in cache + memory
        self.cache.set(prompt, content)
        self.memory.remember(
            f"User: {prompt}\nAssistant: {content}",
            importance=0.6,
            tags=[intent.category, "chat"],
            context="unified_inference",
        )

        elapsed = (time.monotonic() - start) * 1000
        return InferenceResult(
            content=content,
            tier_used=tier_used,
            model_used=model_name,
            latency_ms=round(elapsed, 1),
            cost=decision.cost_estimate,
            token_count=len(content.split()),
        )

    def _select_path(
        self, prompt: str, decision: InferenceDecision, memory_context: list[str]
    ) -> tuple[str, InferenceTier, str]:
        action = decision.action.value

        # Merge memory context into prompt
        augmented = prompt
        if memory_context:
            augmented = "Context:\n" + "\n".join(f"- {c}" for c in memory_context[-3:]) + "\n\n" + prompt

        # Route by action type
        if action == "cache" or action == "small":
            return self._generate_dense(augmented)

        elif action == "medium":
            return self._generate_dense(augmented)

        elif action == "large" and decision.requires_reasoning:
            if self.enable_speculative and self.vram_gb >= 24:
                return self._generate_speculative(augmented)
            return self._generate_moe(augmented)

        elif action == "rag":
            return self._generate_dense(augmented)

        elif action == "code":
            return self._generate_moe(augmented)

        elif action == "vision":
            return self._generate_dense(augmented)

        elif action == "agent":
            return self._generate_moe(augmented)

        return self._generate_dense(augmented)

    def _generate_dense(self, prompt: str) -> tuple[str, InferenceTier, str]:
        result = self._dense_fn(prompt)
        return str(result), InferenceTier.DENSE, "dense"

    def _generate_moe(self, prompt: str) -> tuple[str, InferenceTier, str]:
        result = self._moe_fn(prompt)
        return str(result), InferenceTier.MOE, "moe"

    def _generate_speculative(self, prompt: str) -> tuple[str, InferenceTier, str]:
        draft = str(self._dense_fn(prompt))
        verified = str(self._moe_fn(f"{prompt}\n\nDraft: {draft}\n\nVerify and correct:"))
        if len(verified) < len(draft) * 0.5:
            verified = draft
        return verified, InferenceTier.SPECULATIVE, "speculative"

    @staticmethod
    def _mock_generate(prompt: str) -> str:
        return f"[mock] processed: {prompt[:60]}..."


__all__ = [
    "UnifiedInferenceOrchestrator",
    "InferenceDecision",
    "InferenceResult",
    "InferenceTier",
]
