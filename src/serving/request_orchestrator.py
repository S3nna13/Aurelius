"""Request orchestrator — single entry point for processing chat requests.

Sequences:
  1. Security scan (prompt injection, jailbreak)
  2. PII / data masking
  3. Routing decision (small vs large vs RAG vs vision...)
  4. Memory recall (contextualization)
  5. RAG retrieval (when ``decision.requires_retrieval`` is set)
  6. Generation
  7. Memory write-back

All collaborators are optional and lazily resolved. Components missing at
import time degrade gracefully with a warning rather than crashing the
serving path.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Best-effort imports. The orchestrator must remain functional even when one
# of the optional subsystems cannot be loaded (e.g. torch unavailable in a
# test environment). Each failed import is logged and the corresponding
# capability is simply skipped at runtime.
try:
    from src.routing.model_router import ModelRouter, RouteAction, RoutingDecision, TaskProfile
except Exception as exc:  # pragma: no cover - defensive
    logger.warning("ModelRouter unavailable: %s", exc)
    ModelRouter = None  # type: ignore[assignment]
    RoutingDecision = None  # type: ignore[assignment]
    RouteAction = None  # type: ignore[assignment]
    TaskProfile = None  # type: ignore[assignment]

try:
    from src.routing.intent import IntentClassifier
except Exception as exc:  # pragma: no cover
    logger.warning("IntentClassifier unavailable: %s", exc)
    IntentClassifier = None  # type: ignore[assignment]

try:
    from src.memory.unified_orchestrator import UnifiedInferenceOrchestrator
except Exception as exc:  # pragma: no cover
    logger.warning("UnifiedInferenceOrchestrator unavailable: %s", exc)
    UnifiedInferenceOrchestrator = None  # type: ignore[assignment]

try:
    from src.retrieval.pipeline import RetrievalPipeline, RetrievalResult
except Exception as exc:  # pragma: no cover
    logger.warning("RetrievalPipeline unavailable: %s", exc)
    RetrievalPipeline = None  # type: ignore[assignment]
    RetrievalResult = None  # type: ignore[assignment]

try:
    from src.safety.prompt_injection_detector import PromptInjectionDetector
except Exception as exc:  # pragma: no cover
    logger.warning("PromptInjectionDetector unavailable: %s", exc)
    PromptInjectionDetector = None  # type: ignore[assignment]

try:
    from src.safety.jailbreak_detector import JailbreakDetector
except Exception as exc:  # pragma: no cover
    logger.warning("JailbreakDetector unavailable: %s", exc)
    JailbreakDetector = None  # type: ignore[assignment]

try:
    from src.security.data_masker import DataMasker
except Exception as exc:  # pragma: no cover
    logger.warning("DataMasker unavailable: %s", exc)
    DataMasker = None  # type: ignore[assignment]


@dataclass
class OrchestratorResult:
    """Aggregate output of :meth:`RequestOrchestrator.process`."""

    content: str
    routing_decision: Any | None = None
    retrieval_result: Any | None = None
    from_cache: bool = False
    security_blocked: bool = False
    latency_ms: float = 0.0
    model_used: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _default_generate_fn(messages: list[dict]) -> str:
    """Fallback generator used when no real backend is wired in.

    Echoes the last user message. Tests rely on this being deterministic.
    """
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    return f"[mock] {last_user}"


class RequestOrchestrator:
    """Single entry point that processes a chat request end-to-end."""

    def __init__(
        self,
        router: Any | None = None,
        rag_pipeline: Any | None = None,
        inference_orchestrator: Any | None = None,
        injection_detector: Any | None = None,
        jailbreak_detector: Any | None = None,
        data_masker: Any | None = None,
        generate_fn: Callable[[list[dict]], str] | None = None,
    ) -> None:
        self._router = router
        self._rag = rag_pipeline
        self._inference = inference_orchestrator
        self._injection = injection_detector
        self._jailbreak = jailbreak_detector
        self._masker = data_masker
        self._generate_fn = generate_fn or _default_generate_fn

    # ------------------------------------------------------------------ #
    # Lazy properties
    # ------------------------------------------------------------------ #

    @property
    def router(self):
        if self._router is None and ModelRouter is not None:
            try:
                self._router = ModelRouter()
            except Exception as exc:  # pragma: no cover
                logger.warning("Lazy ModelRouter init failed: %s", exc)
        return self._router

    @property
    def rag_pipeline(self):
        return self._rag

    @property
    def inference_orchestrator(self):
        return self._inference

    @property
    def injection_detector(self):
        return self._injection

    @property
    def jailbreak_detector(self):
        return self._jailbreak

    @property
    def data_masker(self):
        return self._masker

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _last_user_prompt(messages: list[dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return str(m.get("content") or "")
        # Fall back to the most recent message of any role.
        if messages:
            return str(messages[-1].get("content") or "")
        return ""

    def _security_check(self, prompt: str) -> tuple[bool, str]:
        """Return (blocked, reason)."""
        if self._injection is not None:
            try:
                result = self._injection.detect(prompt)
                if getattr(result, "blocked", False):
                    return True, f"prompt_injection:{[s for s in getattr(result, 'signals', [])]}"
            except Exception as exc:  # pragma: no cover
                logger.warning("Injection detector failed: %s", exc)
        if self._jailbreak is not None:
            try:
                score = self._jailbreak.score(prompt)
                if getattr(score, "is_jailbreak", False):
                    return True, f"jailbreak:{getattr(score, 'triggered_signals', [])}"
            except Exception as exc:  # pragma: no cover
                logger.warning("Jailbreak detector failed: %s", exc)
        return False, ""

    def _mask(self, prompt: str) -> str:
        if self._masker is None:
            return prompt
        try:
            return self._masker.mask(prompt)
        except Exception as exc:  # pragma: no cover
            logger.warning("Data masker failed: %s", exc)
            return prompt

    def _route(self, prompt: str, user_id: str, tier: str):
        router = self.router
        if router is None or TaskProfile is None:
            return None
        try:
            profile = TaskProfile(
                user_id=user_id,
                user_tier=tier,
                content=prompt,
                message_length=len(prompt),
            )
            return router.route(profile)
        except Exception as exc:  # pragma: no cover
            logger.warning("Routing failed: %s", exc)
            return None

    def _recall_memory(self, prompt: str) -> list[str]:
        if self._inference is None:
            return []
        memory = getattr(self._inference, "memory", None)
        if memory is None or not hasattr(memory, "contextualize"):
            return []
        try:
            ctx = memory.contextualize(prompt, top_k=3)
            return [str(c) for c in ctx if c]
        except Exception as exc:  # pragma: no cover
            logger.warning("Memory recall failed: %s", exc)
            return []

    def _retrieve(self, prompt: str):
        if self._rag is None:
            return None
        try:
            return self._rag.run(prompt)
        except Exception as exc:  # pragma: no cover
            logger.warning("RAG retrieval failed: %s", exc)
            return None

    def _remember(self, prompt: str, response: str) -> None:
        if self._inference is None:
            return
        memory = getattr(self._inference, "memory", None)
        if memory is None or not hasattr(memory, "remember"):
            return
        try:
            memory.remember(
                content=f"Q: {prompt}\nA: {response}",
                importance=0.5,
                tags=["chat"],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Memory write-back failed: %s", exc)

    @staticmethod
    def _augment_messages(
        messages: list[dict],
        memory_ctx: list[str],
        retrieval,
    ) -> list[dict]:
        """Prepend a synthesized system message containing recall + RAG context."""
        sections: list[str] = []
        if memory_ctx:
            sections.append("Memory context:\n" + "\n".join(f"- {c}" for c in memory_ctx))
        if retrieval is not None:
            ctx = getattr(retrieval, "compressed_context", "") or ""
            if ctx:
                sections.append("Retrieved context:\n" + ctx)
        if not sections:
            return list(messages)
        system_msg = {"role": "system", "content": "\n\n".join(sections)}
        # Place context-system message before the first message so the
        # caller's own system prompt (if any) still wins on conflict.
        return [system_msg, *messages]

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def process(
        self,
        messages: list[dict],
        user_id: str = "anonymous",
        tier: str = "free",
        model: str = "aurelius",
    ) -> OrchestratorResult:
        start = time.perf_counter()
        prompt = self._last_user_prompt(messages or [])

        # 1. Security
        blocked, reason = self._security_check(prompt)
        if blocked:
            logger.info("Request blocked by security check: %s", reason)
            return OrchestratorResult(
                content="I can't help with that request.",
                routing_decision=None,
                retrieval_result=None,
                from_cache=False,
                security_blocked=True,
                latency_ms=(time.perf_counter() - start) * 1000,
                model_used=model,
                metadata={"security_reason": reason},
            )

        # 2. PII masking
        safe_prompt = self._mask(prompt)
        masked_messages = list(messages)
        if safe_prompt != prompt and masked_messages:
            # Replace the last user message with the masked variant.
            for i in range(len(masked_messages) - 1, -1, -1):
                if masked_messages[i].get("role") == "user":
                    masked_messages[i] = {**masked_messages[i], "content": safe_prompt}
                    break

        # 3. Route
        decision = self._route(safe_prompt, user_id, tier)
        model_used = getattr(decision, "model", "") or model

        # 4. Memory recall
        memory_ctx = self._recall_memory(safe_prompt)

        # 5. Optional RAG
        retrieval = None
        requires_retrieval = bool(getattr(decision, "requires_retrieval", False))
        if requires_retrieval and self._rag is not None:
            retrieval = self._retrieve(safe_prompt)

        # 6. Generate
        augmented = self._augment_messages(masked_messages, memory_ctx, retrieval)
        try:
            content = self._generate_fn(augmented)
        except Exception as exc:
            logger.exception("generate_fn raised: %s", exc)
            content = "[error] generation failed"

        # 7. Persist
        self._remember(safe_prompt, content)

        return OrchestratorResult(
            content=content,
            routing_decision=decision,
            retrieval_result=retrieval,
            from_cache=False,
            security_blocked=False,
            latency_ms=(time.perf_counter() - start) * 1000,
            model_used=model_used,
            metadata={
                "memory_hits": len(memory_ctx),
                "retrieval_chunks": len(getattr(retrieval, "chunks", []) or []),
            },
        )


__all__ = ["RequestOrchestrator", "OrchestratorResult"]
