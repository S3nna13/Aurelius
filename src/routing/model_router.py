from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.routing.intent import Intent, IntentClassifier
from src.runtime.ark_config import ArkConfig

MODEL_ROUTER_REGISTRY: dict[str, type] = {}

DEFAULT_SMALL_MODEL = "aurelius-1b"
DEFAULT_MEDIUM_MODEL = "aurelius-3b"
DEFAULT_LARGE_MODEL = "aurelius-7b"
DEFAULT_PRIVATE_MODEL = "aurelius-32b"
DEFAULT_VISION_MODEL = "aurelius-vision"
DEFAULT_CODE_MODEL = "codestral-22b"


class RouteAction(Enum):
    ROUTE_TO_SMALL = "small"  # Fast, cheap local model
    ROUTE_TO_MEDIUM = "medium"  # Balanced model
    ROUTE_TO_LARGE = "large"  # Powerful local frontier model
    ROUTE_TO_VISION = "vision"  # Multimodal model
    ROUTE_TO_CODE = "code"  # Code-specialized model
    ROUTE_TO_RAG = "rag"  # Retrieval pipeline
    ROUTE_TO_AGENT = "agent"  # Multi-agent orchestration
    ROUTE_TO_CACHE = "cache"  # Semantic cache hit
    ROUTE_FALLBACK = "fallback"  # Degraded service
    BLOCK = "block"  # Security block


@dataclass
class RoutingDecision:
    action: RouteAction
    model: str
    confidence: float
    reason: str
    requires_retrieval: bool = False
    requires_tools: bool = False
    requires_reasoning: bool = False
    requires_security_check: bool = False
    context_length: int = 4096
    max_tokens: int = 2048
    temperature: float = 0.7
    cost_estimate: float = 0.0


@dataclass
class TaskProfile:
    """Rich task profile for routing decisions."""

    user_id: str
    user_tier: str = "free"  # free | pro | enterprise
    content: str = ""
    intent: Intent | None = None
    message_length: int = 0
    has_code: bool = False
    has_pii: bool = False
    has_image: bool = False
    has_file: bool = False
    history_length: int = 0
    cost_budget_remaining: float = float("inf")
    latency_sla_ms: float = 5000.0


class ModelRouter:
    """Intelligent model router that decides which model/tool handles each request.

    Considers:
    - Task difficulty (intent, reasoning depth)
    - Required latency (user tier, SLA)
    - Cost budget
    - Privacy sensitivity
    - Required context length
    - Modality requirements
    - Current system load
    - Available models and their capabilities

    Router examples:
    - FAQ → small local model + cache
    - Complex analysis → large model + RAG + citations
    - Code debugging → code model + tool sandbox
    - Sensitive query → private model + private retrieval only
    - Multimodal → vision-language model
    """

    def __init__(
        self,
        config: ArkConfig | None = None,
        intent_classifier: IntentClassifier | None = None,
        model_catalog: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or ArkConfig.load()
        self.intent = intent_classifier or IntentClassifier()
        self._model_catalog = model_catalog or self._default_catalog()

    @staticmethod
    def _default_catalog() -> dict[str, dict[str, Any]]:
        return {
            DEFAULT_SMALL_MODEL: {
                "provider": "local",
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
                "context": 4096,
                "tier": "small",
                "modalities": ["text"],
            },
            DEFAULT_MEDIUM_MODEL: {
                "provider": "local",
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
                "context": 8192,
                "tier": "medium",
                "modalities": ["text"],
            },
            DEFAULT_LARGE_MODEL: {
                "provider": "local",
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
                "context": 8192,
                "tier": "large",
                "modalities": ["text"],
            },
            DEFAULT_PRIVATE_MODEL: {
                "provider": "local",
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
                "context": 32768,
                "tier": "large",
                "modalities": ["text"],
            },
            DEFAULT_VISION_MODEL: {
                "provider": "local",
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
                "context": 32768,
                "tier": "vision",
                "modalities": ["text", "image"],
            },
            DEFAULT_CODE_MODEL: {
                "provider": "local",
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
                "context": 65536,
                "tier": "code",
                "modalities": ["text"],
            },
        }

    def route(self, profile: TaskProfile) -> RoutingDecision:
        """Make a routing decision based on the task profile."""
        if profile.intent is None:
            profile.intent = self.intent.classify(profile.content)

        intent = profile.intent

        # 1. Security: block or redirect sensitive content
        if intent.privacy_sensitive and profile.user_tier == "free":
            return RoutingDecision(
                action=RouteAction.BLOCK,
                model="",
                confidence=0.95,
                reason="Privacy-sensitive query requires enterprise tier",
                requires_security_check=True,
            )

        # 2. Privacy-sensitive → private model only
        if intent.privacy_sensitive:
            return self._route_private(profile)

        # 3. Cache check (fast path)
        if self._is_cacheable(intent):
            return RoutingDecision(
                action=RouteAction.ROUTE_TO_CACHE,
                model="cache",
                confidence=0.6,
                reason="FAQ/greeting — try cache first",
                cost_estimate=0.0,
            )

        # 4. Multimodal → vision model
        if intent.requires_multimodal or profile.has_image:
            return self._route_vision(profile)

        # 5. Code tasks → code model
        if profile.has_code or intent.category == "code":
            return self._route_code(profile)

        # 6. Deep reasoning → large model + RAG
        if intent.reasoning_depth == "high" or intent.requires_reasoning:
            return self._route_deep_reasoning(profile)

        # 7. Retrieval needed → RAG pipeline
        if intent.requires_retrieval or intent.requires_tools:
            return self._route_rag(profile)

        # 8. Default: route by difficulty and user tier
        return self._route_default(profile)

    def _route_private(self, profile: TaskProfile) -> RoutingDecision:
        info = self._model_catalog.get(DEFAULT_PRIVATE_MODEL, {})
        return RoutingDecision(
            action=RouteAction.ROUTE_TO_LARGE,
            model=DEFAULT_PRIVATE_MODEL,
            confidence=0.9,
            reason="Privacy-sensitive — using private local model",
            requires_security_check=True,
            context_length=info.get("context", 32768),
            cost_estimate=self._estimate_cost(info, 1000, 500),
        )

    def _route_vision(self, profile: TaskProfile) -> RoutingDecision:
        model = DEFAULT_VISION_MODEL
        info = self._model_catalog.get(model, {})
        return RoutingDecision(
            action=RouteAction.ROUTE_TO_VISION,
            model=model,
            confidence=0.85,
            reason="Multimodal content — routing to local vision model",
            requires_retrieval=True,
            context_length=info.get("context", 32768),
            cost_estimate=self._estimate_cost(info, 1000, 500),
        )

    def _route_code(self, profile: TaskProfile) -> RoutingDecision:
        model = DEFAULT_CODE_MODEL
        info = self._model_catalog.get(model, {})
        return RoutingDecision(
            action=RouteAction.ROUTE_TO_CODE,
            model=model,
            confidence=0.8,
            reason=f"Code task — routing to local code model {model}",
            requires_tools=True,
            requires_reasoning=True,
            context_length=info.get("context", 65536),
            max_tokens=4096,
            temperature=0.2,
            cost_estimate=self._estimate_cost(info, 2000, 1000),
        )

    def _route_deep_reasoning(self, profile: TaskProfile) -> RoutingDecision:
        model = DEFAULT_PRIVATE_MODEL if profile.user_tier == "enterprise" else DEFAULT_LARGE_MODEL
        info = self._model_catalog.get(model, {})
        return RoutingDecision(
            action=RouteAction.ROUTE_TO_LARGE,
            model=model,
            confidence=0.85,
            reason=f"Deep reasoning — routing to {model} with RAG",
            requires_retrieval=True,
            requires_reasoning=True,
            context_length=info.get("context", 8192 if model == DEFAULT_LARGE_MODEL else 32768),
            max_tokens=4096,
            temperature=0.5,
            cost_estimate=self._estimate_cost(info, 3000, 2000),
        )

    def _route_rag(self, profile: TaskProfile) -> RoutingDecision:
        if profile.user_tier == "enterprise":
            model = DEFAULT_PRIVATE_MODEL
        elif profile.latency_sla_ms > 3000:
            model = DEFAULT_LARGE_MODEL
        else:
            model = DEFAULT_MEDIUM_MODEL
        info = self._model_catalog.get(model, {})
        intent = profile.intent
        return RoutingDecision(
            action=RouteAction.ROUTE_TO_RAG,
            model=model,
            confidence=0.75,
            reason=f"Retrieval needed — RAG pipeline with local model {model}",
            requires_retrieval=True,
            requires_tools=bool(intent and intent.requires_tools),
            context_length=info.get(
                "context", 8192 if model in {DEFAULT_MEDIUM_MODEL, DEFAULT_LARGE_MODEL} else 32768
            ),
            cost_estimate=self._estimate_cost(info, 2000 + len(profile.content) // 4, 1000),
        )

    def _route_default(self, profile: TaskProfile) -> RoutingDecision:
        # Fast path for simple queries → small model
        message_len = len(profile.content)
        if message_len < 100:
            # Try semantic cache
            if self._is_cacheable_simple(profile.content):
                return RoutingDecision(
                    action=RouteAction.ROUTE_TO_CACHE,
                    model="cache",
                    confidence=0.5,
                    reason="Simple query — try cache first",
                    cost_estimate=0.0,
                )

            if profile.user_tier == "enterprise":
                model = DEFAULT_PRIVATE_MODEL
                action = RouteAction.ROUTE_TO_LARGE
            elif profile.user_tier == "pro":
                model = DEFAULT_MEDIUM_MODEL
                action = RouteAction.ROUTE_TO_MEDIUM
            else:
                model = DEFAULT_SMALL_MODEL
                action = RouteAction.ROUTE_TO_SMALL

            info = self._model_catalog.get(model, {})
            return RoutingDecision(
                action=action,
                model=model,
                confidence=0.7,
                reason=f"Simple query — local model sufficient ({model})",
                context_length=info.get("context", 4096),
                cost_estimate=self._estimate_cost(info, max(50, message_len), 100),
            )

        # Pro/Enterprise → better model
        if profile.user_tier in ("pro", "enterprise"):
            model = (
                DEFAULT_PRIVATE_MODEL if profile.user_tier == "enterprise" else DEFAULT_LARGE_MODEL
            )
            info = self._model_catalog.get(model, {})
            return RoutingDecision(
                action=RouteAction.ROUTE_TO_LARGE
                if profile.user_tier == "enterprise"
                else RouteAction.ROUTE_TO_MEDIUM,
                model=model,
                confidence=0.7,
                reason=f"Standard query, tier {profile.user_tier} — routing to local model {model}",
                cost_estimate=self._estimate_cost(info, message_len, 500),
            )

        return RoutingDecision(
            action=RouteAction.ROUTE_TO_MEDIUM,
            model=DEFAULT_MEDIUM_MODEL,
            confidence=0.6,
            reason="Default routing — local medium model",
        )

    @staticmethod
    def _is_cacheable(intent: Intent) -> bool:
        return intent.category in ("greeting", "faq") and intent.confidence > 0.6

    @staticmethod
    def _is_cacheable_simple(content: str) -> bool:
        simple_patterns = ("hi", "hello", "hey", "thanks", "ok", "bye", "goodbye")
        return content.lower().strip() in simple_patterns or len(content.split()) <= 3

    @staticmethod
    def _estimate_cost(model_info: dict[str, Any], input_tokens: int, output_tokens: int) -> float:
        if not model_info:
            return 0.0
        input_cost = model_info.get("cost_per_1k_input", 0) * input_tokens / 1000
        output_cost = model_info.get("cost_per_1k_output", 0) * output_tokens / 1000
        return round(input_cost + output_cost, 6)

    def list_available_models(self) -> dict[str, dict[str, Any]]:
        return dict(self._model_catalog)


# Register the router on import
MODEL_ROUTER_REGISTRY["default"] = ModelRouter
