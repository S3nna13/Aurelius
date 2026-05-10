from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

MODEL_ROUTER_REGISTRY: dict[str, type] = {}
from src.routing.intent import Intent, IntentClassifier
from src.runtime.ark_config import ArkConfig


class RouteAction(Enum):
    ROUTE_TO_SMALL = "small"  # Fast, cheap local model
    ROUTE_TO_MEDIUM = "medium"  # Balanced model
    ROUTE_TO_LARGE = "large"  # Powerful cloud model
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
            "gpt-4o": {
                "provider": "openai",
                "cost_per_1k_input": 0.0025,
                "cost_per_1k_output": 0.01,
                "context": 128000,
                "tier": "large",
                "modalities": ["text", "image"],
            },
            "gpt-4o-mini": {
                "provider": "openai",
                "cost_per_1k_input": 0.00015,
                "cost_per_1k_output": 0.0006,
                "context": 128000,
                "tier": "medium",
                "modalities": ["text", "image"],
            },
            "claude-3.5-sonnet": {
                "provider": "anthropic",
                "cost_per_1k_input": 0.003,
                "cost_per_1k_output": 0.015,
                "context": 200000,
                "tier": "large",
                "modalities": ["text"],
            },
            "claude-3-haiku": {
                "provider": "anthropic",
                "cost_per_1k_input": 0.00025,
                "cost_per_1k_output": 0.00125,
                "context": 200000,
                "tier": "small",
                "modalities": ["text"],
            },
            "llama-3.1-8b": {
                "provider": "local",
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
                "context": 32768,
                "tier": "small",
                "modalities": ["text"],
            },
            "llama-3.1-70b": {
                "provider": "local",
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
                "context": 32768,
                "tier": "large",
                "modalities": ["text"],
            },
            "codestral-22b": {
                "provider": "local",
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
                "context": 65536,
                "tier": "code",
                "modalities": ["text"],
            },
            "gpt-4o-vision": {
                "provider": "openai",
                "cost_per_1k_input": 0.005,
                "cost_per_1k_output": 0.015,
                "context": 128000,
                "tier": "vision",
                "modalities": ["text", "image"],
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
        return RoutingDecision(
            action=RouteAction.ROUTE_TO_LARGE,
            model="llama-3.1-70b",
            confidence=0.9,
            reason="Privacy-sensitive — using private local model",
            requires_security_check=True,
            context_length=32768,
            cost_estimate=0.0,
        )

    def _route_vision(self, profile: TaskProfile) -> RoutingDecision:
        model = "gpt-4o-vision"
        info = self._model_catalog.get(model, {})
        return RoutingDecision(
            action=RouteAction.ROUTE_TO_VISION,
            model=model,
            confidence=0.85,
            reason="Multimodal content — routing to vision model",
            requires_retrieval=True,
            context_length=info.get("context", 128000),
            cost_estimate=self._estimate_cost(info, 1000, 500),
        )

    def _route_code(self, profile: TaskProfile) -> RoutingDecision:
        # Code tasks use specialized models
        if profile.user_tier == "enterprise":
            model = "codestral-22b"
        elif profile.user_tier == "pro":
            model = "gpt-4o"
        else:
            model = "gpt-4o-mini"

        info = self._model_catalog.get(model, {})
        return RoutingDecision(
            action=RouteAction.ROUTE_TO_CODE
            if "codestral" in model
            else RouteAction.ROUTE_TO_LARGE,
            model=model,
            confidence=0.8,
            reason=f"Code task — routing to {model}",
            requires_tools=True,
            requires_reasoning=True,
            context_length=info.get("context", 65536),
            max_tokens=4096,
            temperature=0.2,
            cost_estimate=self._estimate_cost(info, 2000, 1000),
        )

    def _route_deep_reasoning(self, profile: TaskProfile) -> RoutingDecision:
        model = "gpt-4o" if profile.user_tier in ("pro", "enterprise") else "gpt-4o-mini"
        info = self._model_catalog.get(model, {})
        return RoutingDecision(
            action=RouteAction.ROUTE_TO_LARGE,
            model=model,
            confidence=0.85,
            reason=f"Deep reasoning — routing to {model} with RAG",
            requires_retrieval=True,
            requires_reasoning=True,
            context_length=info.get("context", 128000),
            max_tokens=4096,
            temperature=0.5,
            cost_estimate=self._estimate_cost(info, 3000, 2000),
        )

    def _route_rag(self, profile: TaskProfile) -> RoutingDecision:
        model = "gpt-4o" if profile.latency_sla_ms > 3000 else "gpt-4o-mini"
        info = self._model_catalog.get(model, {})
        intent = profile.intent
        return RoutingDecision(
            action=RouteAction.ROUTE_TO_RAG,
            model=model,
            confidence=0.75,
            reason=f"Retrieval needed — RAG pipeline with {model}",
            requires_retrieval=True,
            requires_tools=bool(intent and intent.requires_tools),
            context_length=info.get("context", 128000),
            cost_estimate=self._estimate_cost(info, 2000 + len(profile.content) // 4, 1000),
        )

    def _route_default(self, profile: TaskProfile) -> RoutingDecision:
        # Fast path for simple queries → small model
        message_len = len(profile.content)
        if message_len < 100 and profile.user_tier != "enterprise":
            # Try semantic cache
            if self._is_cacheable_simple(profile.content):
                return RoutingDecision(
                    action=RouteAction.ROUTE_TO_CACHE,
                    model="cache",
                    confidence=0.5,
                    reason="Simple query — try cache first",
                    cost_estimate=0.0,
                )

            return RoutingDecision(
                action=RouteAction.ROUTE_TO_SMALL,
                model="gpt-4o-mini",
                confidence=0.7,
                reason="Simple query — small model sufficient",
                cost_estimate=self._estimate_cost({}, max(50, message_len), 100),
            )

        # Pro/Enterprise → better model
        if profile.user_tier in ("pro", "enterprise"):
            model = "gpt-4o" if profile.user_tier == "enterprise" else "gpt-4o-mini"
            info = self._model_catalog.get(model, {})
            return RoutingDecision(
                action=RouteAction.ROUTE_TO_MEDIUM,
                model=model,
                confidence=0.7,
                reason=f"Standard query, tier {profile.user_tier} — routing to {model}",
                cost_estimate=self._estimate_cost(info, message_len, 500),
            )

        return RoutingDecision(
            action=RouteAction.ROUTE_TO_MEDIUM,
            model="gpt-4o-mini",
            confidence=0.6,
            reason="Default routing",
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
