"""Aurelius model serving infrastructure."""

# Additive: expose OpenAI API shape validators for pre-flight request/response
# checking. Imported lazily-safe (no heavy deps) so adding it here does not
# affect existing module import ordering.
from .openai_api_validator import (  # noqa: E402,F401
    APIValidationError,
    OpenAIChatRequestValidator,
    OpenAIChatResponseValidator,
    API_SHAPE_REGISTRY,
)

# Additive: application-level prompt cache (distinct from the KV-level
# prefix cache in ``src/longcontext/prefix_cache.py``). Caches full
# (prompt, params) -> completion pairs with TTL and LRU eviction so the
# API server can short-circuit repeated identical agent-loop requests.
from .prompt_cache import CachedResponse, PromptCache  # noqa: E402,F401

# Additive: composable guardrail middleware (distinct from the existing
# monolithic ``src.serving.guardrails`` pipeline). Wraps a raw ``generate_fn``
# with pre-request validation (jailbreak + prompt-injection heuristics),
# structured logging, and a post-response output safety scan.
from .guardrail_middleware import (  # noqa: E402,F401
    GuardrailMiddleware,
    MiddlewareDecision,
    REFUSAL_STRING,
)
