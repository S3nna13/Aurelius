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

# Additive: JSON schema–guided and grammar-constrained output decoders.
# At each decoding step these compute a token mask that allows only tokens
# leading to a valid parse state, enabling reliable structured outputs and
# function-calling API shapes without importing Outlines / LMQL / Guidance.
from .structured_output_decoder import (  # noqa: E402,F401
    GrammarConstrainedDecoder,
    JsonParseState,
    StructuredOutputDecoder,
    TokenTrie,
    STRUCTURED_OUTPUT_REGISTRY,
)

# Merge structured-output decoders into the API shape registry so they are
# discoverable alongside existing validators.
API_SHAPE_REGISTRY.update({
    "structured_output.json_schema": StructuredOutputDecoder,
    "structured_output.grammar": GrammarConstrainedDecoder,
})

# Decoder registry — maps logical decoder names to classes.
DECODER_REGISTRY: dict = {
    "json_schema": StructuredOutputDecoder,
    "grammar": GrammarConstrainedDecoder,
}

# Additive: resilience primitives (circuit breaker). Exposed under a
# dedicated ``RESILIENCE_REGISTRY`` because a circuit breaker is a
# serving-layer resilience primitive, distinct from stream handlers or
# API-shape validators.
from .circuit_breaker import (  # noqa: E402,F401
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    CircuitStateTransition,
)

RESILIENCE_REGISTRY: dict = {
    "circuit_breaker": CircuitBreaker,
}

# --- SSE streaming frame encoder (additive) ---------------------------------
from .sse_stream_encoder import SSEStreamEncoder, split_sse_records  # noqa: E402

STREAM_HANDLER_REGISTRY: dict = {
    "sse": SSEStreamEncoder,
}
