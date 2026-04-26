"""Aurelius model serving infrastructure."""

# Additive: expose OpenAI API shape validators for pre-flight request/response
# checking. Imported lazily-safe (no heavy deps) so adding it here does not
# affect existing module import ordering.
# Additive: composable guardrail middleware (distinct from the existing
# monolithic ``src.serving.guardrails`` pipeline). Wraps a raw ``generate_fn``
# with pre-request validation (jailbreak + prompt-injection heuristics),
# structured logging, and a post-response output safety scan.
from .guardrail_middleware import (  # noqa: E402,F401
    REFUSAL_STRING,
    GuardrailMiddleware,
    MiddlewareDecision,
)
from .openai_api_validator import (  # noqa: E402,F401
    API_SHAPE_REGISTRY,
    APIValidationError,
    OpenAIChatRequestValidator,
    OpenAIChatResponseValidator,
)

# Additive: application-level prompt cache (distinct from the KV-level
# prefix cache in ``src/longcontext/prefix_cache.py``). Caches full
# (prompt, params) -> completion pairs with TTL and LRU eviction so the
# API server can short-circuit repeated identical agent-loop requests.
from .prompt_cache import CachedResponse, PromptCache  # noqa: E402,F401

# Additive: JSON schema–guided and grammar-constrained output decoders.
# At each decoding step these compute a token mask that allows only tokens
# leading to a valid parse state, enabling reliable structured outputs and
# function-calling API shapes without importing Outlines / LMQL / Guidance.
from .structured_output_decoder import (  # noqa: E402,F401
    STRUCTURED_OUTPUT_REGISTRY,
    GrammarConstrainedDecoder,
    JsonParseState,
    StructuredOutputDecoder,
    TokenTrie,
)

# Merge structured-output decoders into the API shape registry so they are
# discoverable alongside existing validators.
API_SHAPE_REGISTRY.update(
    {
        "structured_output.json_schema": StructuredOutputDecoder,
        "structured_output.grammar": GrammarConstrainedDecoder,
    }
)

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
from .sse_stream_encoder import SSEStreamEncoder as SSEStreamEncoder  # noqa: E402
from .sse_stream_encoder import split_sse_records as split_sse_records

STREAM_HANDLER_REGISTRY: dict = {
    "sse": SSEStreamEncoder,
}

# --- OpenAI-compatible SSE chat-completion stream (additive) ---------------
from .sse_chat_stream import (  # noqa: E402,F401
    ChatCompletionChunk,
    ChoiceDelta,
    SSEChatStream,
    SSEParseError,
    parse_sse_event,
)

STREAM_HANDLER_REGISTRY["sse_chat"] = SSEChatStream

# --- Function-calling API shape validator (additive) -----------------------
from .function_calling_api import (  # noqa: E402,F401
    ALLOWED_TYPES as FUNCTION_CALLING_ALLOWED_TYPES,
)
from .function_calling_api import (
    DEFAULT_TOOL_CHOICE as DEFAULT_TOOL_CHOICE,
)
from .function_calling_api import (
    FunctionCallError as FunctionCallError,
)
from .function_calling_api import (
    FunctionCallValidator,
)
from .function_calling_api import (
    FunctionSchema as FunctionSchema,
)
from .function_calling_api import (
    ToolCall as ToolCall,
)
from .function_calling_api import (
    ToolChoice as ToolChoice,
)
from .function_calling_api import (
    ToolDefinition as ToolDefinition,
)

API_SHAPE_REGISTRY["function_calling.validator"] = FunctionCallValidator

# Additive: GPT-5-style agentic Responses API shape (stateful multi-turn,
# tool calls, reasoning chains, SSE streaming events).
from .responses_api import (  # noqa: E402,F401
    RESPONSES_API_REGISTRY,
    ResponsesAPIHandler,
    ResponsesAPIModel,
    ResponsesAPIRequest,
    ResponsesAPIResponse,
    ResponsesAPIValidator,
)

# Register the Responses API handler under the "responses" key so it sits
# alongside all other API-shape validators in the shared registry.
API_SHAPE_REGISTRY["responses"] = ResponsesAPIHandler

# --- Streaming tool-call accumulation (additive) ---------------------------
# --- Aurelius unified frontend server (cycle-212) ----------------------------
from .aurelius_server import (  # noqa: E402,F401
    AureliusHandler,
    AureliusServer,
    create_aurelius_server,
)

# --- Authentication middleware (additive, AUR-SEC-2026-0012) ---------------
# Mitigates Spoofing and Elevation of Privilege on the serving surface by
# requiring callers to present a hashed API key before any request is served.
from .auth_middleware import (  # noqa: E402,F401
    AUTH_MIDDLEWARE_REGISTRY,
    DEFAULT_AUTH_MIDDLEWARE,
    APIKey,
    AuthConfig,
    AuthMiddleware,
    AuthResult,
)

# --- Multi-turn context compression (additive) -----------------------------
from .context_compressor import (  # noqa: E402,F401
    CONTEXT_COMPRESSOR_REGISTRY,
    CompressedTurn,
    CompressionStrategy,
    ContextCompressor,
)

# --- Hermes notification system (cycle-212) ----------------------------------
from .hermes_notifier import (  # noqa: E402,F401
    DEFAULT_HERMES_NOTIFIER,
    HERMES_NOTIFIER_REGISTRY,
    DeliveryResult,
    HermesError,
    HermesNotifier,
    Notification,
)

# --- Load balancer + request queue + response dedup (additive, cycle-147) -----
from .load_balancer import (  # noqa: E402,F401
    LOAD_BALANCER_REGISTRY,
    BackendNode,
    LBStrategy,
    LoadBalancer,
)

# --- Mission Control dashboard (cycle-208) ---------------------------------
from .mission_control import (  # noqa: E402,F401
    ActivityEntry,
    ActivityLog,
    MissionControlHandler,
    MissionControlServer,
    create_mission_control_server,
)

# --- Token-bucket rate limiter (additive, STRIDE-DoS defense) --------------
from .rate_limiter import (  # noqa: E402,F401
    DEFAULT_RATE_LIMITER,
    RATE_LIMIT_REGISTRY,
    RateLimitConfig,
    RateLimiterChain,
    RateLimitResult,
    TokenBucketLimiter,
)

# --- request coalescer (cycle-209 subagent) ----------------------------------
from .request_coalescer import (  # noqa: E402,F401
    REQUEST_COALESCER_REGISTRY,
    RequestCoalescer,
)
from .request_queue import (  # noqa: E402,F401
    REQUEST_QUEUE_REGISTRY,
    QueuedRequest,
    QueuePriority,
    RequestQueue,
)
from .response_dedup import (  # noqa: E402,F401
    RESPONSE_DEDUP_REGISTRY,
    DedupEntry,
    DedupStrategy,
    ResponseDedup,
)
from .tool_call_streaming import (  # noqa: E402,F401
    TOOL_CALL_ACCUMULATOR_REGISTRY,
    ToolCallBuffer,
    ToolCallState,
    ToolCallStreamAccumulator,
)
